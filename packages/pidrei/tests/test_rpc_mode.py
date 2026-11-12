"""Mirrors pi coding-agent test/rpc-prompt-response-semantics.test.ts and
test/rpc.test.ts.

pi's rpc.test.ts spawns the real CLI behind describe.skipIf(!API_KEY); the
pidrei mirror drives run_rpc_mode in-process against canned stream
functions over a real AgentSessionRuntime, exactly like the
prompt-response-semantics suite mocks output-guard and the JSONL reader.
The export_html command asserts the documented Phase 3 deviation (HTML
export lands with the Phase 4 theme system).
"""

import contextlib
import json
import os
import signal
import threading
from typing import Any

import pytest
import tonio.colored as tonio

from pidrei.core.agent_session_runtime import AgentSessionRuntime, CreateAgentSessionRuntimeResult
from pidrei.core.agent_session_services import AgentSessionServices
from pidrei.core.session_manager import SessionManager
from pidrei.modes.rpc import rpc_mode
from pidrei_ai.types import DoneEvent, Model, ModelCost, StartEvent
from pidrei_ai.utils.event_stream import AssistantMessageEventStream
from pidrei_utils.cancel import CancelToken

from .agent_session_helpers import create_agent_session, create_assistant_message, create_test_resource_loader
from .harness import create_test_extensions_result
from .test_agent_session_compaction import _make_llm_stream_fn


@contextlib.contextmanager
def _patched(module, **attrs):
    saved = {name: getattr(module, name) for name in attrs}
    for name, value in attrs.items():
        setattr(module, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(module, name, value)


def _stream_fn(release: tonio.Event | None = None, streaming: tonio.Event | None = None):
    """Canned stream that finishes at once, or once `release` is set: a test
    that must act while the response is still streaming holds it open with a
    gate (and waits for `streaming`) instead of racing a fixed delay."""

    async def stream_fn(_model, _context, _options=None):
        stream = AssistantMessageEventStream()
        stream.push(StartEvent(partial=create_assistant_message("")))
        if streaming is not None:
            streaming.set()

        async def finish() -> None:
            if release is not None:
                await release.wait()
            stream.push(DoneEvent(reason="stop", message=create_assistant_message("done")))

        tonio.spawn.without_tracking(finish())
        return stream

    return stream_fn


def _fake_model() -> Model:
    return Model(
        id="fake-model",
        name="Fake Model",
        api="openai-completions",
        provider="fake-provider",
        base_url="https://example.invalid",
        reasoning=False,
        input=[],
        cost=ModelCost(),
        context_window=0,
        max_tokens=0,
    )


class _FakeSignalDispatcher:
    """Captures the mode's signal handlers, so a test delivers a signal by awaiting one."""

    def __init__(self):
        self.handlers: dict[int, Any] = {}

    def set_handler(self, signum: int, handler) -> None:
        self.handlers[signum] = handler


class _RpcHarness:
    def __init__(self):
        self.output_lines: list[str] = []
        self._waiters_lock = threading.Lock()
        self._waiters: list[tuple[Any, tonio.Event]] = []
        self.line_handler = None
        self.end_handler = None
        self._stop = tonio.Event()
        self._ready = tonio.Event()
        self._exit = None
        self._handle = None
        self.runtime_host = None
        self.signal_dispatcher = _FakeSignalDispatcher()
        # `hard_exit` calls: (exit code, whether the slow dispose had been released by then).
        self.exits: list[tuple[int, bool | None]] = []
        self._exited = tonio.Event()
        self.dispose_started = tonio.Event()
        self.dispose_release: tonio.Event | None = None
        self.dispose_calls = 0
        self._spawned: list = []

    def _hard_exit(self, code: int) -> None:
        """Stands in for `hard_exit`: records the exit as an output record, so its order
        relative to the responses can be asserted."""
        self.exits.append((code, self.dispose_release.is_set() if self.dispose_release is not None else None))
        self._write(json.dumps({"type": "exit", "code": code}) + "\n")
        self._exited.set()

    def slow_dispose(self) -> None:
        """Make the runtime's dispose wait for `dispose_release` (pi's mocked dispose)."""
        self.dispose_release = tonio.Event()

    def run(self, coro) -> tonio.Event:
        """Spawn a call into the mode (its stdin-end callback, a signal handler); the
        returned Event is set when it returns. `stop()` awaits it."""
        done = tonio.Event()

        async def run() -> None:
            try:
                await coro
            finally:
                done.set()

        self._spawned.append(tonio.spawn(run()))
        return done

    def records(self) -> list[dict[str, Any]]:
        parsed = []
        for chunk in self.output_lines:
            for line in chunk.split("\n"):
                if line.strip():
                    parsed.append(json.loads(line))
        return parsed

    def responses(self, request_id: str, command: str) -> list[dict[str, Any]]:
        return [
            record
            for record in self.records()
            if record.get("id") == request_id and record.get("type") == "response" and record.get("command") == command
        ]

    def settled(self) -> bool:
        return any(record.get("type") == "agent_settled" for record in self.records())

    def _write(self, chunk: str) -> None:
        """Stands in for `write_raw_stdout`: every RPC record arrives here, so
        waiters are re-checked on each write instead of polled."""
        self.output_lines.append(chunk)
        with self._waiters_lock:
            waiters = list(self._waiters)
        for predicate, reached in waiters:
            if not reached.is_set() and predicate():
                reached.set()

    async def until(self, predicate, timeout: float = 5.0) -> None:
        reached = tonio.Event()
        waiter = (predicate, reached)
        with self._waiters_lock:
            self._waiters.append(waiter)
        # Registered first, then checked: a write landing in between is not missed.
        if predicate():
            reached.set()
        await reached.wait(timeout)
        with self._waiters_lock:
            self._waiters.remove(waiter)
        assert reached.is_set(), "timed out waiting for the RPC output"

    def send(self, command: dict[str, Any]) -> None:
        assert self.line_handler is not None
        self.line_handler(json.dumps(command))

    async def request(self, command: dict[str, Any], timeout: float = 5.0) -> dict[str, Any]:
        request_id = command["id"]
        self.send(command)
        await self.until(lambda: len(self.responses(request_id, command["type"])) > 0, timeout)
        return self.responses(request_id, command["type"])[0]

    async def start(self, runtime_host) -> None:
        self.runtime_host = runtime_host
        real_dispose = runtime_host.dispose

        async def dispose() -> None:
            self.dispose_calls += 1
            self.dispose_started.set()
            if self.dispose_release is not None:
                await self.dispose_release.wait()
            await real_dispose()

        runtime_host.dispose = dispose

        async def fake_pump(_stdin, on_line, on_end) -> None:
            self.line_handler = on_line
            self.end_handler = on_end
            self._ready.set()
            await self._stop.wait()

        async def _noop_async() -> None:
            pass

        self._exit = contextlib.ExitStack()
        self._exit.enter_context(
            _patched(
                rpc_mode,
                take_over_stdout=lambda: None,
                write_raw_stdout=self._write,
                drain_output=_noop_async,
                _pump_stdin_commands=fake_pump,
                hard_exit=self._hard_exit,
                kill_tracked_detached_children=lambda: None,
            )
        )
        # The fake pump stands in for the only reader of stdin.
        self._handle = tonio.spawn(
            rpc_mode.run_rpc_mode(runtime_host, stdin=None, signal_dispatcher=self.signal_dispatcher)
        )
        await self._ready.wait(5)
        assert self._ready.is_set(), "the RPC mode never attached its input"

    async def stop(self) -> None:
        # A started shutdown ends in `hard_exit`: let it reach the stand-in before the real
        # one is restored, also when the test failed midway.
        if self.dispose_release is not None:
            self.dispose_release.set()
        if self.dispose_started.is_set():
            await self._exited.wait(5)
        for spawned in self._spawned:
            await spawned
        self._stop.set()
        if self._handle is not None:
            await self._handle
        if self._exit is not None:
            self._exit.close()
        session = self.runtime_host.session if self.runtime_host is not None else None
        if session is not None:
            with contextlib.suppress(Exception):
                if session.is_streaming:
                    await session.abort()
            session.dispose()


async def _create_runtime_host(
    temp_dir: str,
    *,
    stream_fn,
    with_auth: bool = True,
    model: Model | None = None,
    persisted: bool = False,
    settings_overrides: dict | None = None,
    extension_factories: list | None = None,
) -> AgentSessionRuntime:
    """Real AgentSessionRuntime whose factory builds sessions from the shared
    hermetic helper, so new_session/fork/switch flow through real code."""
    temp_dir = str(temp_dir)
    session_dir = os.path.join(temp_dir, "sessions")
    resource_loader = (
        create_test_resource_loader(await create_test_extensions_result(extension_factories, temp_dir))
        if extension_factories
        else None
    )

    async def create_runtime(
        *,
        cwd: str,
        agent_dir: str,
        session_manager: SessionManager,
        session_start_event: dict[str, Any] | None = None,
        project_trust_context: Any = None,
    ) -> CreateAgentSessionRuntimeResult:
        session = await create_agent_session(
            temp_dir,
            stream_fn=stream_fn,
            session_manager=session_manager,
            model=model,
            provider_auth="anthropic" if with_auth else None,
            settings_overrides=settings_overrides,
            resource_loader=resource_loader,
        )
        services = AgentSessionServices(
            cwd=cwd,
            agent_dir=agent_dir,
            model_runtime=session.model_runtime,
            settings_manager=session.settings_manager,
            resource_loader=session.resource_loader,
        )
        return CreateAgentSessionRuntimeResult(session=session, services=services)

    initial_session_manager = (
        await SessionManager(temp_dir, session_dir) if persisted else SessionManager.in_memory(temp_dir)
    )
    result = await create_runtime(
        cwd=temp_dir, agent_dir=temp_dir, session_manager=initial_session_manager, session_start_event=None
    )
    return AgentSessionRuntime(result.session, result.services, create_runtime)


def _read_session_entries(session_file: str) -> list[dict[str, Any]]:
    with open(session_file, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle.read().strip().split("\n")]


class TestRpcPromptResponseSemantics:
    @pytest.mark.tonio
    async def test_emits_one_failure_response_when_prompt_preflight_rejects(self, tmp_path):
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_path, stream_fn=_stream_fn(), with_auth=False, model=_fake_model())
        await harness.start(host)
        try:
            harness.send({"id": "b1", "type": "prompt", "message": "Hello"})

            await harness.until(lambda: len(harness.responses("b1", "prompt")) == 1)
            response = harness.responses("b1", "prompt")[0]
            assert response["success"] is False
            assert (
                "No API key found for fake-provider.\n\nUse /login to log into a provider via OAuth or API key. See:"
                in response["error"]
            )
        finally:
            await harness.stop()

    @pytest.mark.tonio
    async def test_emits_one_success_response_when_prompt_preflight_succeeds(self, tmp_path):
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_path, stream_fn=_stream_fn())
        await harness.start(host)
        try:
            harness.send({"id": "b2", "type": "prompt", "message": "Hello"})

            await harness.until(lambda: len(harness.responses("b2", "prompt")) == 1)
            response = harness.responses("b2", "prompt")[0]
            assert response["success"] is True
            await host.session.wait_for_idle()
        finally:
            await harness.stop()

    @pytest.mark.tonio
    async def test_emits_one_success_response_when_prompt_is_queued_during_streaming(self, tmp_path):
        harness = _RpcHarness()
        release = tonio.Event()
        streaming = tonio.Event()
        # A gate, not a fixed delay, keeps the first response streaming until
        # the follow-up has been answered.
        host = await _create_runtime_host(tmp_path, stream_fn=_stream_fn(release, streaming))
        await harness.start(host)
        try:
            harness.send({"id": "b3-start", "type": "prompt", "message": "Start"})
            await harness.until(lambda: len(harness.responses("b3-start", "prompt")) == 1)
            await streaming.wait(5)
            assert streaming.is_set()

            harness.output_lines.clear()
            harness.send({"id": "b3", "type": "prompt", "message": "Queue this", "streamingBehavior": "followUp"})

            await harness.until(lambda: len(harness.responses("b3", "prompt")) == 1)
            response = harness.responses("b3", "prompt")[0]
            assert response["success"] is True

            release.set()
            await host.session.wait_for_idle()
        finally:
            release.set()
            await harness.stop()

    @pytest.mark.tonio
    async def test_returns_and_clears_queued_steering_and_follow_up_messages(self, tmp_path):
        # Mirrors the clear_queue case pi added to rpc-prompt-response-semantics.test.ts
        # (the rest of that suite is a recorded parity gap).
        harness = _RpcHarness()
        release = tonio.Event()
        streaming = tonio.Event()
        # A gate, not a fixed delay, keeps the first response streaming while
        # the queue is filled and cleared.
        host = await _create_runtime_host(tmp_path, stream_fn=_stream_fn(release, streaming))
        await harness.start(host)
        try:
            harness.send({"id": "clear-start", "type": "prompt", "message": "Start"})
            await harness.until(lambda: len(harness.responses("clear-start", "prompt")) == 1)
            await streaming.wait(5)
            assert streaming.is_set()

            harness.send(
                {
                    "id": "clear-steering",
                    "type": "prompt",
                    "message": "Change direction",
                    "streamingBehavior": "steer",
                }
            )
            await harness.until(lambda: len(harness.responses("clear-steering", "prompt")) == 1)

            harness.send(
                {
                    "id": "clear-follow-up",
                    "type": "prompt",
                    "message": "Summarize when finished",
                    "streamingBehavior": "followUp",
                }
            )
            await harness.until(lambda: len(harness.responses("clear-follow-up", "prompt")) == 1)

            response = await harness.request({"id": "clear", "type": "clear_queue"})
            assert response == {
                "id": "clear",
                "type": "response",
                "command": "clear_queue",
                "success": True,
                "data": {
                    "steering": ["Change direction"],
                    "followUp": ["Summarize when finished"],
                },
            }

            # Let the run finish and settle: with the queue cleared nothing
            # follows it.
            release.set()
            await harness.until(harness.settled)
            await host.session.wait_for_idle()
            assert len([record for record in harness.records() if record.get("type") == "agent_start"]) == 1
        finally:
            release.set()
            await harness.stop()


def _exit_order(harness: _RpcHarness, request_id: str) -> list[str]:
    """pi's `parseOutputLines(...).map(line => line.id === id ? "response" : line.type)`."""
    return [
        "response" if record.get("id") == request_id else record.get("type")
        for record in harness.records()
        if record.get("id") == request_id or record.get("type") == "exit"
    ]


def _capture_shutdown(captured: list):
    """An extension that keeps its session_start context, to request a shutdown later."""

    async def factory(pi) -> None:
        async def on_session_start(_event, ctx):
            captured.append(ctx)

        pi.on("session_start", on_session_start)

    return factory


class TestRpcExtensionShutdownRequests:
    """pi's "RPC extension shutdown requests" describe. `process.exit` is the
    harness's `hard_exit` stand-in, which records an `exit` output record."""

    # #11599: an async ctx.shutdown() while idle must not wait for another stdin command.
    @pytest.mark.tonio
    async def test_shuts_down_an_idle_session_when_an_extension_requests_shutdown_asynchronously(self, tmp_path):
        harness = _RpcHarness()
        contexts: list = []
        host = await _create_runtime_host(
            tmp_path, stream_fn=_stream_fn(), with_auth=False, extension_factories=[_capture_shutdown(contexts)]
        )
        await harness.start(host)
        try:
            assert contexts, "session_start never ran"
            assert harness.exits == []
            contexts[-1].shutdown()
            await harness.until(lambda: len(harness.exits) > 0)
            assert harness.exits == [(0, None)]
        finally:
            await harness.stop()

    @pytest.mark.tonio
    async def test_writes_the_command_response_before_shutting_down_for_a_shutdown_requested_by_an_extension_command(
        self, tmp_path
    ):
        harness = _RpcHarness()

        async def factory(pi) -> None:
            async def quit_now(_args, ctx):
                ctx.shutdown()

            pi.register_command("quit-now", handler=quit_now)

        host = await _create_runtime_host(
            tmp_path, stream_fn=_stream_fn(), with_auth=False, extension_factories=[factory]
        )
        await harness.start(host)
        try:
            harness.send({"id": "q", "type": "prompt", "message": "/quit-now"})
            await harness.until(lambda: len(harness.exits) > 0)
            assert len(harness.exits) == 1
            assert _exit_order(harness, "q") == ["response", "exit"]
        finally:
            await harness.stop()

    # pi requests the shutdown and sends the command in one tick, before the scheduled idle
    # check runs. Here the check is spawned, so the order is made explicit: the command has
    # started (an Event its handler sets) when the shutdown is requested.
    @pytest.mark.tonio
    async def test_waits_for_a_command_that_arrives_before_an_idle_shutdown_request_is_processed(self, tmp_path):
        harness = _RpcHarness()
        contexts: list = []
        slow_started = tonio.Event()
        slow_release = tonio.Event()

        async def factory(pi) -> None:
            async def slow(_args, _ctx):
                slow_started.set()
                await slow_release.wait()

            pi.register_command("slow", handler=slow)

        host = await _create_runtime_host(
            tmp_path,
            stream_fn=_stream_fn(),
            with_auth=False,
            extension_factories=[_capture_shutdown(contexts), factory],
        )
        await harness.start(host)
        try:
            harness.send({"id": "s", "type": "prompt", "message": "/slow"})
            await slow_started.wait(5)
            assert slow_started.is_set(), "the command never started"
            contexts[-1].shutdown()
            slow_release.set()
            await harness.until(lambda: len(harness.exits) > 0)
            assert len(harness.exits) == 1
            assert _exit_order(harness, "s") == ["response", "exit"]
        finally:
            slow_release.set()
            await harness.stop()

    # The line reaches the mode before the shutdown is requested, but nothing waits for the
    # command to start: the mode counts it when the line is handed over, so the spawned idle
    # check sees it. The handler holds the response until the request has had its chance.
    @pytest.mark.tonio
    async def test_waits_for_a_delivered_command_that_has_not_started_when_an_idle_shutdown_is_requested(
        self, tmp_path
    ):
        harness = _RpcHarness()
        contexts: list = []
        slow_started = tonio.Event()
        slow_release = tonio.Event()

        async def factory(pi) -> None:
            async def slow(_args, _ctx):
                slow_started.set()
                await slow_release.wait()

            pi.register_command("slow", handler=slow)

        host = await _create_runtime_host(
            tmp_path,
            stream_fn=_stream_fn(),
            with_auth=False,
            extension_factories=[_capture_shutdown(contexts), factory],
        )
        await harness.start(host)
        try:
            harness.send({"id": "s", "type": "prompt", "message": "/slow"})
            contexts[-1].shutdown()
            await slow_started.wait(5)
            assert slow_started.is_set(), "the command never started"
            slow_release.set()
            await harness.until(lambda: len(harness.exits) > 0)
            assert len(harness.exits) == 1
            assert _exit_order(harness, "s") == ["response", "exit"]
        finally:
            slow_release.set()
            await harness.stop()


class TestRpcShutdownTriggers:
    """pi's "RPC shutdown triggers" describe. pi emits stdin `end` and SIGTERM on the
    process and sleeps before asserting "not exited yet"; here the mode's stdin-end
    callback and its installed signal handler are awaited directly, and each exit records
    whether the slow dispose had been released by then."""

    @staticmethod
    async def _start_with_slow_dispose(tmp_path) -> _RpcHarness:
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_path, stream_fn=_stream_fn(), with_auth=False)
        await harness.start(host)
        harness.slow_dispose()
        return harness

    @staticmethod
    async def _returned(done: tonio.Event) -> None:
        await done.wait(5)
        assert done.is_set(), "the call never returned"

    @staticmethod
    def _signal(harness: _RpcHarness) -> tonio.Event:
        return harness.run(harness.signal_dispatcher.handlers[signal.SIGTERM](signal.SIGTERM))

    # fcatuhe's report on #11599: Paseo closes stdin and sends SIGTERM together.
    @pytest.mark.tonio
    async def test_lets_a_shutdown_started_by_stdin_eof_finish_when_one_sigterm_follows(self, tmp_path):
        harness = await self._start_with_slow_dispose(tmp_path)
        try:
            end = harness.run(harness.end_handler())
            await harness.dispose_started.wait(5)
            assert harness.dispose_started.is_set()
            # The signal handler stays installed during the shutdown.
            assert signal.SIGTERM in harness.signal_dispatcher.handlers
            await self._returned(self._signal(harness))
            assert harness.exits == []

            harness.dispose_release.set()
            await self._returned(end)
            assert harness.exits == [(0, True)]
        finally:
            await harness.stop()

    @pytest.mark.tonio
    async def test_lets_a_shutdown_started_by_sigterm_finish_when_stdin_eof_follows(self, tmp_path):
        harness = await self._start_with_slow_dispose(tmp_path)
        try:
            signalled = self._signal(harness)
            await harness.dispose_started.wait(5)
            assert harness.dispose_started.is_set()
            end = harness.run(harness.end_handler())

            harness.dispose_release.set()
            await self._returned(signalled)
            await self._returned(end)
            assert harness.exits == [(143, True)]
            assert harness.dispose_calls == 1
        finally:
            await harness.stop()

    @pytest.mark.tonio
    async def test_exits_at_once_on_a_repeated_signal_during_shutdown(self, tmp_path):
        harness = await self._start_with_slow_dispose(tmp_path)
        try:
            end = harness.run(harness.end_handler())
            await harness.dispose_started.wait(5)
            assert harness.dispose_started.is_set()
            await self._returned(self._signal(harness))
            await self._returned(self._signal(harness))
            assert harness.exits == [(143, False)]

            harness.dispose_release.set()
            await self._returned(end)
        finally:
            await harness.stop()


class TestRpcMode:
    @pytest.mark.tonio
    async def test_should_get_state(self, tmp_path):
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_path, stream_fn=_stream_fn())
        await harness.start(host)
        try:
            response = await harness.request({"id": "r1", "type": "get_state"})
            state = response["data"]
            assert state["model"]["provider"] == "anthropic"
            assert state["model"]["id"] == "claude-sonnet-4-5"
            assert state["isStreaming"] is False
            assert state["messageCount"] == 0
        finally:
            await harness.stop()

    @pytest.mark.tonio
    async def test_should_save_messages_to_session_file(self, tmp_path):
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_path, stream_fn=_stream_fn(), persisted=True)
        await harness.start(host)
        try:
            harness.send({"id": "p1", "type": "prompt", "message": "Reply with just the word 'hello'"})
            await harness.until(harness.settled)

            message_end_events = [r for r in harness.records() if r.get("type") == "message_end"]
            assert len(message_end_events) >= 2  # user + assistant

            session_file = host.session.session_file
            assert session_file is not None
            # No polling: message persistence is awaited by the session's
            # event listener, which the run finishes before `agent_settled`.
            assert os.path.exists(session_file)

            entries = _read_session_entries(session_file)
            assert entries[0]["type"] == "session"

            messages = [e for e in entries if e["type"] == "message"]
            assert len(messages) >= 2
            roles = [m["message"]["role"] for m in messages]
            assert "user" in roles
            assert "assistant" in roles
        finally:
            await harness.stop()

    @pytest.mark.tonio
    async def test_should_handle_manual_compaction(self, tmp_path):
        harness = _RpcHarness()
        stream_fn, _state = _make_llm_stream_fn()
        host = await _create_runtime_host(
            tmp_path,
            stream_fn=stream_fn,
            persisted=True,
            settings_overrides={"compaction": {"keepRecentTokens": 1}},
        )
        await harness.start(host)
        try:
            harness.send({"id": "p1", "type": "prompt", "message": "Say hello"})
            await harness.until(harness.settled)

            response = await harness.request({"id": "c1", "type": "compact"})
            assert response["success"] is True
            assert response["data"]["summary"]
            assert response["data"]["tokensBefore"] > 0

            session_file = host.session.session_file
            entries = _read_session_entries(session_file)
            compaction_entries = [e for e in entries if e["type"] == "compaction"]
            assert len(compaction_entries) == 1
            assert compaction_entries[0]["summary"]
        finally:
            await harness.stop()

    @pytest.mark.tonio
    async def test_should_execute_bash_command(self, tmp_path):
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_path, stream_fn=_stream_fn())
        await harness.start(host)
        try:
            response = await harness.request({"id": "b1", "type": "bash", "command": "echo hello"})
            result = response["data"]
            assert result["output"].strip() == "hello"
            assert result["exitCode"] == 0
            assert result["cancelled"] is False
        finally:
            await harness.stop()

    @pytest.mark.tonio
    async def test_should_add_bash_output_to_context(self, tmp_path):
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_path, stream_fn=_stream_fn(), persisted=True)
        await harness.start(host)
        try:
            harness.send({"id": "p1", "type": "prompt", "message": "Say hi"})
            await harness.until(harness.settled)

            unique_value = "test-bash-context-value"
            await harness.request({"id": "b1", "type": "bash", "command": f"echo {unique_value}"})

            session_file = host.session.session_file

            def bash_entries() -> list[dict[str, Any]]:
                if not os.path.exists(session_file):
                    return []
                return [
                    e
                    for e in _read_session_entries(session_file)
                    if e["type"] == "message" and e.get("message", {}).get("role") == "bashExecution"
                ]

            # Read at once: the run has settled, so `execute_bash` persists the
            # result directly (`record_bash_result` awaits the append) before
            # the bash response is written.
            bash_messages = bash_entries()
            assert len(bash_messages) == 1
            assert unique_value in bash_messages[0]["message"]["output"]
        finally:
            await harness.stop()

    @pytest.mark.tonio
    async def test_should_set_and_get_thinking_level(self, tmp_path):
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_path, stream_fn=_stream_fn())
        await harness.start(host)
        try:
            response = await harness.request({"id": "t1", "type": "set_thinking_level", "level": "high"})
            assert response["success"] is True

            state = (await harness.request({"id": "t2", "type": "get_state"}))["data"]
            assert state["thinkingLevel"] == "high"
        finally:
            await harness.stop()

    @pytest.mark.tonio
    async def test_should_cycle_thinking_level(self, tmp_path):
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_path, stream_fn=_stream_fn())
        await harness.start(host)
        try:
            initial_state = (await harness.request({"id": "t1", "type": "get_state"}))["data"]

            result = (await harness.request({"id": "t2", "type": "cycle_thinking_level"}))["data"]
            assert result is not None
            assert result["level"] != initial_state["thinkingLevel"]

            new_state = (await harness.request({"id": "t3", "type": "get_state"}))["data"]
            assert new_state["thinkingLevel"] == result["level"]
        finally:
            await harness.stop()

    @pytest.mark.tonio
    async def test_should_get_available_thinking_levels(self, tmp_path):
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_path, stream_fn=_stream_fn())
        await harness.start(host)
        try:
            levels = (await harness.request({"id": "t1", "type": "get_available_thinking_levels"}))["data"]["levels"]
            assert len(levels) > 0

            state = (await harness.request({"id": "t2", "type": "get_state"}))["data"]
            assert state["thinkingLevel"] in levels

            initial_level = state["thinkingLevel"]
            cycled = (await harness.request({"id": "t3", "type": "cycle_thinking_level"}))["data"]
            if cycled is not None:
                assert cycled["level"] in levels
                if len(levels) > 1:
                    assert cycled["level"] != initial_level
        finally:
            await harness.stop()

    @pytest.mark.tonio
    async def test_should_get_available_models(self, tmp_path):
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_path, stream_fn=_stream_fn())
        await harness.start(host)
        try:
            models = (await harness.request({"id": "m1", "type": "get_available_models"}))["data"]["models"]
            assert len(models) > 0

            for model in models:
                assert model["provider"]
                assert model["id"]
                assert model["contextWindow"] > 0
                assert isinstance(model["reasoning"], bool)
        finally:
            await harness.stop()

    @pytest.mark.tonio
    async def test_should_get_session_stats(self, tmp_path):
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_path, stream_fn=_stream_fn(), persisted=True)
        await harness.start(host)
        try:
            harness.send({"id": "p1", "type": "prompt", "message": "Hello"})
            await harness.until(harness.settled)

            stats = (await harness.request({"id": "s1", "type": "get_session_stats"}))["data"]
            assert stats["sessionFile"]
            assert stats["sessionId"]
            assert stats["userMessages"] >= 1
            assert stats["assistantMessages"] >= 1
        finally:
            await harness.stop()

    @pytest.mark.tonio
    async def test_should_create_new_session(self, tmp_path):
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_path, stream_fn=_stream_fn(), persisted=True)
        await harness.start(host)
        try:
            harness.send({"id": "p1", "type": "prompt", "message": "Hello"})
            await harness.until(harness.settled)

            state = (await harness.request({"id": "s1", "type": "get_state"}))["data"]
            assert state["messageCount"] > 0

            response = await harness.request({"id": "n1", "type": "new_session"})
            assert response["data"]["cancelled"] is False

            state = (await harness.request({"id": "s2", "type": "get_state"}))["data"]
            assert state["messageCount"] == 0
        finally:
            await harness.stop()

    @pytest.mark.tonio
    async def test_export_html_exports_the_session(self, tmp_path):
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_path, stream_fn=_stream_fn(), persisted=True)
        await harness.start(host)
        try:
            harness.send({"id": "p1", "type": "prompt", "message": "hello"})
            await harness.until(harness.settled)

            output_path = str(tmp_path / "export.html")
            response = await harness.request({"id": "e1", "type": "export_html", "outputPath": output_path})
            assert response["success"] is True
            assert response["data"]["path"] == output_path
            with open(output_path, encoding="utf-8") as f:
                assert "<!DOCTYPE html>" in f.read(200)
        finally:
            await harness.stop()

    @pytest.mark.tonio
    async def test_should_get_last_assistant_text(self, tmp_path):
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_path, stream_fn=_stream_fn(), persisted=True)
        await harness.start(host)
        try:
            text = (await harness.request({"id": "l1", "type": "get_last_assistant_text"}))["data"]["text"]
            assert text is None

            harness.send({"id": "p1", "type": "prompt", "message": "Reply with just: test123"})
            await harness.until(harness.settled)

            text = (await harness.request({"id": "l2", "type": "get_last_assistant_text"}))["data"]["text"]
            assert "done" in text
        finally:
            await harness.stop()

    @pytest.mark.tonio
    async def test_should_get_session_entries_with_since_cursor(self, tmp_path):
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_path, stream_fn=_stream_fn(), persisted=True)
        await harness.start(host)
        try:
            harness.send({"id": "p1", "type": "prompt", "message": "Reply with just 'ok'"})
            await harness.until(harness.settled)

            data = (await harness.request({"id": "e1", "type": "get_entries"}))["data"]
            entries = data["entries"]
            leaf_id = data["leafId"]
            assert len(entries) >= 2  # user + assistant
            for entry in entries:
                assert entry["id"]
            assert leaf_id == entries[-1]["id"]

            since = (await harness.request({"id": "e2", "type": "get_entries", "since": entries[0]["id"]}))["data"]
            assert [e["id"] for e in since["entries"]] == [e["id"] for e in entries[1:]]
            assert since["leafId"] == leaf_id

            missing = await harness.request({"id": "e3", "type": "get_entries", "since": "nonexistent-id"})
            assert missing["success"] is False
            assert "Entry not found" in missing["error"]
        finally:
            await harness.stop()

    @pytest.mark.tonio
    async def test_should_get_session_tree(self, tmp_path):
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_path, stream_fn=_stream_fn(), persisted=True)
        await harness.start(host)
        try:
            harness.send({"id": "p1", "type": "prompt", "message": "Reply with just 'ok'"})
            await harness.until(harness.settled)

            entries_data = (await harness.request({"id": "e1", "type": "get_entries"}))["data"]
            tree_data = (await harness.request({"id": "t1", "type": "get_tree"}))["data"]
            assert tree_data["leafId"] == entries_data["leafId"]

            # Single root whose chain matches the entries
            tree = tree_data["tree"]
            assert len(tree) == 1
            chain_ids = []
            nodes = tree
            while len(nodes) == 1:
                chain_ids.append(nodes[0]["entry"]["id"])
                nodes = nodes[0]["children"]
            assert len(nodes) == 0
            assert chain_ids == [e["id"] for e in entries_data["entries"]]
        finally:
            await harness.stop()

    @pytest.mark.tonio
    async def test_should_retain_pre_compaction_entries_in_get_entries(self, tmp_path):
        harness = _RpcHarness()
        stream_fn, _state = _make_llm_stream_fn()
        host = await _create_runtime_host(
            tmp_path,
            stream_fn=stream_fn,
            persisted=True,
            settings_overrides={"compaction": {"keepRecentTokens": 1}},
        )
        await harness.start(host)
        try:
            harness.send({"id": "p1", "type": "prompt", "message": "Reply with just 'ok'"})
            await harness.until(harness.settled)
            before = (await harness.request({"id": "e1", "type": "get_entries"}))["data"]

            await harness.request({"id": "c1", "type": "compact"})

            after = (await harness.request({"id": "e2", "type": "get_entries"}))["data"]
            # Append-only: pre-compaction entries are still there, in the same order
            before_ids = [e["id"] for e in before["entries"]]
            assert [e["id"] for e in after["entries"][: len(before_ids)]] == before_ids
            assert any(e["type"] == "compaction" for e in after["entries"])
        finally:
            await harness.stop()

    @pytest.mark.tonio
    async def test_should_set_and_get_session_name(self, tmp_path):
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_path, stream_fn=_stream_fn(), persisted=True)
        await harness.start(host)
        try:
            state = (await harness.request({"id": "s1", "type": "get_state"}))["data"]
            assert "sessionName" not in state

            # Session files are only written after the first assistant message
            harness.send({"id": "p1", "type": "prompt", "message": "Reply with just 'ok'"})
            await harness.until(harness.settled)

            await harness.request({"id": "n1", "type": "set_session_name", "name": "my-test-session"})

            state = (await harness.request({"id": "s2", "type": "get_state"}))["data"]
            assert state["sessionName"] == "my-test-session"

            session_file = host.session.session_file
            # The session_info append is awaited before the set_session_name
            # response.
            assert os.path.exists(session_file)
            entries = _read_session_entries(session_file)
            session_info_entries = [e for e in entries if e["type"] == "session_info"]
            assert len(session_info_entries) == 1
            assert session_info_entries[0]["name"] == "my-test-session"
        finally:
            await harness.stop()


def _ui_requests(harness: _RpcHarness, method: str) -> list[dict[str, Any]]:
    return [
        record
        for record in harness.records()
        if record.get("type") == "extension_ui_request" and record.get("method") == method
    ]


class TestRpcExtensionUi:
    """pidrei-only: the extension UI options reach the RPC client and bound the wait."""

    @pytest.mark.tonio
    async def test_a_dialog_timeout_is_sent_and_ends_the_wait(self, tmp_path):
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_path, stream_fn=_stream_fn())
        await harness.start(host)
        try:
            ui = host.session.extension_runner.get_ui_context()
            answer = await ui.select("Pick", ["a", "b"], {"timeout": 50})

            assert answer is None
            assert _ui_requests(harness, "select")[0]["timeout"] == 50
        finally:
            await harness.stop()

    @pytest.mark.tonio
    async def test_a_cancelled_signal_ends_the_wait(self, tmp_path):
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_path, stream_fn=_stream_fn())
        await harness.start(host)
        try:
            ui = host.session.extension_runner.get_ui_context()
            signal = CancelToken()
            handle = ui.confirm("Sure?", "Really?", {"signal": signal})
            await harness.until(lambda: len(_ui_requests(harness, "confirm")) == 1)
            signal.cancel()

            assert await handle is False
        finally:
            await harness.stop()

    @pytest.mark.tonio
    async def test_widget_placement_is_sent(self, tmp_path):
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_path, stream_fn=_stream_fn())
        await harness.start(host)
        try:
            host.session.extension_runner.get_ui_context().set_widget("w", ["line"], {"placement": "belowEditor"})

            assert _ui_requests(harness, "setWidget")[0]["widgetPlacement"] == "belowEditor"
        finally:
            await harness.stop()
