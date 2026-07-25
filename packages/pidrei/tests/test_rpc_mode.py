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
from typing import Any

import pytest
import tonio.colored as tonio

from pidrei.core.agent_session_runtime import AgentSessionRuntime, CreateAgentSessionRuntimeResult
from pidrei.core.agent_session_services import AgentSessionServices
from pidrei.core.session_manager import SessionManager
from pidrei.modes.rpc import rpc_mode
from pidrei_ai.types import DoneEvent, Model, ModelCost, StartEvent
from pidrei_ai.utils.event_stream import AssistantMessageEventStream

from .agent_session_helpers import create_agent_session, create_assistant_message
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


async def _wait_for(predicate, timeout: float = 5.0) -> None:
    waited = 0.0
    while not predicate():
        if waited >= timeout:
            raise AssertionError("timed out waiting for condition")
        await tonio.time.sleep(0.01)
        waited += 0.01


def _delayed_stream_fn(delay_s: float):
    def stream_fn(_model, _context, _options=None):
        stream = AssistantMessageEventStream()
        stream.push(StartEvent(partial=create_assistant_message("")))

        async def finish() -> None:
            if delay_s > 0:
                await tonio.time.sleep(delay_s)
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


class _RpcHarness:
    def __init__(self):
        self.output_lines: list[str] = []
        self.line_handler = None
        self._stop = tonio.Event()
        self._ready = tonio.Event()
        self._exit = None
        self._handle = None
        self.runtime_host = None

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

    def send(self, command: dict[str, Any]) -> None:
        assert self.line_handler is not None
        self.line_handler(json.dumps(command))

    async def request(self, command: dict[str, Any], timeout: float = 5.0) -> dict[str, Any]:
        request_id = command["id"]
        self.send(command)
        await _wait_for(lambda: len(self.responses(request_id, command["type"])) > 0, timeout)
        return self.responses(request_id, command["type"])[0]

    async def start(self, runtime_host) -> None:
        self.runtime_host = runtime_host

        async def fake_pump(on_line, _on_end) -> None:
            self.line_handler = on_line
            self._ready.set()
            await self._stop.wait()

        async def _noop_async() -> None:
            pass

        self._exit = contextlib.ExitStack()
        self._exit.enter_context(
            _patched(
                rpc_mode,
                take_over_stdout=lambda: None,
                write_raw_stdout=self.output_lines.append,
                wait_for_raw_stdout_backpressure=_noop_async,
                flush_raw_stdout=_noop_async,
                _pump_stdin_commands=fake_pump,
            )
        )
        self._handle = tonio.spawn(rpc_mode.run_rpc_mode(runtime_host))
        await self._ready.wait()

    async def stop(self) -> None:
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
) -> AgentSessionRuntime:
    """Real AgentSessionRuntime whose factory builds sessions from the shared
    hermetic helper, so new_session/fork/switch flow through real code."""
    temp_dir = str(temp_dir)
    session_dir = os.path.join(temp_dir, "sessions")

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
        SessionManager.create(temp_dir, session_dir) if persisted else SessionManager.in_memory(temp_dir)
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
    async def test_emits_one_failure_response_when_prompt_preflight_rejects(self, tmp_dir):
        harness = _RpcHarness()
        host = await _create_runtime_host(
            tmp_dir, stream_fn=_delayed_stream_fn(0), with_auth=False, model=_fake_model()
        )
        await harness.start(host)
        try:
            harness.send({"id": "b1", "type": "prompt", "message": "Hello"})

            await _wait_for(lambda: len(harness.responses("b1", "prompt")) == 1)
            response = harness.responses("b1", "prompt")[0]
            assert response["success"] is False
            assert (
                "No API key found for fake-provider.\n\nUse /login to log into a provider via OAuth or API key. See:"
                in response["error"]
            )
        finally:
            await harness.stop()

    @pytest.mark.tonio
    async def test_emits_one_success_response_when_prompt_preflight_succeeds(self, tmp_dir):
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_dir, stream_fn=_delayed_stream_fn(0))
        await harness.start(host)
        try:
            harness.send({"id": "b2", "type": "prompt", "message": "Hello"})

            await _wait_for(lambda: len(harness.responses("b2", "prompt")) == 1)
            response = harness.responses("b2", "prompt")[0]
            assert response["success"] is True
            await host.session.wait_for_idle()
        finally:
            await harness.stop()

    @pytest.mark.tonio
    async def test_emits_one_success_response_when_prompt_is_queued_during_streaming(self, tmp_dir):
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_dir, stream_fn=_delayed_stream_fn(0.1))
        await harness.start(host)
        try:
            harness.send({"id": "b3-start", "type": "prompt", "message": "Start"})
            await _wait_for(lambda: len(harness.responses("b3-start", "prompt")) == 1)

            harness.output_lines.clear()
            harness.send({"id": "b3", "type": "prompt", "message": "Queue this", "streamingBehavior": "followUp"})

            await _wait_for(lambda: len(harness.responses("b3", "prompt")) == 1)
            response = harness.responses("b3", "prompt")[0]
            assert response["success"] is True

            await tonio.time.sleep(0.15)
            await host.session.wait_for_idle()
        finally:
            await harness.stop()


class TestRpcMode:
    @pytest.mark.tonio
    async def test_should_get_state(self, tmp_dir):
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_dir, stream_fn=_delayed_stream_fn(0))
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
    async def test_should_save_messages_to_session_file(self, tmp_dir):
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_dir, stream_fn=_delayed_stream_fn(0), persisted=True)
        await harness.start(host)
        try:
            harness.send({"id": "p1", "type": "prompt", "message": "Reply with just the word 'hello'"})
            await _wait_for(lambda: any(r.get("type") == "agent_settled" for r in harness.records()))

            message_end_events = [r for r in harness.records() if r.get("type") == "message_end"]
            assert len(message_end_events) >= 2  # user + assistant

            session_file = host.session.session_file
            assert session_file is not None
            await _wait_for(lambda: os.path.exists(session_file))

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
    async def test_should_handle_manual_compaction(self, tmp_dir):
        harness = _RpcHarness()
        stream_fn, _state = _make_llm_stream_fn()
        host = await _create_runtime_host(
            tmp_dir,
            stream_fn=stream_fn,
            persisted=True,
            settings_overrides={"compaction": {"keepRecentTokens": 1}},
        )
        await harness.start(host)
        try:
            harness.send({"id": "p1", "type": "prompt", "message": "Say hello"})
            await _wait_for(lambda: any(r.get("type") == "agent_settled" for r in harness.records()))

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
    async def test_should_execute_bash_command(self, tmp_dir):
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_dir, stream_fn=_delayed_stream_fn(0))
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
    async def test_should_add_bash_output_to_context(self, tmp_dir):
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_dir, stream_fn=_delayed_stream_fn(0), persisted=True)
        await harness.start(host)
        try:
            harness.send({"id": "p1", "type": "prompt", "message": "Say hi"})
            await _wait_for(lambda: any(r.get("type") == "agent_settled" for r in harness.records()))

            unique_value = "test-bash-context-value"
            await harness.request({"id": "b1", "type": "bash", "command": f"echo {unique_value}"})

            session_file = host.session.session_file
            await _wait_for(lambda: os.path.exists(session_file))
            entries = _read_session_entries(session_file)
            bash_messages = [
                e for e in entries if e["type"] == "message" and e.get("message", {}).get("role") == "bashExecution"
            ]
            assert len(bash_messages) == 1
            assert unique_value in bash_messages[0]["message"]["output"]
        finally:
            await harness.stop()

    @pytest.mark.tonio
    async def test_should_set_and_get_thinking_level(self, tmp_dir):
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_dir, stream_fn=_delayed_stream_fn(0))
        await harness.start(host)
        try:
            response = await harness.request({"id": "t1", "type": "set_thinking_level", "level": "high"})
            assert response["success"] is True

            state = (await harness.request({"id": "t2", "type": "get_state"}))["data"]
            assert state["thinkingLevel"] == "high"
        finally:
            await harness.stop()

    @pytest.mark.tonio
    async def test_should_cycle_thinking_level(self, tmp_dir):
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_dir, stream_fn=_delayed_stream_fn(0))
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
    async def test_should_get_available_thinking_levels(self, tmp_dir):
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_dir, stream_fn=_delayed_stream_fn(0))
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
    async def test_should_get_available_models(self, tmp_dir):
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_dir, stream_fn=_delayed_stream_fn(0))
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
    async def test_should_get_session_stats(self, tmp_dir):
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_dir, stream_fn=_delayed_stream_fn(0), persisted=True)
        await harness.start(host)
        try:
            harness.send({"id": "p1", "type": "prompt", "message": "Hello"})
            await _wait_for(lambda: any(r.get("type") == "agent_settled" for r in harness.records()))

            stats = (await harness.request({"id": "s1", "type": "get_session_stats"}))["data"]
            assert stats["sessionFile"]
            assert stats["sessionId"]
            assert stats["userMessages"] >= 1
            assert stats["assistantMessages"] >= 1
        finally:
            await harness.stop()

    @pytest.mark.tonio
    async def test_should_create_new_session(self, tmp_dir):
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_dir, stream_fn=_delayed_stream_fn(0), persisted=True)
        await harness.start(host)
        try:
            harness.send({"id": "p1", "type": "prompt", "message": "Hello"})
            await _wait_for(lambda: any(r.get("type") == "agent_settled" for r in harness.records()))

            state = (await harness.request({"id": "s1", "type": "get_state"}))["data"]
            assert state["messageCount"] > 0

            response = await harness.request({"id": "n1", "type": "new_session"})
            assert response["data"]["cancelled"] is False

            state = (await harness.request({"id": "s2", "type": "get_state"}))["data"]
            assert state["messageCount"] == 0
        finally:
            await harness.stop()

    @pytest.mark.tonio
    async def test_export_html_reports_phase3_deviation(self, tmp_dir):
        # pi exports to HTML here; pidrei's export-html lands with the Phase 4
        # theme system, so the RPC command answers with an error response.
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_dir, stream_fn=_delayed_stream_fn(0))
        await harness.start(host)
        try:
            response = await harness.request({"id": "e1", "type": "export_html"})
            assert response["success"] is False
            assert "HTML export is not available yet" in response["error"]
        finally:
            await harness.stop()

    @pytest.mark.tonio
    async def test_should_get_last_assistant_text(self, tmp_dir):
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_dir, stream_fn=_delayed_stream_fn(0), persisted=True)
        await harness.start(host)
        try:
            text = (await harness.request({"id": "l1", "type": "get_last_assistant_text"}))["data"]["text"]
            assert text is None

            harness.send({"id": "p1", "type": "prompt", "message": "Reply with just: test123"})
            await _wait_for(lambda: any(r.get("type") == "agent_settled" for r in harness.records()))

            text = (await harness.request({"id": "l2", "type": "get_last_assistant_text"}))["data"]["text"]
            assert "done" in text
        finally:
            await harness.stop()

    @pytest.mark.tonio
    async def test_should_get_session_entries_with_since_cursor(self, tmp_dir):
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_dir, stream_fn=_delayed_stream_fn(0), persisted=True)
        await harness.start(host)
        try:
            harness.send({"id": "p1", "type": "prompt", "message": "Reply with just 'ok'"})
            await _wait_for(lambda: any(r.get("type") == "agent_settled" for r in harness.records()))

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
    async def test_should_get_session_tree(self, tmp_dir):
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_dir, stream_fn=_delayed_stream_fn(0), persisted=True)
        await harness.start(host)
        try:
            harness.send({"id": "p1", "type": "prompt", "message": "Reply with just 'ok'"})
            await _wait_for(lambda: any(r.get("type") == "agent_settled" for r in harness.records()))

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
    async def test_should_retain_pre_compaction_entries_in_get_entries(self, tmp_dir):
        harness = _RpcHarness()
        stream_fn, _state = _make_llm_stream_fn()
        host = await _create_runtime_host(
            tmp_dir,
            stream_fn=stream_fn,
            persisted=True,
            settings_overrides={"compaction": {"keepRecentTokens": 1}},
        )
        await harness.start(host)
        try:
            harness.send({"id": "p1", "type": "prompt", "message": "Reply with just 'ok'"})
            await _wait_for(lambda: any(r.get("type") == "agent_settled" for r in harness.records()))
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
    async def test_should_set_and_get_session_name(self, tmp_dir):
        harness = _RpcHarness()
        host = await _create_runtime_host(tmp_dir, stream_fn=_delayed_stream_fn(0), persisted=True)
        await harness.start(host)
        try:
            state = (await harness.request({"id": "s1", "type": "get_state"}))["data"]
            assert "sessionName" not in state

            # Session files are only written after the first assistant message
            harness.send({"id": "p1", "type": "prompt", "message": "Reply with just 'ok'"})
            await _wait_for(lambda: any(r.get("type") == "agent_settled" for r in harness.records()))

            await harness.request({"id": "n1", "type": "set_session_name", "name": "my-test-session"})

            state = (await harness.request({"id": "s2", "type": "get_state"}))["data"]
            assert state["sessionName"] == "my-test-session"

            session_file = host.session.session_file
            await _wait_for(lambda: os.path.exists(session_file))
            entries = _read_session_entries(session_file)
            session_info_entries = [e for e in entries if e["type"] == "session_info"]
            assert len(session_info_entries) == 1
            assert session_info_entries[0]["name"] == "my-test-session"
        finally:
            await harness.stop()
