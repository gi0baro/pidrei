"""Mirror of pi's suite/regressions/9068-user-bash-fail-closed.test.ts.

pi mocks the output guard and the JSONL reader to drive `runRpcMode`
in-process, and calls the interactive editor submit handler on a stub
context; pidrei does the same through the rpc_mode module seams and
`InteractiveMode._handle_editor_submit`. Responses are awaited through an
event the fake stdout sets (pi uses `vi.waitFor`).
"""

import contextlib
import json
from functools import partial
from types import SimpleNamespace
from typing import Any

import pytest
import tonio.colored as tonio

from pidrei.core.bash_executor import BashResult
from pidrei.modes.interactive.interactive_mode import InteractiveMode
from pidrei.modes.rpc import rpc_mode

from .harness import create_harness


LOCAL_RESULT = BashResult(output="local output", exit_code=0, cancelled=False, truncated=False)


class _NotifyingLines(list):
    """Fake stdout: records written chunks and signals every write."""

    def __init__(self):
        super().__init__()
        self.written = tonio.Event()

    def append(self, chunk: str) -> None:
        super().append(chunk)
        self.written.set()

    def records(self) -> list[dict[str, Any]]:
        return [json.loads(line) for chunk in self for line in chunk.split("\n") if line.strip()]


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


async def _wait_for_record(lines: _NotifyingLines, predicate, timeout: float = 5.0) -> dict[str, Any]:
    while True:
        lines.written.clear()
        found = next((record for record in lines.records() if predicate(record)), None)
        if found is not None:
            return found
        await lines.written.wait(timeout)
        assert lines.written.is_set(), "timed out waiting for RPC output"


def _runtime_host(session) -> SimpleNamespace:
    async def cancelled(*_args, **_kwargs):
        return {"cancelled": True}

    async def noop(*_args, **_kwargs):
        return None

    return SimpleNamespace(
        session=session,
        new_session=cancelled,
        switch_session=cancelled,
        fork=cancelled,
        dispose=noop,
        set_rebind_session=lambda _rebind: None,
    )


def _throwing_extension(pi) -> None:
    async def handler(_event, _ctx):
        raise Exception("Routing failed")

    pi.on("user_bash", handler)


def _empty_result_extension(pi) -> None:
    async def handler(_event, _ctx):
        return {}

    pi.on("user_bash", handler)


def _undefined_extension(pi) -> None:
    async def handler(_event, _ctx):
        return None

    pi.on("user_bash", handler)


@pytest.mark.tonio
@pytest.mark.parametrize(
    ("extension", "error", "execute_count"),
    [
        (_throwing_extension, "Routing failed", 0),
        (_empty_result_extension, "Invalid user_bash handler result", 0),
        (_undefined_extension, None, 1),
    ],
    ids=[
        "fails the request without executing bash when a handler throws",
        "fails the request without executing bash when a handler returns an empty result",
        "executes bash normally when a handler returns undefined",
    ],
)
async def test_rpc_user_bash_failure_handling(monkeypatch, extension, error, execute_count):
    harness = await create_harness(extension_factories=[extension])
    executed: list[str] = []

    async def execute_bash(command, *_args, **_kwargs):
        executed.append(command)
        return LOCAL_RESULT

    monkeypatch.setattr(harness.session, "execute_bash", execute_bash)

    lines = _NotifyingLines()
    stop = tonio.Event()
    ready = tonio.Event()
    handler: dict[str, Any] = {}

    async def fake_pump(on_line, _on_end) -> None:
        handler["on_line"] = on_line
        ready.set()
        await stop.wait()

    async def noop_async() -> None:
        pass

    try:
        with _patched(
            rpc_mode,
            take_over_stdout=lambda: None,
            write_raw_stdout=lines.append,
            wait_for_raw_stdout_backpressure=noop_async,
            flush_raw_stdout=noop_async,
            _pump_stdin_commands=fake_pump,
        ):
            run = tonio.spawn(rpc_mode.run_rpc_mode(_runtime_host(harness.session)))
            await ready.wait()
            handler["on_line"](json.dumps({"id": "bash-request", "type": "bash", "command": "pwd"}))

            response = await _wait_for_record(
                lines, lambda record: record.get("type") == "response" and record.get("id") == "bash-request"
            )
            stop.set()
            await run

        assert response["command"] == "bash"
        assert response["success"] is (error is None)
        if error is not None:
            assert error in response["error"]
            assert any(
                record.get("type") == "extension_error"
                and record.get("event") == "user_bash"
                and error in record.get("error", "")
                for record in lines.records()
            )
        else:
            assert response["data"]["output"] == "local output"
            assert response["data"]["exitCode"] == 0
        assert len(executed) == execute_count
    finally:
        harness.cleanup()


@pytest.mark.tonio
@pytest.mark.parametrize(("text", "exclude_from_context"), [("!pwd", False), ("!!pwd", True)])
async def test_interactive_user_bash_fails_closed_when_a_handler_returns_an_empty_result(
    monkeypatch, text, exclude_from_context
):
    events: list[dict[str, Any]] = []

    def extension(pi) -> None:
        async def handler(event, _ctx):
            events.append(event)
            return {}

        pi.on("user_bash", handler)

    harness = await create_harness(extension_factories=[extension])
    executed: list[str] = []

    async def execute_bash(command, *_args, **_kwargs):
        executed.append(command)
        return LOCAL_RESULT

    monkeypatch.setattr(harness.session, "execute_bash", execute_bash)
    try:
        context = SimpleNamespace(
            session=harness.session,
            session_manager=harness.session_manager,
            _is_bash_mode=True,
            history=[],
            show_warning=lambda _message: None,
            _update_editor_border_color=lambda: None,
            _set_editor_text=lambda _text: None,
        )
        context._add_editor_history = context.history.append
        context._handle_bash_command = partial(InteractiveMode._handle_bash_command, context)

        await InteractiveMode._handle_editor_submit(context, text)

        assert events == [
            {
                "type": "user_bash",
                "command": "pwd",
                "excludeFromContext": exclude_from_context,
                "cwd": harness.session_manager.get_cwd(),
            }
        ]
        assert executed == []
    finally:
        harness.cleanup()
