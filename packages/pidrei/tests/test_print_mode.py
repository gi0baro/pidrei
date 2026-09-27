"""Mirrors pi coding-agent test/print-mode.test.ts.

pi mocks output via vi.spyOn(console.error) and fake runtime hosts; here
the output-guard functions (stdout and stderr alike) are swapped on the module.
"""

import contextlib
from types import SimpleNamespace

import pytest

from pidrei.modes import print_mode
from pidrei.modes.print_mode import PrintModeOptions, run_print_mode
from pidrei_ai.types import ImageContent

from .agent_session_helpers import create_assistant_message


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


class _FakeExtensionRunner:
    def __init__(self):
        self.emitted = []

    def has_handlers(self, event_type):
        return event_type == "session_shutdown"

    async def emit(self, event):
        self.emitted.append(event)


class _FakeSession:
    def __init__(self, assistant_message):
        self.session_manager = SimpleNamespace(get_header=lambda: None)
        self.agent = SimpleNamespace(wait_for_idle=self.wait_for_idle, subscribe=lambda _listener: lambda: None)
        self.state = SimpleNamespace(messages=[assistant_message])
        self.extension_runner = _FakeExtensionRunner()
        self.bind_calls = []
        self.prompt_calls = []
        self.subscribe_calls = 0

    async def bind_extensions(self, bindings):
        self.bind_calls.append(bindings)

    def subscribe(self, _listener):
        self.subscribe_calls += 1
        return lambda: None

    async def prompt(self, text, options=None):
        self.prompt_calls.append((text, options))

    async def wait_for_idle(self):
        pass

    async def reload(self):
        pass


class _FakeRuntimeHost:
    def __init__(self, session):
        self.session = session
        self.dispose_calls = 0

    def set_rebind_session(self, _callback=None):
        pass

    async def new_session(self, **_kwargs):
        return {"cancelled": False}

    async def fork(self, _entry_id, **_kwargs):
        return {"cancelled": False, "selectedText": ""}

    async def switch_session(self, _path, **_kwargs):
        return {"cancelled": False}

    async def dispose(self):
        self.dispose_calls += 1
        await self.session.extension_runner.emit({"type": "session_shutdown", "reason": "quit"})


def _create_runtime_host(assistant_message):
    return _FakeRuntimeHost(_FakeSession(assistant_message))


class TestRunPrintMode:
    @pytest.mark.tonio
    async def test_emits_session_shutdown_in_text_mode(self):
        runtime_host = _create_runtime_host(create_assistant_message("done"))
        session = runtime_host.session
        images = [ImageContent(mime_type="image/png", data="abc")]
        outputs = []

        async def drain():
            pass

        with _patched(print_mode, write_raw_stdout=outputs.append, drain_output=drain):
            exit_code = await run_print_mode(
                runtime_host, PrintModeOptions(mode="text", initial_message="Say done", initial_images=images)
            )

        assert exit_code == 0
        assert len(session.prompt_calls) == 1
        prompt_text, prompt_options = session.prompt_calls[0]
        assert prompt_text == "Say done"
        assert prompt_options.images == images
        assert session.extension_runner.emitted == [{"type": "session_shutdown", "reason": "quit"}]
        assert outputs == ["done\n"]

    @pytest.mark.tonio
    async def test_emits_session_shutdown_in_json_mode(self):
        runtime_host = _create_runtime_host(create_assistant_message("done"))
        session = runtime_host.session
        outputs = []

        async def drain():
            pass

        with _patched(print_mode, write_raw_stdout=outputs.append, drain_output=drain):
            exit_code = await run_print_mode(runtime_host, PrintModeOptions(mode="json", messages=["hello"]))

        assert exit_code == 0
        assert len(session.prompt_calls) == 1
        prompt_text, prompt_options = session.prompt_calls[0]
        assert prompt_text == "hello"
        assert prompt_options is None
        assert session.extension_runner.emitted == [{"type": "session_shutdown", "reason": "quit"}]

    @pytest.mark.tonio
    async def test_emits_session_shutdown_and_returns_non_zero_on_assistant_error(self):
        runtime_host = _create_runtime_host(
            create_assistant_message("", stop_reason="error", error_message="provider failure")
        )
        session = runtime_host.session
        stderr = []

        async def drain():
            pass

        with _patched(print_mode, write_raw_stdout=lambda _text: None, write_stderr=stderr.append, drain_output=drain):
            exit_code = await run_print_mode(runtime_host, PrintModeOptions(mode="text"))

        assert exit_code == 1
        assert "provider failure" in "".join(stderr)
        assert session.extension_runner.emitted == [{"type": "session_shutdown", "reason": "quit"}]
