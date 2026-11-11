"""Mirrors pi coding-agent test/print-mode.test.ts.

pi mocks output via vi.spyOn(console.error) and fake runtime hosts; here
the output-guard functions (stdout and stderr alike) are swapped on the module.
"""

import contextlib
import signal
from types import SimpleNamespace

import pytest
import tonio.colored as tonio

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


class _FakeSignalDispatcher:
    """Captures the mode's signal handlers, so a test delivers a signal by calling one."""

    def __init__(self):
        self.handlers = {}

    def set_handler(self, signum, handler):
        self.handlers[signum] = handler


class _Done:
    """A spawned call and an Event set when it returns."""

    def __init__(self, coro):
        self._event = tonio.Event()
        self._handle = tonio.spawn(self._run(coro))

    async def _run(self, coro):
        try:
            await coro
        finally:
            self._event.set()

    async def wait(self):
        await self._event.wait(5)
        assert self._event.is_set(), "the call never returned"
        await self._handle


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

    # #10799
    @pytest.mark.tonio
    async def test_warns_on_stderr_and_returns_non_zero_when_the_response_was_truncated(self):
        runtime_host = _create_runtime_host(create_assistant_message("partial", stop_reason="length"))
        stdout = []
        stderr = []

        async def drain():
            pass

        with _patched(print_mode, write_raw_stdout=stdout.append, write_stderr=stderr.append, drain_output=drain):
            exit_code = await run_print_mode(runtime_host, PrintModeOptions(mode="text"))

        assert exit_code == 1
        assert stdout == ["partial\n"]
        assert stderr == ["Response was truncated: the model reached its output token limit.\n"]

    # A signal during the final disposal used to hit no handler and kill pi mid-dispose.
    # pi emits SIGTERM on the process and sleeps before asserting "not exited yet"; here the
    # installed handler is called directly, and each exit records whether the disposal had
    # been released by then.
    @pytest.mark.tonio
    async def test_waits_for_the_final_disposal_when_sigterm_arrives_during_it_and_exits_on_a_repeated_sigterm(self):
        runtime_host = _create_runtime_host(create_assistant_message("done"))
        dispose_started = tonio.Event()
        dispose_release = tonio.Event()
        exited = tonio.Event()
        exits: list[tuple[int, bool]] = []
        dispatcher = _FakeSignalDispatcher()

        async def slow_dispose():
            runtime_host.dispose_calls += 1
            dispose_started.set()
            await dispose_release.wait()

        def record_exit(code):
            exits.append((code, dispose_release.is_set()))
            exited.set()

        async def drain():
            pass

        runtime_host.dispose = slow_dispose
        with _patched(
            print_mode,
            write_raw_stdout=lambda _text: None,
            drain_output=drain,
            hard_exit=record_exit,
            kill_tracked_detached_children=lambda: None,
        ):
            run = _Done(
                run_print_mode(
                    runtime_host,
                    PrintModeOptions(mode="text", initial_message="Say done"),
                    signal_dispatcher=dispatcher,
                )
            )
            signals: list[_Done] = []
            try:
                await dispose_started.wait(5)
                assert dispose_started.is_set(), "the final disposal never started"
                assert runtime_host.dispose_calls == 1
                assert set(dispatcher.handlers) == {signal.SIGTERM, signal.SIGHUP}

                signals.append(_Done(dispatcher.handlers[signal.SIGTERM](signal.SIGTERM)))
                signals.append(_Done(dispatcher.handlers[signal.SIGTERM](signal.SIGTERM)))
                await exited.wait(5)
                # One of the two counts as the repeated signal and exits at once; the other
                # waits for the disposal.
                assert exits == [(143, False)]
                assert runtime_host.dispose_calls == 1
            finally:
                dispose_release.set()
                for done in [run, *signals]:
                    await done.wait()
        assert [code for code, released in exits if not released] == [143]
        # After the disposal: the waiting signal's exit and the main exit, both with the
        # signal's status (the repeated signal's stand-in exit returns, so it waits too).
        assert [code for code, released in exits if released] == [143, 143, 143]
        assert runtime_host.dispose_calls == 1
