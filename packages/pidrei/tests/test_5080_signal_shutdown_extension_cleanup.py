"""Mirror of pi's regressions/5080-signal-shutdown-extension-cleanup.test.ts.

On SIGTERM/SIGHUP the graceful shutdown must emit `session_shutdown`
(`runtime_host.dispose`) BEFORE touching the terminal. Extension teardown such
as removing a socket does not write to the tty, so it must not be skipped if a
later terminal-restore write fails on a dead or stalled terminal. The
interactive quit path (Ctrl+D, /quit) keeps the opposite order, to preserve the
final TUI frame.

pi calls the unbound prototype method with a duck-typed `this`; the same here,
with `InteractiveMode.shutdown` called on a stand-in object. pi stubs
`process.exit` to throw; pidrei exits through `os._exit`, so that is what gets
stubbed.
"""

import contextlib
import errno
import io
import os
import shutil
import tempfile
import termios
import threading
from types import SimpleNamespace

import pytest

from pidrei.config import APP_NAME
from pidrei.modes.interactive import interactive_mode
from pidrei.modes.interactive.interactive_mode import InteractiveMode
from pidrei.utils.colors import dim


class ProcessExitError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"os._exit({code})")
        self.code = code


@contextlib.contextmanager
def _stubbed_exit():
    """Context manager (predates tonio 0.9.14 yield-fixture support)."""
    original = os._exit

    def fake_exit(code: int) -> None:
        raise ProcessExitError(code)

    os._exit = fake_exit
    try:
        yield
    finally:
        os._exit = original


class _TtyBuffer(io.StringIO):
    """pi sets `process.stdout.isTTY`; `format_resume_command` gates on
    `stdout_isatty()`, which asks `sys.stdout`, so the stand-in has to claim to
    be one."""

    def isatty(self) -> bool:
        return True


@contextlib.contextmanager
def _captured_stdout():
    """The resume hint is written with the output guard's `write_stdout`,
    swapped on the module; `sys.stdout` only answers `isatty()`."""
    import sys

    from pidrei.modes.interactive import interactive_mode

    original = sys.stdout
    original_write_stdout = interactive_mode.write_stdout
    buffer = _TtyBuffer()
    sys.stdout = buffer
    interactive_mode.write_stdout = buffer.write
    try:
        yield buffer
    finally:
        sys.stdout = original
        interactive_mode.write_stdout = original_write_stdout


def create_session_manager(session_file: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        is_persisted=lambda: session_file is not None,
        get_session_file=lambda: session_file,
        get_session_id=lambda: "test-session",
        get_session_dir=lambda: "/tmp/pidrei-sessions",
        uses_default_session_dir=lambda: True,
    )


async def _noop_async() -> None:
    """`InteractiveThemeController.disable_auto_sync` is async (it toggles
    terminal notifications, which is a terminal write)."""


def create_context(order: list[str], session_manager=None) -> SimpleNamespace:
    async def dispose() -> None:
        order.append("dispose")

    async def drain_input(_timeout_ms) -> None:
        order.append("drainInput")

    async def stop() -> None:
        order.append("stop")

    return SimpleNamespace(
        _is_shutting_down=False,
        _shutdown_guard=threading.Lock(),
        _unregister_signal_handlers=lambda: None,
        runtime_host=SimpleNamespace(dispose=dispose),
        ui=SimpleNamespace(terminal=SimpleNamespace(drain_input=drain_input)),
        _theme_controller=SimpleNamespace(disable_auto_sync=_noop_async),
        stop=stop,
        session_manager=session_manager if session_manager is not None else create_session_manager(),
        _emergency_terminal_exit=lambda: None,
    )


async def call_shutdown(context, options: dict | None = None) -> None:
    with contextlib.suppress(ProcessExitError):
        await InteractiveMode.shutdown(context, options)


@pytest.fixture
def temp_session_file(request):
    directory = tempfile.mkdtemp(prefix="pidrei-shutdown-resume-hint-")
    request.addfinalizer(lambda: shutil.rmtree(directory, ignore_errors=True))
    path = os.path.join(directory, "session.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n")
    return path


@pytest.mark.tonio
async def test_signal_triggered_shutdown_emits_session_shutdown_before_terminal_writes():
    order: list[str] = []
    context = create_context(order)

    with _stubbed_exit():
        await call_shutdown(context, {"fromSignal": True})

    assert order == ["dispose", "drainInput", "stop"]
    assert context._is_shutting_down is True


@pytest.mark.tonio
async def test_interactive_quit_stops_the_tui_before_emitting_session_shutdown():
    order: list[str] = []
    context = create_context(order)

    with _stubbed_exit():
        await call_shutdown(context)

    assert order == ["drainInput", "stop", "dispose"]


@pytest.mark.tonio
async def test_interactive_quit_prints_a_resume_hint_for_persisted_sessions(temp_session_file):
    order: list[str] = []
    context = create_context(order, create_session_manager(temp_session_file))

    with _stubbed_exit(), _captured_stdout() as out:
        await call_shutdown(context)
        # `dim()` decides on colour from the *current* stdout, so the expected
        # string has to be built while the tty stand-in is still installed.
        expected = f"{dim('To resume this session:')} {APP_NAME} --session test-session\n"

    assert order == ["drainInput", "stop", "dispose"]
    assert out.getvalue() == expected


@pytest.mark.tonio
async def test_signal_triggered_shutdown_does_not_print_a_resume_hint(temp_session_file):
    order: list[str] = []
    context = create_context(order, create_session_manager(temp_session_file))

    with _stubbed_exit(), _captured_stdout() as out:
        await call_shutdown(context, {"fromSignal": True})

    assert "To resume this session:" not in out.getvalue()


@pytest.mark.tonio
async def test_re_entrant_shutdown_is_a_no_op():
    order: list[str] = []
    context = create_context(order)
    context._is_shutting_down = True

    with _stubbed_exit():
        await call_shutdown(context, {"fromSignal": True})

    assert order == []


# Regression for the `read EIO` crash reports (crash_tty_read_eio).
#
# When the terminal goes away, stdin reads and raw mode fail with EIO (pidrei
# is left in an orphaned background process group) or ENOTTY (macOS revoked
# the tty). The input reader hands a failed read to the crash handler
# (`_uncaught_crash`, through the TUI's `report_error`), and a raw-mode
# failure reaches it as an uncaught error; for a dead terminal it must exit
# quietly instead of reporting a crash.
#
# pi tests its stdin `error` listener and `uncaughtCrash` apart; here both are
# `_uncaught_crash`, so the stdin cases below also cover pi's "uncaught dead
# terminal errors" case. Crash recording (`/bug`) is not ported: "not recorded"
# is the terminal left alone (`ui.stop` not called), "recorded" is the crash
# report. "Handlers are removed on unregister" is Node listener bookkeeping and
# is not mirrored. The reader's hand-off is covered in pidrei_tui's
# test_terminal_pty.py.


def create_crash_context(calls: list[str]) -> SimpleNamespace:
    async def stop() -> None:
        calls.append("ui.stop")

    async def close() -> None:
        calls.append("ui.close")

    context = SimpleNamespace(
        _is_shutting_down=False,
        _shutdown_guard=threading.Lock(),
        _unregister_signal_handlers=lambda: None,
        ui=SimpleNamespace(stop=stop, close=close),
    )
    context._emergency_terminal_exit = lambda: InteractiveMode._emergency_terminal_exit(context)
    return context


async def capture_crash_exit(context, error: BaseException) -> int:
    with _stubbed_exit(), pytest.raises(ProcessExitError) as exited:
        await InteractiveMode._uncaught_crash(context, error)
    return exited.value.code


@pytest.mark.tonio
@pytest.mark.parametrize(
    "error",
    [
        OSError(errno.EIO, "read EIO"),
        termios.error(errno.EIO, "setRawMode EIO"),
        termios.error(errno.ENOTTY, "setRawMode ENOTTY"),
    ],
    ids=["read EIO", "setRawMode EIO", "setRawMode ENOTTY"],
)
async def test_stdin_dead_terminal_errors_exit_quietly_without_reporting_a_crash(monkeypatch, error):
    calls: list[str] = []
    report = io.StringIO()
    monkeypatch.setattr(interactive_mode, "write_stderr", report.write)

    assert await capture_crash_exit(create_crash_context(calls), error) == 129
    assert calls == []
    assert report.getvalue() == ""


@pytest.mark.tonio
@pytest.mark.parametrize(
    "error",
    [OSError(errno.ECONNREFUSED, "read ECONNREFUSED"), RuntimeError("boom")],
    ids=["other stdin error", "other uncaught error"],
)
async def test_other_errors_still_crash(monkeypatch, error):
    calls: list[str] = []
    report = io.StringIO()
    monkeypatch.setattr(interactive_mode, "write_stderr", report.write)

    assert await capture_crash_exit(create_crash_context(calls), error) == 1
    assert calls == ["ui.stop", "ui.close"]
    assert report.getvalue().startswith(f"{APP_NAME} exiting due to uncaught exception:\n")
