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
import io
import os
import shutil
import tempfile
from types import SimpleNamespace

import pytest

from pidrei.config import APP_NAME
from pidrei.modes.interactive.interactive_mode import InteractiveMode
from pidrei.utils.colors import dim


class ProcessExitError(Exception):
    pass


@contextlib.contextmanager
def _stubbed_exit():
    """No yield fixtures under tonio; a context manager instead."""
    original = os._exit

    def fake_exit(_code: int) -> None:
        raise ProcessExitError

    os._exit = fake_exit
    try:
        yield
    finally:
        os._exit = original


class _TtyBuffer(io.StringIO):
    """pi sets `process.stdout.isTTY`; `format_resume_command` gates on
    `sys.stdout.isatty()`, so the capture has to claim to be one."""

    def isatty(self) -> bool:
        return True


@contextlib.contextmanager
def _captured_stdout():
    import sys

    original = sys.stdout
    buffer = _TtyBuffer()
    sys.stdout = buffer
    try:
        yield buffer
    finally:
        sys.stdout = original


def create_session_manager(session_file: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        is_persisted=lambda: session_file is not None,
        get_session_file=lambda: session_file,
        get_session_id=lambda: "test-session",
        get_session_dir=lambda: "/tmp/pidrei-sessions",
        uses_default_session_dir=lambda: True,
    )


def create_context(order: list[str], session_manager=None) -> SimpleNamespace:
    async def dispose() -> None:
        order.append("dispose")

    async def drain_input(_timeout_ms) -> None:
        order.append("drainInput")

    async def stop() -> None:
        order.append("stop")

    return SimpleNamespace(
        _is_shutting_down=False,
        _unregister_signal_handlers=lambda: None,
        runtime_host=SimpleNamespace(dispose=dispose),
        ui=SimpleNamespace(terminal=SimpleNamespace(drain_input=drain_input)),
        _theme_controller=SimpleNamespace(disable_auto_sync=lambda: None),
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
