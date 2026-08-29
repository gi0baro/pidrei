"""Mirror of pi coding-agent test/interactive-mode-suspend.test.ts.

pi spies on process.on/once/kill and setInterval; here the module's os.kill,
signal.signal, and tonio signal receiver are swapped by hand (predates
tonio 0.9.14; `monkeypatch` works in tonio tests now). The pi test for the
win32 status message is not ported (POSIX-only port), and Python needs no
event-loop keep-alive timer while suspended.
"""

import signal as signal_module
from types import SimpleNamespace

import pytest

import pidrei.modes.interactive.interactive_mode as interactive_mode_module
from pidrei.modes.interactive.interactive_mode import InteractiveMode


class _FakeSigcontReceiver:
    """Yields SIGCONT immediately, standing in for the real suspend/resume."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        return signal_module.SIGCONT


def _create_ui():
    ui = SimpleNamespace(start_calls=[], stop_calls=[], request_render_calls=[])

    async def start():
        ui.start_calls.append(True)

    async def stop():
        ui.stop_calls.append(True)

    ui.start = start
    ui.stop = stop
    ui.request_render = lambda force=False: ui.request_render_calls.append(force)
    return ui


class _Swaps:
    """Hand-rolled monkeypatching with guaranteed restore."""

    def __init__(self):
        self._saved = []

    def set(self, obj, name, value):
        self._saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def restore(self):
        for obj, name, value in reversed(self._saved):
            setattr(obj, name, value)


@pytest.mark.tonio
async def test_ignores_sigint_while_suspended_and_restores_the_tui_on_sigcont():
    ui = _create_ui()
    context = SimpleNamespace(ui=ui)
    kill_calls: list = []
    signal_calls: list = []

    def fake_signal(sig, handler):
        signal_calls.append((sig, handler))
        return "previous-sigint-handler"

    swaps = _Swaps()
    swaps.set(interactive_mode_module.signal, "signal", fake_signal)
    swaps.set(interactive_mode_module.os, "kill", lambda pid, sig: kill_calls.append((pid, sig)))
    swaps.set(
        interactive_mode_module.tonio_signals,
        "signal_receiver",
        lambda *sigs: _FakeSigcontReceiver(),
    )
    try:
        await InteractiveMode._handle_ctrl_z(context)
    finally:
        swaps.restore()

    assert signal_calls[0] == (signal_module.SIGINT, signal_module.SIG_IGN)
    assert ui.stop_calls == [True]
    assert kill_calls == [(0, signal_module.SIGTSTP)]
    # SIGINT is restored to the previous handler after resume
    assert signal_calls[-1] == (signal_module.SIGINT, "previous-sigint-handler")
    assert ui.start_calls == [True]
    assert ui.request_render_calls == [True]


@pytest.mark.tonio
async def test_cleans_up_the_temporary_handlers_if_suspension_fails():
    ui = _create_ui()
    context = SimpleNamespace(ui=ui)
    signal_calls: list = []
    suspend_error = RuntimeError("suspend failed")

    def fake_signal(sig, handler):
        signal_calls.append((sig, handler))
        return "previous-sigint-handler"

    def failing_kill(pid, sig):
        raise suspend_error

    swaps = _Swaps()
    swaps.set(interactive_mode_module.signal, "signal", fake_signal)
    swaps.set(interactive_mode_module.os, "kill", failing_kill)
    swaps.set(
        interactive_mode_module.tonio_signals,
        "signal_receiver",
        lambda *sigs: _FakeSigcontReceiver(),
    )
    try:
        with pytest.raises(RuntimeError, match="suspend failed"):
            await InteractiveMode._handle_ctrl_z(context)
    finally:
        swaps.restore()

    assert ui.stop_calls == [True]
    assert signal_calls[-1] == (signal_module.SIGINT, "previous-sigint-handler")
    assert ui.start_calls == []
    assert ui.request_render_calls == []
