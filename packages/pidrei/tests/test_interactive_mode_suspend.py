"""Mirror of pi coding-agent test/interactive-mode-suspend.test.ts.

pi spies on process.on/once/kill and setInterval; here the module's os.kill
is swapped, and a recording dispatcher stands in for the process's
`SignalDispatcher`, so the test fires SIGCONT by calling the handler the
mode installed, where pi's test calls the registered listener. The resume
is a spawned flow (`SpawnedFlows`). Not ported: pi's SIGINT listener while
suspended (the runtime delivers no SIGINT to pidrei's code), the win32
status message (POSIX-only port), and the event-loop keep-alive timer.
"""

import signal as signal_module
from types import SimpleNamespace

import pytest

import pidrei.modes.interactive.interactive_mode as interactive_mode_module
from pidrei.modes.interactive.interactive_mode import InteractiveMode

from .flow_helpers import SpawnedFlows


class _RecordingDispatcher:
    def __init__(self):
        self.handlers = {}

    def set_handler(self, signum, handler):
        self.handlers[signum] = handler


def _create_ui(takes_suspend_signal=True):
    ui = SimpleNamespace(start_calls=[], stop_calls=[], request_render_calls=[], suspend_signal_calls=[])

    async def start():
        ui.start_calls.append(True)

    async def stop():
        ui.stop_calls.append(True)

    def suspend_signal():
        ui.suspend_signal_calls.append(True)
        return takes_suspend_signal

    ui.start = start
    ui.stop = stop
    ui.request_render = lambda force=False: ui.request_render_calls.append(force)
    ui.suspend_signal = suspend_signal
    return ui


def _create_context(ui):
    context = SimpleNamespace(
        ui=ui,
        _spawn_flow=SpawnedFlows(),
        _signal_dispatcher=_RecordingDispatcher(),
        _suspend_resumed=None,
    )
    context._resume_after_suspend = lambda *args: InteractiveMode._resume_after_suspend(context, *args)
    InteractiveMode._register_signal_handlers(context)
    return context


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
async def test_restores_the_tui_on_sigcont():
    ui = _create_ui()
    context = _create_context(ui)
    kill_calls: list = []

    swaps = _Swaps()
    swaps.set(interactive_mode_module.os, "kill", lambda pid, sig: kill_calls.append((pid, sig)))
    try:
        await InteractiveMode._handle_ctrl_z(context)

        assert ui.stop_calls == [True]
        assert kill_calls == [(0, signal_module.SIGTSTP)]
        # Suspended: the resume waits for SIGCONT on its own task.
        assert ui.start_calls == []

        await context._signal_dispatcher.handlers[signal_module.SIGCONT](signal_module.SIGCONT)
        await context._spawn_flow.finish()
    finally:
        swaps.restore()

    assert ui.start_calls == [True]
    assert ui.request_render_calls == [True]
    assert context._suspend_resumed is None


@pytest.mark.tonio
async def test_cleans_up_if_suspension_fails():
    ui = _create_ui()
    context = _create_context(ui)
    suspend_error = RuntimeError("suspend failed")

    def failing_kill(pid, sig):
        raise suspend_error

    swaps = _Swaps()
    swaps.set(interactive_mode_module.os, "kill", failing_kill)
    try:
        with pytest.raises(RuntimeError, match="suspend failed"):
            await InteractiveMode._handle_ctrl_z(context)
    finally:
        swaps.restore()

    assert ui.stop_calls == [True]
    assert context._suspend_resumed is None
    assert context._spawn_flow == []
    assert ui.start_calls == []
    assert ui.request_render_calls == []


# pidrei-only: pi's TUI listens for SIGTSTP itself; here the mode's handler on
# the signal dispatcher forwards it (recipe `tui-island`, "Suspend signal").


@pytest.mark.tonio
async def test_forwards_an_external_sigtstp_to_the_tui():
    ui = _create_ui(takes_suspend_signal=True)
    context = _create_context(ui)
    stops: list = []

    async def recording_stop_on_signal(signum):
        stops.append(signum)

    swaps = _Swaps()
    swaps.set(interactive_mode_module, "stop_on_signal", recording_stop_on_signal)
    try:
        await context._signal_dispatcher.handlers[signal_module.SIGTSTP](signal_module.SIGTSTP)
    finally:
        swaps.restore()

    assert ui.suspend_signal_calls == [True]
    assert stops == []


@pytest.mark.tonio
async def test_stops_the_process_on_sigtstp_while_the_tui_does_not_take_it():
    # Not started, stopped (a child owns the terminal), or `_handle_ctrl_z`'s
    # own SIGTSTP after its stop: the default action, as before.
    ui = _create_ui(takes_suspend_signal=False)
    context = _create_context(ui)
    stops: list = []

    async def recording_stop_on_signal(signum):
        stops.append(signum)

    swaps = _Swaps()
    swaps.set(interactive_mode_module, "stop_on_signal", recording_stop_on_signal)
    try:
        await context._signal_dispatcher.handlers[signal_module.SIGTSTP](signal_module.SIGTSTP)
    finally:
        swaps.restore()

    assert ui.suspend_signal_calls == [True]
    assert stops == [signal_module.SIGTSTP]
