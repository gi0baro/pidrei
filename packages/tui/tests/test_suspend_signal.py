"""Mirror of pi tui test/suspend-signal.test.ts.

Regression for #12272: an external SIGTSTP stopped pi without restoring the
terminal. The handler must run only while the TUI owns the terminal, so a
child that owns it after stop() (external editor, interactive command) is
suspended with the default action.

pi's TUI registers a SIGTSTP listener and its cases count
`process.listenerCount("SIGTSTP")`. Here the host's signal handler calls
`suspend_signal()` (recipe `tui-island`, "Suspend signal"), so the cases
assert what it returns and whether the handler ran; none sends a signal.
"does not register a listener twice when start() runs again" and "does not
remove the listener while it stays needed" are about listener bookkeeping,
which has no counterpart: nothing is registered or removed.
"""

import pytest
import tonio.colored as tonio

from pidrei_tui.tui_main_screen import TuiMainScreen

from .virtual_terminal import VirtualTerminal


class _HandlerCalls:
    """The suspend handler: counts its calls and sets `ran` last, since it
    runs on the input path, on another coroutine."""

    def __init__(self) -> None:
        self.calls = 0
        self.ran = tonio.Event()

    def __call__(self) -> None:
        self.calls += 1
        self.ran.set()

    async def wait(self) -> None:
        await self.ran.wait(5)
        assert self.ran.is_set(), "the suspend handler never ran"
        self.ran = tonio.Event()


@pytest.mark.tonio
async def test_calls_the_handler_on_sigtstp_only_while_started():
    handler = _HandlerCalls()
    tui = TuiMainScreen(VirtualTerminal(40, 10))
    tui.set_suspend_signal_handler(handler)
    await tui.start()
    try:
        assert tui.suspend_signal() is True
        await handler.wait()
        assert handler.calls == 1

        await tui.stop()
        assert tui.suspend_signal() is False

        await tui.start()
        assert tui.suspend_signal() is True
        await handler.wait()
        assert handler.calls == 2
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_does_not_take_sigtstp_before_the_first_start():
    tui = TuiMainScreen(VirtualTerminal(40, 10))
    tui.set_suspend_signal_handler(lambda: None)
    assert tui.suspend_signal() is False

    await tui.start()
    try:
        assert tui.suspend_signal() is True
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_takes_sigtstp_until_stop_has_restored_the_terminal():
    taken: list[bool] = []

    class RecordingTerminal(VirtualTerminal):
        async def stop(self) -> None:
            await super().stop()
            # The terminal is restored and its input handler gone: the
            # suspend is still taken, and dropped with the input.
            taken.append(tui.suspend_signal())

    handler = _HandlerCalls()
    tui = TuiMainScreen(RecordingTerminal(40, 10))
    tui.set_suspend_signal_handler(handler)
    await tui.start()
    await tui.stop()

    assert taken == [True]
    assert tui.suspend_signal() is False
    assert handler.calls == 0


@pytest.mark.tonio
async def test_applies_a_handler_set_or_cleared_while_started_immediately():
    tui = TuiMainScreen(VirtualTerminal(40, 10))
    await tui.start()
    try:
        assert tui.suspend_signal() is False

        tui.set_suspend_signal_handler(lambda: None)
        assert tui.suspend_signal() is True

        tui.set_suspend_signal_handler(None)
        assert tui.suspend_signal() is False
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_does_not_take_sigtstp_when_a_handler_is_set_while_stopped():
    tui = TuiMainScreen(VirtualTerminal(40, 10))
    await tui.start()
    await tui.stop()
    tui.set_suspend_signal_handler(lambda: None)
    assert tui.suspend_signal() is False
