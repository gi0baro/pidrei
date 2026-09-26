"""pidrei-specific: `CountdownTimer` ticks on its interval's task, so a tick
that was already due when the dialog disposed the countdown must do nothing
(pi's `clearInterval` guarantees no tick after it)."""

import threading
from types import SimpleNamespace

import pytest

from pidrei.modes.interactive.components.countdown_timer import CountdownTimer

from .ui_timer_helpers import manual_ui_timers


@pytest.mark.tonio
async def test_a_tick_due_when_the_countdown_is_disposed_does_nothing():
    tui = SimpleNamespace(request_render=lambda: None, state_lock=threading.RLock())
    ticks: list[int] = []
    expired: list[bool] = []
    with manual_ui_timers() as timers:
        countdown = CountdownTimer(1000, tui, ticks.append, lambda: expired.append(True))
    [(_delay, _cancelled, tick)] = timers.scheduled
    assert ticks == [1]

    # The user answers: the dialog disposes the countdown. The interval's
    # fire had already passed its cancelled check, so its tick still runs.
    countdown.dispose()
    tick()

    assert ticks == [1]
    assert expired == []
