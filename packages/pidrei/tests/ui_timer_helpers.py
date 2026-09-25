"""Manual timers for components that animate or count down through pidrei_tui's
`Timeout`/`Interval` (pi's fake timers)."""

import contextlib

from pidrei_tui._owner import TimerHandle
from pidrei_tui._timers import set_ui_owner


class ManualUiTimers:
    """Stands in for the ambient UI owner's timers: `Timeout`/`Interval` created
    while installed are recorded and never fire, so spinner frames and countdowns
    only move when a test moves them. Real ones run on another task (no UI owner
    runs in these tests) and change what the component renders under the test."""

    serving = True

    def __init__(self) -> None:
        self.scheduled: list[tuple[float, TimerHandle, object]] = []

    def every(self, delay_ms, fn) -> TimerHandle:
        handle = TimerHandle()
        self.scheduled.append((delay_ms, handle, fn))
        return handle

    after = every


@contextlib.contextmanager
def manual_ui_timers():
    """Install `ManualUiTimers` as the ambient UI owner for the block (the
    conftest guard fails loudly if a test leaves one installed)."""
    timers = ManualUiTimers()
    set_ui_owner(timers)
    try:
        yield timers
    finally:
        set_ui_owner(None)
