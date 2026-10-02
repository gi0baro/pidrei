"""Manual timers for components that animate or count down through pidrei_tui's
`Timeout`/`Interval` (pi's fake timers)."""

import contextlib

from pidrei_utils import timers as timers_module


class ManualUiTimers:
    """While installed, `Timeout`/`Interval` are recorded instead of started,
    so spinner frames and countdowns only move when a test moves them. Each
    entry is `(delay_ms, cancelled_event, fn)`; `fn` is the timer's callback."""

    def __init__(self) -> None:
        self.scheduled: list[tuple[float, object, object]] = []

    def _start(self, cancelled, delay_ms, fn, repeat, on_error) -> None:
        self.scheduled.append((delay_ms, cancelled, fn))


@contextlib.contextmanager
def manual_ui_timers():
    """Swap the timers' start for `ManualUiTimers` for the block."""
    timers = ManualUiTimers()
    original = timers_module._start
    timers_module._start = timers._start
    try:
        yield timers
    finally:
        timers_module._start = original
