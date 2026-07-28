"""Mirror of pi coding-agent src/modes/interactive/components/countdown-timer.ts."""

import math

from pidrei_tui._timers import Interval


class CountdownTimer:
    """Reusable countdown timer for dialog components."""

    def __init__(self, timeout_ms: float, tui, on_tick, on_expire) -> None:
        self._tui = tui
        self._on_tick = on_tick
        self._on_expire = on_expire
        self._remaining_seconds = math.ceil(timeout_ms / 1000)
        self._on_tick(self._remaining_seconds)
        self._interval: Interval | None = Interval(1000, self._tick)

    async def _tick(self) -> None:
        self._remaining_seconds -= 1
        self._on_tick(self._remaining_seconds)
        if self._tui is not None:
            self._tui.request_render()

        if self._remaining_seconds <= 0:
            self.dispose()
            self._on_expire()

    def dispose(self) -> None:
        if self._interval is not None:
            self._interval.cancel()
            self._interval = None
