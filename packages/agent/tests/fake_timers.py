"""vitest fake timers for the adaptive publisher.

pi drives `Date.now()` and `setTimeout` from `vi.useFakeTimers()`; pidrei's
publisher reads the clock through `pidrei_ai.utils.clock.now_ms` and arms its
trailing timer through `adaptive_publisher._set_timeout`, so a test swaps both
for this queue and advances time by hand.
"""

import contextlib

from pidrei_agent.harness.utils import adaptive_publisher
from pidrei_ai.utils import clock


class FakeTimers:
    def __init__(self, start_ms: int = 0) -> None:
        self.now = start_ms
        self._timers: list[tuple[int, int, object]] = []
        self._sequence = 0

    def now_ms(self) -> int:
        return self.now

    def set_timeout(self, delay_ms: float, callback):
        self._sequence += 1
        entry = (self.now + int(delay_ms), self._sequence, callback)
        self._timers.append(entry)

        def cancel() -> None:
            if entry in self._timers:
                self._timers.remove(entry)

        return cancel

    def advance(self, ms: int) -> None:
        """`vi.advanceTimersByTime`: fire every timer due within `ms`, in order."""
        target = self.now + ms
        while True:
            due = sorted((entry for entry in self._timers if entry[0] <= target), key=lambda e: (e[0], e[1]))
            if not due:
                break
            entry = due[0]
            self._timers.remove(entry)
            self.now = max(self.now, entry[0])
            entry[2]()
        self.now = target


@contextlib.contextmanager
def fake_timers(start_ms: int = 0):
    timers = FakeTimers(start_ms)
    original_now = clock.now_ms
    original_set_timeout = adaptive_publisher._set_timeout
    clock.now_ms = timers.now_ms
    adaptive_publisher._set_timeout = timers.set_timeout
    try:
        yield timers
    finally:
        clock.now_ms = original_now
        adaptive_publisher._set_timeout = original_set_timeout
