"""vitest fake timers over pidrei's clock and timer seams.

pi drives `Date.now()` and `setTimeout` from `vi.useFakeTimers()`; pidrei reads
the clock through `pidrei_utils.clock.now_ms` and arms timers as
`pidrei_utils.timers.Timeout`/`Interval`, which start through
`timers._start`, so a test swaps both for this queue and advances time by
hand.

Shared by the agent and pidrei suites (the pidrei suite imports it from this
directory, like its `tui/tests` helpers); the root conftest restores both
seams after every test.
"""

import contextlib
import threading

from pidrei_utils import clock, timers


class FakeTimers:
    """Thread-safe like the timers it replaces, which production arms from any
    task: the queue and clock are guarded, and callbacks run outside the lock
    (they may arm or cancel timers). A timer's `cancel()` sets its event; a
    cancelled entry never fires and no longer counts as pending."""

    def __init__(self, start_ms: int = 0) -> None:
        self._lock = threading.Lock()
        self.now = start_ms
        self._timers: list[tuple[int, int, float, object, object, bool]] = []
        self._sequence = 0

    def now_ms(self) -> int:
        return self.now

    def _arm_locked(self, delay_ms: float, cancelled, fn, repeat: bool) -> None:
        self._sequence += 1
        self._timers.append((self.now + int(delay_ms), self._sequence, delay_ms, cancelled, fn, repeat))

    def start(self, cancelled, delay_ms: float, fn, repeat: bool, _on_error) -> None:
        """Replaces `timers._start`: queue the timer instead of starting it."""
        with self._lock:
            self._arm_locked(delay_ms, cancelled, fn, repeat)

    @property
    def pending(self) -> int:
        """`vi.getTimerCount()`: timers armed and not yet fired or cancelled."""
        with self._lock:
            return sum(1 for entry in self._timers if not entry[3].is_set())

    def _pop_due_locked(self, target_ms: int):
        due = sorted(
            (entry for entry in self._timers if entry[0] <= target_ms and not entry[3].is_set()),
            key=lambda e: (e[0], e[1]),
        )
        if not due:
            return None
        entry = due[0]
        self._timers.remove(entry)
        self.now = max(self.now, entry[0])
        if entry[5]:
            self._arm_locked(entry[2], entry[3], entry[4], True)
        return entry

    def pop_due(self, target_ms: int) -> object | None:
        """Remove the earliest timer due by `target_ms` and return its callback
        without calling it (advancing `now` to its due time), or None.

        For timers whose callback only spawns async work: the test awaits that
        work itself instead of racing a detached task."""
        with self._lock:
            entry = self._pop_due_locked(target_ms)
        return entry[4] if entry is not None else None

    def advance(self, ms: int) -> None:
        """`vi.advanceTimersByTime`: fire every timer due within `ms`, in order."""
        with self._lock:
            target = self.now + ms
        while True:
            with self._lock:
                entry = self._pop_due_locked(target)
            if entry is None:
                break
            entry[4]()
        with self._lock:
            self.now = target


@contextlib.contextmanager
def fake_timers(start_ms: int = 0):
    fake = FakeTimers(start_ms)
    original_now = clock.now_ms
    original_start = timers._start
    clock.now_ms = fake.now_ms
    timers._start = fake.start
    try:
        yield fake
    finally:
        clock.now_ms = original_now
        timers._start = original_start
