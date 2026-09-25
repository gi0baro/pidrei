"""vitest fake timers over pidrei's clock and timer seams.

pi drives `Date.now()` and `setTimeout` from `vi.useFakeTimers()`; pidrei reads
the clock through `pidrei_ai.utils.clock.now_ms` and arms timers through
`pidrei_ai.utils.timers.set_timeout`, so a test swaps both for this queue and
advances time by hand.

Shared by the agent and pidrei suites (the pidrei suite imports it from this
directory, like its `tui/tests` helpers); each suite's conftest restores both
seams after every test.
"""

import contextlib
import threading

from pidrei_ai.utils import clock, timers


class FakeTimers:
    """Thread-safe like the `timers.set_timeout` it replaces, which production
    calls from any task: the queue and clock are guarded, and callbacks run
    outside the lock (they may arm or cancel timers)."""

    def __init__(self, start_ms: int = 0) -> None:
        self._lock = threading.Lock()
        self.now = start_ms
        self._timers: list[tuple[int, int, object]] = []
        self._sequence = 0

    def now_ms(self) -> int:
        return self.now

    def set_timeout(self, delay_ms: float, callback):
        with self._lock:
            self._sequence += 1
            entry = (self.now + int(delay_ms), self._sequence, callback)
            self._timers.append(entry)

        def cancel() -> None:
            with self._lock:
                if entry in self._timers:
                    self._timers.remove(entry)

        return cancel

    @property
    def pending(self) -> int:
        """`vi.getTimerCount()`: timers armed and not yet fired or cancelled."""
        with self._lock:
            return len(self._timers)

    def _pop_due_locked(self, target_ms: int) -> tuple[int, int, object] | None:
        due = sorted((entry for entry in self._timers if entry[0] <= target_ms), key=lambda e: (e[0], e[1]))
        if not due:
            return None
        entry = due[0]
        self._timers.remove(entry)
        self.now = max(self.now, entry[0])
        return entry

    def pop_due(self, target_ms: int) -> object | None:
        """Remove the earliest timer due by `target_ms` and return its callback
        without calling it (advancing `now` to its due time), or None.

        For timers whose callback only spawns async work: the test awaits that
        work itself instead of racing a detached task."""
        with self._lock:
            entry = self._pop_due_locked(target_ms)
        return entry[2] if entry is not None else None

    def advance(self, ms: int) -> None:
        """`vi.advanceTimersByTime`: fire every timer due within `ms`, in order."""
        with self._lock:
            target = self.now + ms
        while True:
            with self._lock:
                entry = self._pop_due_locked(target)
            if entry is None:
                break
            entry[2]()
        with self._lock:
            self.now = target


@contextlib.contextmanager
def fake_timers(start_ms: int = 0):
    fake = FakeTimers(start_ms)
    original_now = clock.now_ms
    original_set_timeout = timers.set_timeout
    clock.now_ms = fake.now_ms
    timers.set_timeout = fake.set_timeout
    try:
        yield fake
    finally:
        clock.now_ms = original_now
        timers.set_timeout = original_set_timeout
