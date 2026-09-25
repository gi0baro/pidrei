"""Shared helpers for pidrei_tui tests."""

import contextlib
import os
import threading

from pidrei_tui._owner import OwnerTask, TimerHandle


class ManualOwnerTimers:
    """pi's `t.mock.timers` for one `OwnerTask`: its `after` queues the
    callback instead of sleeping, and `tick` fires what falls due, in due
    order, on the ticking task (what an owner that was never started does
    with a real fire)."""

    def __init__(self, owner: OwnerTask) -> None:
        # `after` can also be called from another task (a request task's
        # inline apply): the queue is guarded, and callbacks run outside the
        # lock.
        self._lock = threading.Lock()
        self._now = 0.0
        self._queue: list[tuple[float, TimerHandle, object]] = []
        owner.after = self._after

    def _after(self, delay_ms: float, fn) -> TimerHandle:
        handle = TimerHandle()
        with self._lock:
            self._queue.append((self._now + delay_ms, handle, fn))
        return handle

    @property
    def remaining(self) -> list[float]:
        """Milliseconds left on each live timer."""
        with self._lock:
            return [due - self._now for due, handle, _ in self._queue if not handle.cancelled]

    async def tick(self, ms: float) -> None:
        with self._lock:
            target = self._now + ms
        while True:
            with self._lock:
                due = [entry for entry in self._queue if entry[0] <= target and not entry[1].cancelled]
                if not due:
                    break
                entry = min(due, key=lambda item: item[0])
                self._queue.remove(entry)
                self._now = entry[0]
            await entry[2]()
        with self._lock:
            self._now = target


@contextlib.contextmanager
def env_var(name, value):
    original = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = original
