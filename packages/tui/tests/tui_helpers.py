"""Shared helpers for pidrei_tui tests."""

import contextlib
import os
import threading

from pidrei_tui.components import editor as editor_module


class _ManualTimeout:
    __slots__ = ("cancelled",)

    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class ManualTimers:
    """pi's `t.mock.timers` for the editor's `Timeout`: while the test runs,
    `editor.Timeout` queues the callback instead of sleeping (restored by
    `monkeypatch`), and `tick` fires what falls due, in due order, on the
    ticking task."""

    def __init__(self, monkeypatch) -> None:
        # A timer can also be created from another task (a request task):
        # the queue is guarded, and callbacks run outside the lock.
        self._lock = threading.Lock()
        self._now = 0.0
        self._queue: list[tuple[float, _ManualTimeout, object]] = []
        monkeypatch.setattr(editor_module, "Timeout", self._timeout)

    def _timeout(self, delay_ms: float, fn) -> _ManualTimeout:
        timer = _ManualTimeout()
        with self._lock:
            self._queue.append((self._now + delay_ms, timer, fn))
        return timer

    @property
    def remaining(self) -> list[float]:
        """Milliseconds left on each live timer."""
        with self._lock:
            return [due - self._now for due, timer, _ in self._queue if not timer.cancelled]

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
            entry[2]()
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
