"""Deterministic waits on the credential/models-store backends' serialization points.

The cancellation tests over `FileAuthStorageBackend`/`InMemoryAuthStorageBackend`
wait for two things: that an operation is parked (queued behind another, or
retrying a held file lock) and that an abandoned operation has finished. Sleeping
for either lets a slow runner assert before the step happened, so these seams
report the step itself.
"""

import threading
from collections.abc import Callable

import tonio.colored as tonio

from pidrei_ai.utils import clock


class ObservedLock:
    """Wraps a backend's operation lock (`_async_lock`, pi's promise chain):
    `arrived` counts operations that reached it, `released` counts finished
    critical sections. Arrival is recorded as the last step before the inner
    acquire, so `arrived == n` means the n-th operation is at the lock."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self._mutex = threading.Lock()
        self.arrived = 0
        self.released = 0
        self._waiters: list[tuple[Callable[[ObservedLock], bool], tonio.Event]] = []

    @classmethod
    def install(cls, backend) -> ObservedLock:
        observed = cls(backend._async_lock)
        backend._async_lock = observed
        return observed

    async def __aenter__(self):
        self._record("arrived")
        return await self._inner.__aenter__()

    async def __aexit__(self, *exc_info):
        result = await self._inner.__aexit__(*exc_info)
        self._record("released")
        return result

    def _record(self, counter: str) -> None:
        with self._mutex:
            setattr(self, counter, getattr(self, counter) + 1)
            reached = [waiter for waiter in self._waiters if waiter[0](self)]
            for waiter in reached:
                self._waiters.remove(waiter)
        for _predicate, event in reached:
            event.set()

    async def until(self, predicate: Callable[[ObservedLock], bool]) -> None:
        reached = tonio.Event()
        with self._mutex:
            if predicate(self):
                return
            self._waiters.append((predicate, reached))
        await reached.wait(5)
        assert reached.is_set(), f"lock never reached the state (arrived={self.arrived}, released={self.released})"


def park_on_file_lock_retry(monkeypatch) -> tonio.Event:
    """Returns an Event set when an operation finds the file lock held and
    starts backing off: `_acquire_lock_async` sleeps through the `clock.sleep_ms`
    seam between attempts, and the wrapper sets the Event right before that
    (cancellable) sleep."""
    parked = tonio.Event()
    original = clock.sleep_ms

    async def sleep_ms(ms, cancel=None):
        parked.set()
        await original(ms, cancel)

    monkeypatch.setattr(clock, "sleep_ms", sleep_ms)
    return parked
