"""Rate-limited state publication (port of pi `harness/utils/adaptive-publisher.ts`).

pi paces publications with `Date.now()` and a single `setTimeout`, driven in
tests by vitest fake timers. The wall clock comes from the `clock` seam the
OAuth flows already virtualize, and the trailing timer goes through
`_set_timeout` — a module attribute tests replace with a manual timer queue,
the same substitution vitest performs. Producers push from several tonio tasks
(the execution env's stdout/stderr readers), so the state sits behind a lock;
pi's single thread needs none.
"""

import threading
from collections.abc import Callable
from typing import Any

import tonio.colored as tonio

from pidrei_ai.utils import clock


def _set_timeout(delay_ms: float, callback: Callable[[], None]) -> Callable[[], None]:
    """`setTimeout`: run `callback` after `delay_ms` unless the returned cancel is called first."""
    cancelled = tonio.Event()

    async def run() -> None:
        await cancelled.wait(delay_ms / 1000)
        if not cancelled.is_set():
            callback()

    tonio.spawn.without_tracking(run())
    return cancelled.set


class AdaptivePublisher[TValue, TUpdate]:
    """Publishes the latest state without queuing intermediate mutations.

    The first dirty state after idle is immediate. Each publication then buys a
    delay proportional to its encoded size, with a minimum interval that also
    bounds event count. A single trailing timer guarantees eventual publication.
    """

    def __init__(
        self,
        *,
        snapshot: Callable[[], TValue],
        update: Callable[[TValue | None, TValue], TUpdate | None],
        measure: Callable[[TUpdate], int],
        publish: Callable[[TUpdate], None],
        on_error: Callable[[Exception], None],
        min_interval_ms: float = 100,
        target_bytes_per_second: float = 100 * 1024,
    ) -> None:
        self._snapshot = snapshot
        self._update = update
        self._measure = measure
        self._publish = publish
        self._on_error = on_error
        self._min_interval_ms = min_interval_ms
        self._target_bytes_per_second = target_bytes_per_second
        self._lock = threading.RLock()
        self._published: Any = None
        self._has_published = False
        self._dirty = False
        self._next_emit_at = 0.0
        self._cancel_timer: Callable[[], None] | None = None
        self._disposed = False

    def mark_dirty(self) -> None:
        with self._lock:
            if self._disposed:
                return
            self._dirty = True
            wait = self._next_emit_at - clock.now_ms()
            if wait <= 0:
                self.flush()
                return
            self._arm_timer(wait)

    def flush(self, force: bool = False) -> None:
        with self._lock:
            if self._disposed or not self._dirty:
                return
            now = clock.now_ms()
            if not force and now < self._next_emit_at:
                self._arm_timer(self._next_emit_at - now)
                return
            self._clear_timer()
            current = self._snapshot()
            update = self._update(self._published if self._has_published else None, current)
            if update is None:
                self._published = current
                self._has_published = True
                self._dirty = False
                return
            encoded_bytes = self._measure(update)
            self._published = current
            self._has_published = True
            self._dirty = False
            self._next_emit_at = now + max(self._min_interval_ms, encoded_bytes * 1000 / self._target_bytes_per_second)
            # Commit before delivery. A consumer may apply the update and then raise
            # or reenter the producer; retaining the old baseline would duplicate
            # that delta.
            self._publish(update)

    def dispose(self) -> None:
        with self._lock:
            self._clear_timer()
            self._disposed = True

    def _clear_timer(self) -> None:
        if self._cancel_timer is not None:
            self._cancel_timer()
            self._cancel_timer = None

    def _arm_timer(self, wait_ms: float) -> None:
        if self._cancel_timer is not None:
            return

        def fire() -> None:
            with self._lock:
                self._cancel_timer = None
            try:
                self.flush()
            except Exception as error:
                self._on_error(error)

        self._cancel_timer = _set_timeout(wait_ms, fire)
