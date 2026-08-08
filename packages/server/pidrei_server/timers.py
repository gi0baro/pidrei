"""Node one-shot timer mapping (`setTimeout`/`clearTimeout`) on tonio.

The timer task parks on an `Event` with a timeout: `cancel()` sets the event,
which both suppresses the callback and wakes the task immediately so no
runtime work lingers for the full delay. Node's `unref()` has no tonio
counterpart (spawned tasks never pin the runtime) and is dropped.
"""

from collections.abc import Callable

import tonio.colored as tonio


class Timer:
    """One-shot timer: run `callback` after `delay_ms` unless cancelled."""

    __slots__ = ("_cancelled",)

    def __init__(self, delay_ms: float, callback: Callable[[], object]) -> None:
        self._cancelled = tonio.Event()
        tonio.spawn.without_tracking(self._run(delay_ms / 1000, callback))

    async def _run(self, delay: float, callback: Callable[[], object]) -> None:
        await self._cancelled.wait(delay)
        if not self._cancelled.is_set():
            callback()

    def cancel(self) -> None:
        self._cancelled.set()
