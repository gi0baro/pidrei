"""setTimeout/setInterval equivalents for the tui package.

No pi counterpart: pi leans on the JS event loop's global timers. Here each
timer is a small tonio task that ends cooperatively through an Event (scope
cancellation would not unwind an aborted task, so nothing relies on it).

`cancel()` is best-effort like an Event can make it: a firing callback that
already passed the cancellation check will still run. Owners that need
clearTimeout/clearInterval determinism must re-check identity/generation
under their own lock inside the callback (see StdinBuffer/ProcessTerminal).

`fn` must return an awaitable (async-only callback policy); the result is
awaited rather than dropped. Both timers sit on the input path — the
StdinBuffer flush and the Kitty negotiation flush both re-enter input
handling, which is async — so a coroutine returned here must not be
discarded.
"""

import tonio.colored as tonio


class Timeout:
    """One-shot timer: run `fn` after `delay_ms` unless cancelled first."""

    def __init__(self, delay_ms: float, fn) -> None:
        self._cancelled = tonio.Event()
        tonio.spawn.without_tracking(self._run(delay_ms / 1000, fn))

    async def _run(self, delay_s: float, fn) -> None:
        await self._cancelled.wait(delay_s)
        if self._cancelled.is_set():
            return
        await fn()

    def cancel(self) -> None:
        self._cancelled.set()


class Interval:
    """Repeating timer: run `fn` every `delay_ms` until cancelled."""

    def __init__(self, delay_ms: float, fn) -> None:
        self._cancelled = tonio.Event()
        tonio.spawn.without_tracking(self._run(delay_ms / 1000, fn))

    async def _run(self, delay_s: float, fn) -> None:
        while True:
            await self._cancelled.wait(delay_s)
            if self._cancelled.is_set():
                return
            await fn()

    def cancel(self) -> None:
        self._cancelled.set()
