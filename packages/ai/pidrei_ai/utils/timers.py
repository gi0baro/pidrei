"""`setTimeout`/`clearTimeout` behind a single seam.

pi arms timers with `setTimeout` and drives them from tests with vitest fake
timers. Callers here go through `timers.set_timeout(...)` (the module
attribute, never a from-import), so a test can swap it for a manual timer
queue — the same substitution vitest performs — alongside `clock.now_ms`.
"""

from collections.abc import Callable

import tonio.colored as tonio


def set_timeout(delay_ms: float, callback: Callable[[], None]) -> Callable[[], None]:
    """`setTimeout`: run `callback` after `delay_ms` unless the returned cancel is called first.

    The callback runs on its own task; it must not block and should spawn any
    async work it starts.
    """
    cancelled = tonio.Event()

    async def run() -> None:
        await cancelled.wait(delay_ms / 1000)
        if not cancelled.is_set():
            callback()

    tonio.spawn.without_tracking(run())
    return cancelled.set
