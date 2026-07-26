"""Wall clock and interruptible sleep behind a single seam.

pi reads `Date.now()` and `setTimeout` directly and drives them from tests with
vitest fake timers. tonio has no fake clock. `utils/retry.py` could get away
with real delays because its backoffs are sub-100 ms, but the OAuth device-code
flows wait 5 s between polls and up to 15 minutes for a deadline, and clamp the
poll interval to a 1 s minimum — real delays there would be neither fast nor
stable. So every time-dependent step in those flows goes through these two
functions, and the tests replace them with a virtual clock: the same
substitution vitest performs, at a seam pidrei owns.
"""

import time

import tonio.colored as tonio
from tonio.colored import time as tonio_time

from pidrei_ai.utils.cancel import AbortError, CancelToken


def now_ms() -> int:
    """Mirror of `Date.now()`: Unix time in milliseconds."""
    return int(time.time() * 1000)


_SLEPT = object()
_CANCELLED = object()


async def sleep_ms(ms: float, cancel: CancelToken | None = None) -> None:
    """Sleep `ms` milliseconds, raising `AbortError` if `cancel` fires first.

    Callers that need pi's flow-specific cancellation message catch `AbortError`
    and re-raise; the token is checked before sleeping so an already-cancelled
    token never waits.
    """
    if cancel is not None and cancel.cancelled:
        raise AbortError("Operation was aborted")
    seconds = ms / 1000
    if cancel is None:
        await tonio_time.sleep(seconds)
        return

    async def _timer() -> object:
        await tonio_time.sleep(seconds)
        return _SLEPT

    async def _aborted() -> object:
        await cancel.wait()
        return _CANCELLED

    if await tonio.select(_timer(), _aborted()) is _CANCELLED:
        raise AbortError("Operation was aborted")
