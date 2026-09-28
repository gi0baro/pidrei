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
from datetime import UTC, datetime

from tonio.colored import time as tonio_time

from pidrei_ai.utils.cancel import AbortError, CancelToken


# The one place pidrei reads the wall clock (ruff bans the stdlib reads
# elsewhere). `now_iso` and `now_datetime` derive from `now_ms`, so a test that
# swaps `clock.now_ms` moves all three.


def now_ms() -> int:
    """Mirror of `Date.now()`: Unix time in milliseconds."""
    return int(time.time() * 1000)


def now_datetime() -> datetime:
    """The current time as an aware UTC datetime (`.astimezone()` for local)."""
    return datetime.fromtimestamp(now_ms() / 1000, UTC)


def now_iso() -> str:
    """Mirror of `new Date().toISOString()`: UTC, millisecond precision, `Z`."""
    return now_datetime().isoformat(timespec="milliseconds").replace("+00:00", "Z")


def monotonic() -> float:
    """Seconds on the runtime's clock, for intervals and deadlines: the clock
    tonio's timers and `Event.wait` timeouts measure against. Needs a runtime;
    there is deliberately no fallback clock (two origins would corrupt deltas)."""
    return tonio_time.time()


async def sleep_ms(ms: float, cancel: CancelToken | None = None) -> None:
    """Sleep `ms` milliseconds, raising `AbortError` if `cancel` fires first.

    Callers that need pi's flow-specific cancellation message catch `AbortError`
    and re-raise. The sleep is the token's own wait with a timeout, so no
    extra task is involved and an already-cancelled token never waits.
    """
    seconds = ms / 1000
    if cancel is None:
        await tonio_time.sleep(seconds)
        return
    if not cancel.cancelled:
        await cancel.wait(seconds)
    if cancel.cancelled:
        raise AbortError("Operation was aborted")
