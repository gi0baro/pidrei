"""The TUI's wall-clock seam.

pidrei_tui depends on no other pidrei package, so it cannot use
`pidrei_ai.utils.clock`; this is its counterpart, and the only place in the TUI
that reads the wall clock (ruff bans the stdlib reads elsewhere).
"""

import time
from datetime import UTC, datetime

from tonio.colored import time as tonio_time


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
