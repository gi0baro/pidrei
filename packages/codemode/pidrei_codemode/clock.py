"""The codemode package's clock seam.

pidrei_codemode depends on no other pidrei package, so it cannot use
`pidrei_ai.utils.clock`; this is its counterpart, and the only place in the
package that reads a clock (ruff bans the stdlib reads elsewhere).
"""

from tonio.colored import time as tonio_time


def monotonic() -> float:
    """Seconds on the runtime's clock, for intervals: the clock tonio's timers
    and `Event.wait` timeouts measure against. Needs a runtime."""
    return tonio_time.time()
