"""Mirror of pi tui src/wheel-scroll.ts."""

import math
import os
import sys
from typing import Literal


# Lines moved per mouse-wheel event, or "auto" to accelerate fast wheel spins.
type WheelScrollLines = int | Literal["auto"]

# Several events closer than this belong to one physical notch (Ghostty emits them ~4 ms apart)
# or come from a high-resolution source. They move one line each and do not accelerate.
BURST_GAP_MS = 5
# A pause longer than this ends a scroll gesture.
GESTURE_GAP_MS = 200
# Average event gap that maps to one line per event. Faster events scale up proportionally.
REFERENCE_GAP_MS = 100
MAX_AUTO_LINES = 6


def _terminal_accelerates_wheel() -> bool:
    """Local macOS terminals receive wheel and trackpad deltas that the OS has already accelerated,
    and they emit one event per line. Other platforms, and SSH sessions where the client platform
    is unknown, usually send one event per wheel notch.
    """
    env = os.environ
    return (
        sys.platform == "darwin"
        and env.get("SSH_CONNECTION") is None
        and env.get("SSH_CLIENT") is None
        and env.get("SSH_TTY") is None
    )


class WheelScrollAccelerator:
    """Converts wheel events into line counts.

    In "auto" mode on terminals that do not accelerate wheel input, the count follows event
    velocity: an isolated notch moves one line, while a fast spin moves up to six lines per event.
    For example, notches 100 ms apart move 1 line each, 50 ms apart move 2, and 20 ms apart move 5.

    Not synchronized: the owner (`TuiAltScreen`) touches it under the UI state lock only.
    """

    def __init__(self, lines: WheelScrollLines = "auto", accelerate: bool | None = None) -> None:
        self._lines = lines
        self._accelerate = not _terminal_accelerates_wheel() if accelerate is None else accelerate
        self._last_time = -math.inf
        self._last_direction = 0
        self._average_gap: float | None = None
        self._carry = 0.0

    def set_lines(self, lines: WheelScrollLines) -> None:
        self._lines = lines
        self._reset()

    def next(self, direction: int, now: float) -> int:
        """Return the positive line count for a wheel event in `direction` (-1 or 1) at time `now`
        (milliseconds)."""
        if self._lines != "auto":
            return max(1, math.floor(self._lines)) if math.isfinite(self._lines) else 1
        if not self._accelerate:
            return 1

        gap = now - self._last_time
        same_gesture = direction == self._last_direction and gap <= GESTURE_GAP_MS
        self._last_time = now
        self._last_direction = direction
        if not same_gesture:
            self._average_gap = None
            self._carry = 0.0
            return 1
        if gap < BURST_GAP_MS:
            return 1

        self._average_gap = gap if self._average_gap is None else (self._average_gap + gap) / 2
        lines = min(MAX_AUTO_LINES, max(1, REFERENCE_GAP_MS / self._average_gap)) + self._carry
        whole = math.floor(lines)
        self._carry = lines - whole
        return whole

    def _reset(self) -> None:
        self._last_time = -math.inf
        self._last_direction = 0
        self._average_gap = None
        self._carry = 0.0
