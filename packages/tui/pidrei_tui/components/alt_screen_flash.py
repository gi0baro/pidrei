"""Mirror of pi tui src/components/alt-screen-flash.ts.

Transient messages the alternate-screen renderer composites over the top-right
of the viewport. Each entry expires on its own timer.

Port deviation: the constructor also takes the renderer's UI state lock; an
entry expires on its timer's task, and the frame walks the entries under
that lock.
"""

from .._timers import Timeout
from ..utils import truncate_to_width


DEFAULT_DURATION_MS = 1000


class AltScreenFlashContainer:
    """Stack of transient flash messages. Entries are {"id", "message", "timer"}."""

    def __init__(self, request_render, state_lock) -> None:
        self._entries: list[dict] = []
        self._next_id = 0
        self._request_render = request_render
        self._state_lock = state_lock

    def flash(self, message: str, duration_ms: float | None = None) -> None:
        if duration_ms is None:
            duration_ms = DEFAULT_DURATION_MS
        entry_id = self._next_id
        self._next_id += 1

        def expire() -> None:
            with self._state_lock:
                for index, entry in enumerate(self._entries):
                    if entry["id"] == entry_id:
                        del self._entries[index]
                        self._request_render()
                        return

        timer = Timeout(max(0, duration_ms), expire)
        self._entries.append({"id": entry_id, "message": message, "timer": timer})
        self._request_render()

    def dispose(self) -> None:
        for entry in self._entries:
            entry["timer"].cancel()
        self._entries = []

    def invalidate(self) -> None:
        pass

    def render(self, width: int) -> list[str]:
        lines: list[str] = []
        for entry in self._entries:
            message = truncate_to_width(f" {entry['message']} ", width, "")
            lines.append(f"\x1b[7m{message}\x1b[27m")
        return lines
