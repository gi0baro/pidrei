"""Mirror of pi tui src/components/alt-screen-flash.ts.

Transient messages the alternate-screen renderer composites over the top-right
of the viewport. Each entry expires on its own timer.
"""

from .._timers import Timeout
from ..utils import truncate_to_width


DEFAULT_DURATION_MS = 1000


class AltScreenFlashContainer:
    """Stack of transient flash messages. Entries are {"id", "message", "timer"}."""

    def __init__(self, request_render) -> None:
        self._entries: list[dict] = []
        self._next_id = 0
        self._request_render = request_render

    def flash(self, message: str, duration_ms: float | None = None) -> None:
        if duration_ms is None:
            duration_ms = DEFAULT_DURATION_MS
        entry_id = self._next_id
        self._next_id += 1

        async def expire() -> None:
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
