"""Port of pi tui test/virtual-terminal.ts on top of pyte.

pi uses @xterm/headless; pyte differences the harness papers over:

- pyte has no APC support (Kitty graphics / cursor marker); APC sequences are
  recorded in the write log for assertions but stripped before feeding pyte.
- pyte cannot distinguish written spaces from untouched cells, so
  ``get_viewport`` right-strips every row (xterm's translateToString(true)
  keeps written spaces); expectations in the mirrored tests are adjusted
  accordingly.
- pyte's resize neither scrolls into nor restores from history; ``resize``
  re-creates the screen and re-feeds the scrollback+display text
  bottom-anchored (xterm.js reflows), losing cell attributes — none of the
  mirrored resize cases assert attributes.
- ``\\x1b[3J`` (scrollback clear) is not applied to pyte history; no mirrored
  case reads the scroll buffer after it.
"""

import re
import time

import pyte
import tonio.colored as tonio


_APC_RE = re.compile(r"\x1b_(?:[^\x07\x1b]|\x1b(?!\\))*(?:\x07|\x1b\\)")

_HISTORY = 500


class VirtualTerminal:
    """Virtual terminal for testing using pyte for terminal emulation."""

    def __init__(self, columns: int = 80, rows: int = 24) -> None:
        self._columns = columns
        self._rows = rows
        self._input_handler = None
        self._resize_handler = None
        self._screen = pyte.HistoryScreen(columns, rows, history=_HISTORY)
        self._stream = pyte.Stream(self._screen)
        self._frames = 0

    async def start(self, on_input, on_resize) -> None:
        self._input_handler = on_input
        self._resize_handler = on_resize
        # Enable bracketed paste mode for consistency with ProcessTerminal
        self._feed("\x1b[?2004h")

    async def drain_input(self, max_ms: float = 1000, idle_ms: float = 50) -> None:
        """No-op for virtual terminal - no stdin to drain."""

    async def stop(self) -> None:
        # Disable bracketed paste mode
        self._feed("\x1b[?2004l")
        self._input_handler = None
        self._resize_handler = None

    def _feed(self, data: str) -> None:
        if "\x1b_" in data:
            data = _APC_RE.sub("", data)
        self._stream.feed(data)

    async def write(self, data: str) -> None:
        self._feed(data)
        # Frame counter for wait_for_render(); see its docstring.
        self._frames += 1

    @property
    def frames(self) -> int:
        """Number of writes the TUI has made — one or more per rendered frame."""
        return self._frames

    @property
    def columns(self) -> int:
        return self._columns

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def kitty_protocol_active(self) -> bool:
        # Virtual terminal always reports Kitty protocol as active for testing
        return True

    def move_by(self, lines: int) -> None:
        if lines > 0:
            self._feed(f"\x1b[{lines}B")
        elif lines < 0:
            self._feed(f"\x1b[{-lines}A")

    def hide_cursor(self) -> None:
        self._feed("\x1b[?25l")

    def show_cursor(self) -> None:
        self._feed("\x1b[?25h")

    def clear_line(self) -> None:
        self._feed("\x1b[K")

    def clear_from_cursor(self) -> None:
        self._feed("\x1b[J")

    def clear_screen(self) -> None:
        self._feed("\x1b[2J\x1b[H")

    def set_title(self, title: str) -> None:
        self._feed(f"\x1b]0;{title}\x07")

    def set_progress(self, active: bool) -> None:
        pass

    # Test-specific methods not in the Terminal protocol

    async def send_input(self, data: str) -> None:
        """Simulate keyboard input."""
        if self._input_handler is not None:
            await self._input_handler(data)

    def resize(self, columns: int, rows: int) -> None:
        """Resize the terminal (xterm-like: content stays bottom-anchored)."""
        buffer_lines = self.get_scroll_buffer()
        # Drop trailing blank rows like xterm's shrink does before scrolling.
        while buffer_lines and not buffer_lines[-1]:
            buffer_lines.pop()

        self._columns = columns
        self._rows = rows
        self._screen = pyte.HistoryScreen(columns, rows, history=_HISTORY)
        self._stream = pyte.Stream(self._screen)
        if buffer_lines:
            self._feed("\r\n".join(buffer_lines))
        if self._resize_handler is not None:
            self._resize_handler()

    def get_viewport(self) -> list[str]:
        """Get the visible viewport (what's currently on screen), right-stripped."""
        return [line.rstrip() for line in self._screen.display]

    def get_scroll_buffer(self) -> list[str]:
        """Get the entire scroll buffer (history + viewport), right-stripped."""
        lines: list[str] = []
        columns = self._columns
        for row in self._screen.history.top:
            lines.append("".join(row[x].data for x in range(columns)).rstrip())
        lines.extend(self.get_viewport())
        return lines

    def get_cursor_position(self) -> dict:
        return {"x": self._screen.cursor.x, "y": self._screen.cursor.y}

    def get_cell_italic(self, row: int, col: int) -> int:
        char = self._screen.buffer[row][col]
        return 1 if char.italics else 0

    def get_cell_underline(self, row: int, col: int) -> int:
        char = self._screen.buffer[row][col]
        return 1 if char.underscore else 0

    async def wait_for_render(self, since: int | None = None, timeout: float = 5.0) -> None:
        """Wait until the TUI has actually written a frame.

        This used to be `await tonio.sleep(0.05)` — a hope, not a wait. The
        render loop is throttled to 16ms but runs as a separate task, so under
        load (the full suite, a busy CI runner) the frame could land after the
        sleep and the assertion would read a stale viewport. That produced a
        long-standing flake across the overlay/focus suites: different test
        names each time, never reproducible when the file ran alone, and
        originally misdiagnosed as order-dependent.

        Pass `since` — the frame count captured *before* requesting the render
        — to actually wait for that frame. Without it this keeps the original
        settle-sleep, because most callers do not request a render at all and
        waiting for a frame that never comes would cost the timeout each time
        (measured: 152 tests went from ~3s to 41s).

        `timeout` bounds the wait so a render that never lands fails the
        assertion it was blocking, rather than hanging the suite.
        """
        if since is None:
            await tonio.sleep(0.05)
            return
        deadline = time.monotonic() + timeout
        while self._frames <= since:
            if time.monotonic() >= deadline:
                return
            await tonio.sleep(0.001)


class LoggingVirtualTerminal(VirtualTerminal):
    """VirtualTerminal that records every write for assertions."""

    def __init__(self, columns: int = 80, rows: int = 24) -> None:
        super().__init__(columns, rows)
        self._writes: list[str] = []

    async def write(self, data: str) -> None:
        self._writes.append(data)
        await super().write(data)

    def get_writes(self) -> str:
        return "".join(self._writes)

    def clear_writes(self) -> None:
        self._writes = []
