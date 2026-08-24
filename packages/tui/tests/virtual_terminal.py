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
import threading
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
        # pyte's parser is a single generator: the render loop (terminal.write)
        # and a test task (hide_cursor via show/hide_overlay) feed it from two
        # worker threads, which raises "generator already executing" — or
        # silently corrupts the screen. pi never has this: one JS thread.
        self._feed_lock = threading.Lock()

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
        with self._feed_lock:
            self._stream.feed(data)

    async def write(self, data: str) -> None:
        self._feed(data)
        # Frame counter for wait_for_render(); see its docstring. Only a
        # rendered frame counts: both renderers open one with synchronized
        # output. The other writes — the alt-screen enter/exit sequences, the
        # main screen's cursor positioning tail — are not frames, and a wait
        # returning on one of them reads a screen the layout never reached
        # (the alt-screen search test anchored on the implicit scroll view
        # that way: `start()`'s enter sequence satisfied its first wait).
        if "\x1b[?2026h" in data:
            self._frames += 1

    @property
    def frames(self) -> int:
        """Number of frames the TUI has rendered."""
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
        # The whole swap happens under the feed lock so a concurrent render
        # write cannot land in the half-rebuilt screen; the re-feed calls the
        # stream directly because the lock is not reentrant (the content is
        # plain text from the buffer, never APC).
        with self._feed_lock:
            buffer_lines = self._read_scroll_buffer_locked()
            # Drop trailing blank rows like xterm's shrink does before scrolling.
            while buffer_lines and not buffer_lines[-1]:
                buffer_lines.pop()

            self._columns = columns
            self._rows = rows
            self._screen = pyte.HistoryScreen(columns, rows, history=_HISTORY)
            self._stream = pyte.Stream(self._screen)
            if buffer_lines:
                self._stream.feed("\r\n".join(buffer_lines))
        if self._resize_handler is not None:
            self._resize_handler()

    # Screen readers take the feed lock: the render loop feeds from another
    # worker thread, and an unlocked read mid-feed returns a half-applied
    # frame (seen on macOS CI as one changed line updated, another not).

    def get_viewport(self) -> list[str]:
        """Get the visible viewport (what's currently on screen), right-stripped."""
        with self._feed_lock:
            return [line.rstrip() for line in self._screen.display]

    def _read_scroll_buffer_locked(self) -> list[str]:
        """History + viewport, right-stripped. Caller holds `_feed_lock`
        (which is not reentrant — `resize()` reads under its own hold)."""
        lines: list[str] = []
        columns = self._columns
        for row in self._screen.history.top:
            lines.append("".join(row[x].data for x in range(columns)).rstrip())
        lines.extend(line.rstrip() for line in self._screen.display)
        return lines

    def get_scroll_buffer(self) -> list[str]:
        """Get the entire scroll buffer (history + viewport), right-stripped."""
        with self._feed_lock:
            return self._read_scroll_buffer_locked()

    def get_cursor_position(self) -> dict:
        with self._feed_lock:
            return {"x": self._screen.cursor.x, "y": self._screen.cursor.y}

    def get_cell_italic(self, row: int, col: int) -> int:
        return 1 if self.get_cell(row, col).italics else 0

    def get_cell_underline(self, row: int, col: int) -> int:
        return 1 if self.get_cell(row, col).underscore else 0

    def get_cell(self, row: int, col: int):
        """The pyte ``Char`` at (row, col).

        pi reads xterm.js cell accessors (``isItalic()``, ``isFgDefault()``, …);
        pyte exposes the same state as plain attributes — ``italics``,
        ``underscore``, ``bold``, ``fg`` (``"default"`` or a colour name / hex
        string). pyte has no faint/dim attribute at all, so cases that assert on
        dim are not mirrored.

        pi folds its two file-local cell readers into this one accessor; here
        they are shared across suites, so ``get_cell_italic``/
        ``get_cell_underline`` stay for their other callers.
        """
        with self._feed_lock:
            return self._screen.buffer[row][col]

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
            if self._frames == 0:
                # Nothing has been drawn yet, so the caller is waiting for the
                # first frame `start()` requested — wait for that one rather
                # than hoping it lands inside the settle sleep (on a slow
                # runner it did not, and a search typed before the first
                # layout anchored on the implicit scroll view).
                since = 0
            else:
                await tonio.sleep(0.05)
                return
        deadline = time.monotonic() + timeout
        while self._frames <= since:
            if time.monotonic() >= deadline:
                return
            await tonio.sleep(0.001)


async def poll_until(check, timeout: float = 2.0):
    """Poll `check` until it returns a truthy value and return that value.

    For asynchronous effects with no completion signal (untracked spawns,
    debounced requests, the render loop's state publish): wait on the
    observable state itself instead of sleeping and hoping. Bounded like
    `wait_for_render` so a condition that never holds hands the last (falsy)
    value to the caller's assertion rather than hanging the suite.
    """
    deadline = time.monotonic() + timeout
    while True:
        value = check()
        if value or time.monotonic() >= deadline:
            return value
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
