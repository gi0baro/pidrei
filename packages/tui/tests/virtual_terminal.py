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
import weakref

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
        self._reply_handler = None
        self._resize_handler = None
        self._screen = pyte.HistoryScreen(columns, rows, history=_HISTORY)
        self._stream = pyte.Stream(self._screen)
        self._frames = 0
        # pyte's parser is a single generator: the render loop (terminal.write)
        # and a test task (hide_cursor via show/hide_overlay) feed it from two
        # worker threads, which raises "generator already executing" — or
        # silently corrupts the screen. pi never has this: one JS thread.
        self._feed_lock = threading.Lock()
        # `until` waiters, re-checked after every write (from the render
        # loop's writer task as well as test tasks).
        self._waiters_lock = threading.Lock()
        self._waiters: list[_Waiter] = []
        # The TUI driving this terminal, for `settle()`. `TUI.start` passes
        # its bound `request_render` as the resize callback, so it is the
        # callback's `__self__`; a restart by another TUI (mode switch)
        # re-points it at the renderer now in charge.
        self._tui = None

    async def start(self, on_input, on_resize, on_reply=None, on_error=None) -> None:
        self._input_handler = on_input
        self._reply_handler = on_reply
        self._resize_handler = on_resize
        self._tui = getattr(on_resize, "__self__", None)
        if self._tui is not None:
            _RenderWatch.install(self._tui)
        # Enable bracketed paste mode for consistency with ProcessTerminal
        self._feed("\x1b[?2004h")

    async def drain_input(self, max_ms: float = 1000, idle_ms: float = 50) -> None:
        """No-op for virtual terminal - no stdin to drain."""

    def arm(self) -> None:
        """No-op: writes land in the screen emulator at once."""

    async def release(self) -> None:
        """No-op, as `arm`."""

    async def stop(self) -> None:
        # Disable bracketed paste mode
        self._feed("\x1b[?2004l")
        self._input_handler = None
        self._reply_handler = None
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
        self._notify_waiters()

    def _notify_waiters(self) -> None:
        with self._waiters_lock:
            waiters = list(self._waiters)
        for waiter in waiters:
            waiter.check()

    async def until(self, predicate, timeout: float = 5.0) -> bool:
        """Wait until `predicate()` holds, re-checked after every write.

        Event-driven replacement for sleep-polling: fits any state that is
        visible no later than a terminal write — the viewport, a recording
        subclass's write log (they log before delegating here), the frame
        counter, and TUI state published before its frame goes out (focus,
        the visible screen, the layout: `_compose_frame` publishes them, then
        the frame is written). Registered first, then checked, so a write landing
        in between is not missed. Bounded: a condition that never holds
        returns False to the caller's assertion instead of hanging the suite.
        """
        waiter = _Waiter(predicate)
        with self._waiters_lock:
            self._waiters.append(waiter)
        try:
            waiter.check()
            await waiter.reached.wait(timeout)
        finally:
            with self._waiters_lock:
                self._waiters.remove(waiter)
        if waiter.error is not None:
            raise waiter.error
        return waiter.reached.is_set()

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

    def write_sync(self, data: str) -> None:
        self._feed(data)

    def set_progress(self, active: bool) -> None:
        pass

    def set_program_status(self, _status) -> None:
        pass

    async def close(self) -> None:
        pass

    # Test-specific methods not in the Terminal protocol

    async def send_input(self, data: str) -> None:
        """Simulate keyboard input: offered to ``on_reply`` first, as
        ProcessTerminal's reader does, then awaited as its consumer does."""
        on_reply = self._reply_handler
        if on_reply is not None and on_reply(data):
            return
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
        self._notify_waiters()
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
        render loop runs as a separate task, so under
        load (the full suite, a busy CI runner) the frame could land after the
        sleep and the assertion would read a stale viewport. That produced a
        long-standing flake across the overlay/focus suites: different test
        names each time, never reproducible when the file ran alone, and
        originally misdiagnosed as order-dependent.

        Pass `since` — the frame count captured *before* requesting the render
        — to wait for a frame after it, then for the TUI to settle: requests
        made back to back can each get their own frame (there is no
        throttle to merge them), so the first new frame need not cover them
        all. Without `since` this only settles (`settle()`): most callers do
        not request a render at all, so waiting for a frame that never comes
        would cost the timeout each time. (It used to be a 50ms
        settle-sleep, the same hope in a smaller dose.)

        `timeout` bounds the wait so a render that never lands fails the
        assertion it was blocking, rather than hanging the suite.
        """
        if since is None:
            if self._frames == 0:
                # Nothing has been drawn yet, so the caller is waiting for the
                # first frame `start()` requested — wait for that one (on a
                # slow runner a search typed before the first layout anchored
                # on the implicit scroll view).
                await self.until(lambda: self._frames > 0, timeout)
            await self.settle()
            return
        await self.until(lambda: self._frames > since, timeout)
        await self.settle()

    async def settle(self, timeout: float = 5.0) -> None:
        """Wait until the TUI is render-idle: every render requested so far
        has been drawn, and every frame is on the wire.

        `_RenderWatch` counts the TUI's render requests and, per frame, the
        requests made before the frame began: those it covers, since each
        request's change comes before its send. Settled once a frame covers
        every request made before this call (or the TUI stopped rendering).
        Work the TUI does off the render loop (a detached copy, a spawned
        query) is not covered: wait for its own signal. Bounded: a frame
        that never comes fails the caller instead of hanging the suite.
        """
        tui = self._tui
        assert tui is not None, "settle() needs a TUI started on this terminal"
        watch = _RenderWatch.install(tui)
        target = watch.requested()
        while True:
            frame_drawn = watch.pending(target)
            if frame_drawn is None or not tui._render_active:
                break
            await frame_drawn.wait(timeout)
            assert frame_drawn.is_set(), "no frame covered the render requests"
        await tui._flush_frames()


class _RenderWatch:
    """Test-side bookkeeping for `VirtualTerminal.settle`, installed on a TUI
    by wrapping its request sender (`_render_requests`, replaced at each
    start) and its `_render_frame` (once)."""

    _watches: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requested = 0
        self._covered = 0
        self._frame_drawn = tonio.Event()

    @classmethod
    def install(cls, tui) -> _RenderWatch:
        watch = cls._watches.get(tui)
        if watch is None:
            watch = cls._watches[tui] = cls()
            render_frame = tui._render_frame

            async def counted_frame() -> None:
                covered = watch.requested()
                await render_frame()
                watch.drawn(covered)

            tui._render_frame = counted_frame
        requests = tui._render_requests
        if requests is not None and not isinstance(requests, _CountingSender):
            tui._render_requests = _CountingSender(requests, watch)
        return watch

    def count_request(self) -> None:
        with self._lock:
            self._requested += 1

    def requested(self) -> int:
        with self._lock:
            return self._requested

    def drawn(self, covered: int) -> None:
        with self._lock:
            self._covered = max(self._covered, covered)
            frame_drawn, self._frame_drawn = self._frame_drawn, tonio.Event()
        frame_drawn.set()

    def pending(self, target: int):
        """None once `target` requests are covered, else the event the next
        frame sets."""
        with self._lock:
            return None if self._covered >= target else self._frame_drawn


class _CountingSender:
    def __init__(self, sender, watch: _RenderWatch) -> None:
        self._sender = sender
        self._watch = watch

    def send_nowait(self, message):
        # Counted before the send: once the token is in the channel the loop
        # can start the frame at once, and that frame must see this request
        # in the count it covers. A send refused as closed needs nothing:
        # `settle` returns once the TUI stops rendering.
        self._watch.count_request()
        return self._sender.send_nowait(message)

    def close(self) -> None:
        self._sender.close()


class _Waiter:
    """One `VirtualTerminal.until` registration."""

    __slots__ = ("error", "predicate", "reached")

    def __init__(self, predicate) -> None:
        self.predicate = predicate
        self.reached = tonio.Event()
        self.error: BaseException | None = None

    def check(self) -> None:
        if self.reached.is_set():
            return
        try:
            hit = self.predicate()
        except Exception as error:
            # Raised to the waiting test, never into the TUI's frame writer.
            self.error = error
            hit = True
        if hit:
            self.reached.set()


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
