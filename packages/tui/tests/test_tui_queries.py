"""Mirror of pi tui test/terminal-colors.test.ts (TUI query cases) and
test/tui-cell-size-input.test.ts."""

import contextlib

import pytest
import tonio.colored as tonio

from pidrei_tui.terminal_image import get_cell_dimensions, reset_capabilities_cache, set_cell_dimensions
from pidrei_tui.tui_main_screen import TuiMainScreen

from .tui_helpers import env_var


class TestTerminal:
    """Minimal recording terminal (pi's TestTerminal)."""

    __test__ = False  # not a pytest class

    def __init__(self, column_count=80, row_count=24):
        self._column_count = column_count
        self._row_count = row_count
        self._input_handler = None
        self._reply_handler = None
        self._resize_handler = None
        self.writes = []
        self._expected_writes: list[tuple[str, tonio.Event]] = []

    def expect_write(self, text):
        """An Event set once `text` has been written. A query registers the reply
        it waits for before writing itself, so a test replies after this, never
        after a sleep."""
        written = tonio.Event()
        self._expected_writes.append((text, written))
        return written

    async def start(self, on_input, on_resize, on_reply=None, on_error=None):
        self._input_handler = on_input
        self._reply_handler = on_reply
        self._resize_handler = on_resize

    async def stop(self):
        self._input_handler = None
        self._reply_handler = None
        self._resize_handler = None

    async def drain_input(self, max_ms=1000, idle_ms=50):
        pass

    def arm(self):
        pass

    async def release(self):
        pass

    async def write(self, data):
        self.writes.append(data)
        for text, written in self._expected_writes:
            if text in data:
                written.set()

    @property
    def columns(self):
        return self._column_count

    @property
    def rows(self):
        return self._row_count

    @property
    def kitty_protocol_active(self):
        return False

    def move_by(self, lines):
        pass

    def hide_cursor(self):
        pass

    def show_cursor(self):
        pass

    def clear_line(self):
        pass

    def clear_from_cursor(self):
        pass

    def clear_screen(self):
        pass

    def set_title(self, title):
        pass

    def set_progress(self, active):
        pass

    async def close(self):
        pass

    async def send_input(self, data):
        # Replies are consumed where ProcessTerminal's reader offers them.
        on_reply = self._reply_handler
        if on_reply is not None and on_reply(data):
            return
        if self._input_handler is not None:
            await self._input_handler(data)

    def send_resize(self):
        if self._resize_handler is not None:
            self._resize_handler()


class InputRecorder:
    def __init__(self):
        self.inputs = []

    def render(self, width):
        return []

    def handle_input(self, data):
        self.inputs.append(data)

    def invalidate(self):
        pass


# TUI.queryTerminalColors
#
# pi awaits the query's promise and passes `onLateReply`; here every report
# reaches the `on_terminal_colors` listeners through the terminal-event loop,
# so the cases assert on what a listener received.

PALETTE_REPLIES = [f"\x1b]4;{index};#000000\x07" for index in range(16)]
DA1 = "\x1b[?62;22c"
BLACK = {"r": 0, "g": 0, "b": 0}
WHITE = {"r": 255, "g": 255, "b": 255}
QUERY_START = "\x1b]10;?\x07\x1b]11;?\x07\x1b]4;0;?\x07"


class ColorReports:
    """An `on_terminal_colors` listener recording the reports it gets."""

    def __init__(self):
        self.reports = []
        self._changed = tonio.Event()

    async def __call__(self, colors):
        self.reports.append(colors)
        changed, self._changed = self._changed, tonio.Event()
        changed.set()

    async def wait_for(self, count):
        while True:
            # Read before the check: a report landing in between sets it.
            changed = self._changed
            if len(self.reports) >= count:
                return
            await changed.wait(5)
            assert changed.is_set(), f"expected {count} reports, got {self.reports}"


async def setup_color_query():
    terminal = TestTerminal()
    tui = TuiMainScreen(terminal)
    component = InputRecorder()
    reports = ColorReports()
    tui.add_child(component)
    tui.set_focus(component)
    tui.on_terminal_colors(reports)
    await tui.start()
    return terminal, tui, component, reports


@pytest.mark.tonio
async def test_queries_all_colors_in_one_write_and_consumes_the_replies():
    terminal, tui, component, reports = await setup_color_query()
    try:
        written = terminal.expect_write(QUERY_START)
        applied = tui.query_terminal_colors(timeout_ms=1000)
        await written.wait(5)
        # pi reads the last write; here the render loop may write a frame
        # after the burst, so pick the burst out by its content.
        [burst] = [data for data in terminal.writes if QUERY_START in data]
        assert burst.startswith(QUERY_START) and burst.endswith("\x1b[c")

        await terminal.send_input("x")
        await terminal.send_input("\x1b]10;#ffffff\x07")
        await terminal.send_input("\x1b]11;rgb:0000/0000/0000\x1b\\")
        for reply in PALETTE_REPLIES:
            await terminal.send_input(reply)
        # Reported once every reply arrived, without waiting for DA1.
        await applied.wait(5)
        assert reports.reports == [{"foreground": WHITE, "background": BLACK, "palette": [BLACK] * 16}]
        await terminal.send_input(DA1)
        assert component.inputs == ["x"]
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_reports_on_da1_with_the_replies_that_arrived_in_query_order():
    terminal, tui, _component, reports = await setup_color_query()
    try:
        first = tui.query_terminal_colors(timeout_ms=1000)
        second = tui.query_terminal_colors(timeout_ms=1000)
        await terminal.send_input("\x1b]11;#000000\x07")
        # An incomplete palette is dropped.
        for reply in PALETTE_REPLIES[:8]:
            await terminal.send_input(reply)
        await terminal.send_input(DA1)
        await terminal.send_input(DA1)

        await first.wait(5)
        await second.wait(5)
        assert reports.reports == [
            {"foreground": None, "background": BLACK, "palette": None},
            {"foreground": None, "background": None, "palette": None},
        ]
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_reports_late_replies_after_a_timeout_and_consumes_them_until_da1():
    terminal, tui, component, reports = await setup_color_query()
    try:
        applied = tui.query_terminal_colors(timeout_ms=1)
        await applied.wait(5)
        assert reports.reports[0]["background"] is None

        await terminal.send_input("\x1b]11;#ffffff\x07")
        await terminal.send_input(DA1)
        await reports.wait_for(2)
        assert reports.reports[1:] == [{"foreground": None, "background": WHITE, "palette": None}]
        assert component.inputs == []

        # With no query pending, color replies are ordinary input again.
        await terminal.send_input("\x1b]11;#ffffff\x07")
        assert component.inputs == ["\x1b]11;#ffffff\x07"]
    finally:
        await tui.stop()


class FailingWriteTerminal(TestTerminal):
    __test__ = False

    async def write(self, data):
        if data.startswith(QUERY_START):
            raise OSError("terminal write failed")
        await super().write(data)


@pytest.mark.tonio
async def test_a_query_whose_write_fails_reports_no_colors_and_stops_collecting_replies():
    # pi's requestTerminalColors applies `{}` when the query fails; the
    # query's own coroutine does that here.
    terminal = FailingWriteTerminal()
    tui = TuiMainScreen(terminal)
    component = InputRecorder()
    reports = ColorReports()
    tui.set_focus(component)
    tui.on_terminal_colors(reports)
    await tui.start()
    try:
        applied = tui.query_terminal_colors(timeout_ms=1000)
        await applied.wait(5)
        assert reports.reports == [{"foreground": None, "background": None, "palette": None}]
        await terminal.send_input(DA1)
        assert component.inputs == [DA1]
    finally:
        await tui.stop()


# TUI cell size responses (tui-cell-size-input.test.ts)


@contextlib.contextmanager
def image_terminal():
    with env_var("TERM_PROGRAM", "ghostty"), env_var("TERM", None), env_var("GHOSTTY_RESOURCES_DIR", None):
        reset_capabilities_cache()
        try:
            yield
        finally:
            reset_capabilities_cache()


@pytest.mark.tonio
async def test_forwards_bare_escape_even_when_a_cell_size_query_was_sent_at_startup():
    with image_terminal():
        terminal = TestTerminal()
        tui = TuiMainScreen(terminal)
        recorder = InputRecorder()

        tui.set_focus(recorder)
        await tui.start()

        await terminal.send_input("\x1b")

        assert recorder.inputs == ["\x1b"]
        await tui.stop()


@pytest.mark.tonio
async def test_consumes_cell_size_responses_and_still_forwards_later_user_input():
    with image_terminal():
        set_cell_dimensions({"widthPx": 9, "heightPx": 18})

        terminal = TestTerminal()
        tui = TuiMainScreen(terminal)
        recorder = InputRecorder()

        tui.set_focus(recorder)
        await tui.start()

        await terminal.send_input("\x1b[6;20;10t")
        assert recorder.inputs == []
        assert get_cell_dimensions() == {"widthPx": 10, "heightPx": 20}

        await terminal.send_input("q")
        assert recorder.inputs == ["q"]
        await tui.stop()
        set_cell_dimensions({"widthPx": 9, "heightPx": 18})
