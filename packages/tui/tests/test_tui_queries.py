"""Mirror of pi tui test/terminal-colors.test.ts (TUI query cases) and
test/tui-cell-size-input.test.ts."""

import contextlib

import pytest
import tonio.colored as tonio

from pidrei_tui.terminal_image import get_cell_dimensions, reset_capabilities_cache, set_cell_dimensions
from pidrei_tui.tui import TUI

from .tui_helpers import env_var


class TestTerminal:
    """Minimal recording terminal (pi's TestTerminal)."""

    __test__ = False  # not a pytest class

    def __init__(self, column_count=80, row_count=24):
        self._column_count = column_count
        self._row_count = row_count
        self._input_handler = None
        self._resize_handler = None
        self.writes = []

    async def start(self, on_input, on_resize):
        self._input_handler = on_input
        self._resize_handler = on_resize

    async def stop(self):
        self._input_handler = None
        self._resize_handler = None

    async def drain_input(self, max_ms=1000, idle_ms=50):
        pass

    async def write(self, data):
        self.writes.append(data)

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

    async def send_input(self, data):
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

    async def handle_input(self, data):
        self.inputs.append(data)

    def invalidate(self):
        pass


# TUI.queryTerminalBackgroundColor


@pytest.mark.tonio
async def test_writes_osc11_query_and_resolves_with_the_parsed_rgb_reply():
    terminal = TestTerminal()
    tui = TUI(terminal)
    await tui.start()
    try:

        async def query():
            return await tui.query_terminal_background_color(timeout_ms=1000)

        async def reply():
            await tonio.sleep(0.01)
            assert "\x1b]11;?\x07" in terminal.writes
            await terminal.send_input("\x1b]11;#ffffff\x07")

        result, _ = await tonio.spawn(query(), reply())
        assert result == {"r": 255, "g": 255, "b": 255}
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_consumes_osc11_replies_before_input_listeners_and_focused_component_dispatch():
    terminal = TestTerminal()
    tui = TUI(terminal)
    component = InputRecorder()
    listener_inputs = []
    tui.add_child(component)
    tui.set_focus(component)
    tui.add_input_listener(lambda data: listener_inputs.append(data))
    await tui.start()
    try:

        async def query():
            return await tui.query_terminal_background_color(timeout_ms=1000)

        async def reply():
            await tonio.sleep(0.01)
            await terminal.send_input("\x1b]11;#000000\x07")

        result, _ = await tonio.spawn(query(), reply())
        assert result == {"r": 0, "g": 0, "b": 0}
        assert listener_inputs == []
        assert component.inputs == []
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_consumes_unparseable_strict_osc11_replies_and_resolves_none():
    terminal = TestTerminal()
    tui = TUI(terminal)
    component = InputRecorder()
    listener_inputs = []
    tui.add_child(component)
    tui.set_focus(component)
    tui.add_input_listener(lambda data: listener_inputs.append(data))
    await tui.start()
    try:

        async def query():
            return await tui.query_terminal_background_color(timeout_ms=1000)

        async def reply():
            await tonio.sleep(0.01)
            await terminal.send_input("\x1b]11;not-a-color\x07")

        result, _ = await tonio.spawn(query(), reply())
        assert result is None
        assert listener_inputs == []
        assert component.inputs == []
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_dispatches_non_matching_input_normally_while_waiting_for_an_osc11_reply():
    terminal = TestTerminal()
    tui = TUI(terminal)
    component = InputRecorder()
    listener_inputs = []
    tui.add_child(component)
    tui.set_focus(component)
    tui.add_input_listener(lambda data: listener_inputs.append(data))
    await tui.start()
    try:
        state = {"settled": False}

        async def query():
            result = await tui.query_terminal_background_color(timeout_ms=1000)
            state["settled"] = True
            return result

        async def interact():
            await tonio.sleep(0.01)
            await terminal.send_input("x")
            await tonio.sleep(0.01)
            assert state["settled"] is False
            assert listener_inputs == ["x"]
            assert component.inputs == ["x"]
            await terminal.send_input("\x1b]11;#ffffff\x07")

        result, _ = await tonio.spawn(query(), interact())
        assert result == {"r": 255, "g": 255, "b": 255}
    finally:
        await tui.stop()


@pytest.mark.tonio
async def test_keeps_consuming_a_late_osc11_reply_after_timeout():
    terminal = TestTerminal()
    tui = TUI(terminal)
    component = InputRecorder()
    listener_inputs = []
    tui.add_child(component)
    tui.set_focus(component)
    tui.add_input_listener(lambda data: listener_inputs.append(data))
    await tui.start()
    try:
        result = await tui.query_terminal_background_color(timeout_ms=1)
        assert result is None

        await terminal.send_input("\x1b]11;#ffffff\x07")

        assert listener_inputs == []
        assert component.inputs == []
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
        tui = TUI(terminal)
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
        tui = TUI(terminal)
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
