"""Mirror of pi tui test/tui-render.test.ts.

Viewport expectations are right-stripped (see virtual_terminal.py).
"""

import os
from contextlib import contextmanager

import pytest
import tonio.colored as tonio

from pidrei_tui import tui as tui_module
from pidrei_tui.components.image import Image
from pidrei_tui.terminal_image import (
    delete_kitty_image,
    encode_kitty,
    reset_capabilities_cache,
    set_capabilities,
    set_cell_dimensions,
)
from pidrei_tui.tui_main_screen import TuiMainScreen

from .tui_helpers import env_var
from .virtual_terminal import LoggingVirtualTerminal, VirtualTerminal


class TestComponent:
    __test__ = False  # not a pytest class

    def __init__(self):
        self.lines = []

    def render(self, width):
        return self.lines

    def invalidate(self):
        pass


class InputComponent(TestComponent):
    __test__ = False

    def __init__(self):
        super().__init__()
        self.render_count = 0

    def render(self, width):
        self.render_count += 1
        return super().render(width)

    async def handle_input(self, data):
        self.lines = [data]


# TUI render scheduling


@pytest.mark.tonio
async def test_renders_keyboard_input_without_waiting_for_a_throttled_frame():
    """pi asserts one render on the next tick; the throttle here is a real
    wait, so it is stretched to 2s for the test — a preempted frame lands in
    milliseconds, a throttled one could not."""
    terminal = VirtualTerminal(40, 10)
    tui = TuiMainScreen(terminal)
    component = InputComponent()
    component.lines = ["initial"]
    tui.add_child(component)
    tui.set_focus(component)
    await tui.start()
    await terminal.wait_for_render()
    render_count_before_input = component.render_count

    original_interval = tui_module._MIN_RENDER_INTERVAL_S
    tui_module._MIN_RENDER_INTERVAL_S = 2.0
    try:
        # Queue a normal throttled render first, and let the loop park in the
        # throttle. Keyboard input must preempt it.
        component.lines = ["pending"]
        since = terminal.frames
        tui.request_render()
        await tonio.sleep(0.02)
        assert terminal.frames == since, "the throttled frame should still be pending"

        await terminal.send_input("first")
        await terminal.send_input("second")
        await terminal.send_input("typed")
        await terminal.wait_for_render(since, timeout=1.0)
    finally:
        tui_module._MIN_RENDER_INTERVAL_S = original_interval

    assert terminal.frames > since, "keyboard input should not wait for the throttle"
    # How many of the three inputs coalesce into one frame is a property of
    # the machine, not of the preemption: a slower box lets the render loop
    # wake between them and renders more than once (CI saw 2 extra renders).
    # What this test pins is that a render happened at all inside the 1s wait
    # while the throttle was 2s, so only "more than before" is asserted here.
    assert component.render_count > render_count_before_input
    assert component.lines == ["typed"]
    await tui.stop()


# TUI debug logging


@pytest.mark.tonio
async def test_writes_redraw_logs_to_the_provided_directory(tmp_path):
    with env_var("PIDREI_DEBUG_REDRAW", "1"):
        terminal = VirtualTerminal(40, 10)
        tui = TuiMainScreen(terminal, None, str(tmp_path))
        component = TestComponent()
        tui.add_child(component)
        component.lines = ["test"]
        await tui.start()
        await terminal.wait_for_render()

        with open(os.path.join(str(tmp_path), "pidrei-debug.log"), encoding="utf-8") as log_file:
            assert "fullRender: first render" in log_file.read()
        await tui.stop()


# TUI Kitty image cleanup (encode_kitty-based cases)


@pytest.mark.tonio
async def test_deletes_changed_image_ids_before_drawing_moved_placements():
    terminal = LoggingVirtualTerminal(40, 10)
    tui = TuiMainScreen(terminal)
    component = TestComponent()
    tui.add_child(component)

    old_image = encode_kitty("AAAA", columns=2, rows=2, image_id=42, move_cursor=False)
    component.lines = ["top", old_image]
    await tui.start()
    await terminal.wait_for_render()
    terminal.clear_writes()

    new_image = encode_kitty("BBBB", columns=2, rows=1, image_id=42, move_cursor=False)
    component.lines = [new_image, ""]
    tui.request_render()
    await terminal.wait_for_render()

    writes = terminal.get_writes()
    delete_index = writes.find(delete_kitty_image(42))
    draw_index = writes.find(new_image)
    assert delete_index >= 0, "changed old image should be deleted"
    assert draw_index >= 0, "new image should be drawn"
    assert delete_index < draw_index, "old image must be deleted before the new placement is drawn"

    await tui.stop()


@pytest.mark.tonio
async def test_redraws_image_lines_when_an_earlier_reserved_image_row_changes():
    terminal = LoggingVirtualTerminal(40, 10)
    tui = TuiMainScreen(terminal)
    component = TestComponent()
    tui.add_child(component)

    image = encode_kitty("AAAA", columns=2, rows=2, image_id=88, move_cursor=False)
    component.lines = ["", image]
    await tui.start()
    await terminal.wait_for_render()
    terminal.clear_writes()

    component.lines = ["covered", image]
    tui.request_render()
    await terminal.wait_for_render()

    writes = terminal.get_writes()
    delete_index = writes.find(delete_kitty_image(88))
    draw_index = writes.find(image)
    assert delete_index >= 0, "image should be deleted when a reserved row changes"
    assert draw_index >= 0, "unchanged image line should be redrawn after deleting the placement"
    assert delete_index < draw_index, "old placement must be deleted before the image line is redrawn"
    assert "\x1b[2J" not in writes, "reserved row changes should not force a full redraw"

    await tui.stop()


@pytest.mark.tonio
async def test_deletes_previously_rendered_image_ids_during_full_redraws():
    terminal = LoggingVirtualTerminal(40, 10)
    tui = TuiMainScreen(terminal)
    component = TestComponent()
    tui.add_child(component)

    component.lines = [encode_kitty("AAAA", columns=2, rows=2, image_id=77, move_cursor=False)]
    await tui.start()
    await terminal.wait_for_render()
    terminal.clear_writes()

    component.lines = ["plain text"]
    tui.request_render(True)
    await terminal.wait_for_render()

    writes = terminal.get_writes()
    delete_index = writes.find(delete_kitty_image(77))
    clear_index = writes.find("\x1b[2J")
    assert delete_index >= 0, "previous image should be deleted during full redraw"
    assert clear_index >= 0, "full redraw should clear the screen"
    assert delete_index < clear_index, "old image should be deleted before the screen is cleared"

    await tui.stop()


# TUI resize handling


@pytest.mark.tonio
async def test_triggers_full_re_render_when_terminal_height_changes():
    with env_var("TERMUX_VERSION", None):
        terminal = VirtualTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        component = TestComponent()
        tui.add_child(component)

        component.lines = ["Line 0", "Line 1", "Line 2"]
        await tui.start()
        await terminal.wait_for_render()

        initial_redraws = tui.full_redraws

        # Resize height
        terminal.resize(40, 15)
        await terminal.wait_for_render()

        # Should have triggered a full redraw
        assert tui.full_redraws > initial_redraws, "Height change should trigger full redraw"

        viewport = terminal.get_viewport()
        assert "Line 0" in viewport[0], "Content preserved after height change"

        await tui.stop()


@pytest.mark.tonio
async def test_skips_full_re_render_on_height_changes_in_termux():
    with env_var("TERMUX_VERSION", "1"):
        terminal = LoggingVirtualTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        component = TestComponent()
        tui.add_child(component)

        component.lines = [f"Line {i}" for i in range(20)]
        await tui.start()
        await terminal.wait_for_render()
        terminal.clear_writes()

        initial_redraws = tui.full_redraws
        for height in [15, 8, 14, 11]:
            terminal.resize(40, height)
            await terminal.wait_for_render()

        assert tui.full_redraws == initial_redraws, "Height change should not trigger full redraw"
        assert "\x1b[2J" not in terminal.get_writes(), "Height change should not clear the screen"
        assert "\x1b[3J" not in terminal.get_writes(), "Height change should not clear scrollback"

        viewport = terminal.get_viewport()
        assert "Line 19" in "\n".join(viewport), "Latest content remains visible after resize"

        await tui.stop()


@pytest.mark.tonio
async def test_triggers_full_re_render_when_terminal_width_changes():
    terminal = VirtualTerminal(40, 10)
    tui = TuiMainScreen(terminal)
    component = TestComponent()
    tui.add_child(component)

    component.lines = ["Line 0", "Line 1", "Line 2"]
    await tui.start()
    await terminal.wait_for_render()

    initial_redraws = tui.full_redraws

    # Resize width
    terminal.resize(60, 10)
    await terminal.wait_for_render()

    # Should have triggered a full redraw
    assert tui.full_redraws > initial_redraws, "Width change should trigger full redraw"

    await tui.stop()


# TUI content shrinkage


@pytest.mark.tonio
async def test_clears_empty_rows_when_content_shrinks_significantly():
    terminal = VirtualTerminal(40, 10)
    tui = TuiMainScreen(terminal)
    tui.set_clear_on_shrink(True)  # Explicitly enable (may be disabled via env var)
    component = TestComponent()
    tui.add_child(component)

    # Start with many lines
    component.lines = ["Line 0", "Line 1", "Line 2", "Line 3", "Line 4", "Line 5"]
    await tui.start()
    await terminal.wait_for_render()

    initial_redraws = tui.full_redraws

    # Shrink to fewer lines
    component.lines = ["Line 0", "Line 1"]
    tui.request_render()
    await terminal.wait_for_render()

    # Should have triggered a full redraw to clear empty rows
    assert tui.full_redraws > initial_redraws, "Content shrinkage should trigger full redraw"

    viewport = terminal.get_viewport()
    assert "Line 0" in viewport[0], "First line preserved"
    assert "Line 1" in viewport[1], "Second line preserved"
    # Lines below should be empty (cleared)
    assert viewport[2].strip() == "", "Line 2 should be cleared"
    assert viewport[3].strip() == "", "Line 3 should be cleared"

    await tui.stop()


@pytest.mark.tonio
async def test_handles_shrink_to_single_line():
    terminal = VirtualTerminal(40, 10)
    tui = TuiMainScreen(terminal)
    tui.set_clear_on_shrink(True)
    component = TestComponent()
    tui.add_child(component)

    component.lines = ["Line 0", "Line 1", "Line 2", "Line 3"]
    await tui.start()
    await terminal.wait_for_render()

    # Shrink to single line
    component.lines = ["Only line"]
    tui.request_render()
    await terminal.wait_for_render()

    viewport = terminal.get_viewport()
    assert "Only line" in viewport[0], "Single line rendered"
    assert viewport[1].strip() == "", "Line 1 should be cleared"

    await tui.stop()


@pytest.mark.tonio
async def test_handles_shrink_to_empty():
    terminal = VirtualTerminal(40, 10)
    tui = TuiMainScreen(terminal)
    tui.set_clear_on_shrink(True)
    component = TestComponent()
    tui.add_child(component)

    component.lines = ["Line 0", "Line 1", "Line 2"]
    await tui.start()
    await terminal.wait_for_render()

    # Shrink to empty
    component.lines = []
    tui.request_render()
    await terminal.wait_for_render()

    viewport = terminal.get_viewport()
    # All lines should be empty
    assert viewport[0].strip() == "", "Line 0 should be cleared"
    assert viewport[1].strip() == "", "Line 1 should be cleared"

    await tui.stop()


# TUI differential rendering


@pytest.mark.tonio
async def test_tracks_cursor_correctly_when_content_shrinks_with_unchanged_remaining_lines():
    terminal = VirtualTerminal(40, 10)
    tui = TuiMainScreen(terminal)
    component = TestComponent()
    tui.add_child(component)

    # Initial render: 5 identical lines
    component.lines = ["Line 0", "Line 1", "Line 2", "Line 3", "Line 4"]
    await tui.start()
    await terminal.wait_for_render()

    # Shrink to 3 lines, all identical to before (no content changes in remaining lines)
    component.lines = ["Line 0", "Line 1", "Line 2"]
    tui.request_render()
    await terminal.wait_for_render()

    # cursor_row should be 2 (last line of new content)
    # Verify by doing another render with a change on line 1
    component.lines = ["Line 0", "CHANGED", "Line 2"]
    tui.request_render()
    await terminal.wait_for_render()

    viewport = terminal.get_viewport()
    # Line 1 should show "CHANGED", proving cursor tracking was correct
    assert "CHANGED" in viewport[1], f'Expected "CHANGED" on line 1, got: {viewport[1]}'

    await tui.stop()


@pytest.mark.tonio
async def test_renders_correctly_when_only_a_middle_line_changes_spinner_case():
    terminal = VirtualTerminal(40, 10)
    tui = TuiMainScreen(terminal)
    component = TestComponent()
    tui.add_child(component)

    # Initial render
    component.lines = ["Header", "Working...", "Footer"]
    await tui.start()
    await terminal.wait_for_render()

    # Simulate spinner animation - only middle line changes
    for frame in ["|", "/", "-", "\\"]:
        component.lines = ["Header", f"Working {frame}", "Footer"]
        tui.request_render()
        await terminal.wait_for_render()

        viewport = terminal.get_viewport()
        assert "Header" in viewport[0], f"Header preserved: {viewport[0]}"
        assert f"Working {frame}" in viewport[1], f"Spinner updated: {viewport[1]}"
        assert "Footer" in viewport[2], f"Footer preserved: {viewport[2]}"

    await tui.stop()


@pytest.mark.tonio
async def test_resets_styles_after_each_rendered_line():
    terminal = VirtualTerminal(20, 6)
    tui = TuiMainScreen(terminal)
    component = TestComponent()
    tui.add_child(component)

    component.lines = ["\x1b[3mItalic", "Plain"]
    await tui.start()
    await terminal.wait_for_render()

    assert terminal.get_cell_italic(1, 0) == 0
    await tui.stop()


@pytest.mark.tonio
async def test_renders_correctly_when_first_line_changes_but_rest_stays_same():
    terminal = VirtualTerminal(40, 10)
    tui = TuiMainScreen(terminal)
    component = TestComponent()
    tui.add_child(component)

    component.lines = ["Line 0", "Line 1", "Line 2", "Line 3"]
    await tui.start()
    await terminal.wait_for_render()

    # Change only first line
    component.lines = ["CHANGED", "Line 1", "Line 2", "Line 3"]
    tui.request_render()
    await terminal.wait_for_render()

    viewport = terminal.get_viewport()
    assert "CHANGED" in viewport[0], f"First line changed: {viewport[0]}"
    assert "Line 1" in viewport[1], f"Line 1 preserved: {viewport[1]}"
    assert "Line 2" in viewport[2], f"Line 2 preserved: {viewport[2]}"
    assert "Line 3" in viewport[3], f"Line 3 preserved: {viewport[3]}"

    await tui.stop()


@pytest.mark.tonio
async def test_renders_correctly_when_last_line_changes_but_rest_stays_same():
    terminal = VirtualTerminal(40, 10)
    tui = TuiMainScreen(terminal)
    component = TestComponent()
    tui.add_child(component)

    component.lines = ["Line 0", "Line 1", "Line 2", "Line 3"]
    await tui.start()
    await terminal.wait_for_render()

    # Change only last line
    component.lines = ["Line 0", "Line 1", "Line 2", "CHANGED"]
    tui.request_render()
    await terminal.wait_for_render()

    viewport = terminal.get_viewport()
    assert "Line 0" in viewport[0], f"Line 0 preserved: {viewport[0]}"
    assert "Line 1" in viewport[1], f"Line 1 preserved: {viewport[1]}"
    assert "Line 2" in viewport[2], f"Line 2 preserved: {viewport[2]}"
    assert "CHANGED" in viewport[3], f"Last line changed: {viewport[3]}"

    await tui.stop()


@pytest.mark.tonio
async def test_renders_correctly_when_multiple_non_adjacent_lines_change():
    terminal = VirtualTerminal(40, 10)
    tui = TuiMainScreen(terminal)
    component = TestComponent()
    tui.add_child(component)

    component.lines = ["Line 0", "Line 1", "Line 2", "Line 3", "Line 4"]
    await tui.start()
    await terminal.wait_for_render()

    # Change lines 1 and 3, keep 0, 2, 4 the same. Frame-counted wait: the
    # settle-sleep default lost this race on a loaded macOS runner.
    frame = terminal.frames
    component.lines = ["Line 0", "CHANGED 1", "Line 2", "CHANGED 3", "Line 4"]
    tui.request_render()
    await terminal.wait_for_render(frame)

    viewport = terminal.get_viewport()
    assert "Line 0" in viewport[0], f"Line 0 preserved: {viewport[0]}"
    assert "CHANGED 1" in viewport[1], f"Line 1 changed: {viewport[1]}"
    assert "Line 2" in viewport[2], f"Line 2 preserved: {viewport[2]}"
    assert "CHANGED 3" in viewport[3], f"Line 3 changed: {viewport[3]}"
    assert "Line 4" in viewport[4], f"Line 4 preserved: {viewport[4]}"

    await tui.stop()


@pytest.mark.tonio
async def test_handles_transition_from_content_to_empty_and_back_to_content():
    terminal = VirtualTerminal(40, 10)
    tui = TuiMainScreen(terminal)
    component = TestComponent()
    tui.add_child(component)

    # Start with content
    component.lines = ["Line 0", "Line 1", "Line 2"]
    await tui.start()
    await terminal.wait_for_render()

    viewport = terminal.get_viewport()
    assert "Line 0" in viewport[0], "Initial content rendered"

    # Clear to empty
    component.lines = []
    tui.request_render()
    await terminal.wait_for_render()

    # Add content back - this should work correctly even after empty state
    component.lines = ["New Line 0", "New Line 1"]
    tui.request_render()
    await terminal.wait_for_render()

    viewport = terminal.get_viewport()
    assert "New Line 0" in viewport[0], f"New content rendered: {viewport[0]}"
    assert "New Line 1" in viewport[1], f"New content line 1: {viewport[1]}"

    await tui.stop()


@pytest.mark.tonio
async def test_full_re_renders_when_deleted_lines_move_the_viewport_upward():
    terminal = VirtualTerminal(20, 5)
    tui = TuiMainScreen(terminal)
    component = TestComponent()
    tui.add_child(component)

    component.lines = [f"Line {i}" for i in range(12)]
    await tui.start()
    await terminal.wait_for_render()

    initial_redraws = tui.full_redraws

    component.lines = [f"Line {i}" for i in range(7)]
    tui.request_render()
    await terminal.wait_for_render()

    assert tui.full_redraws > initial_redraws, "Shrink should trigger a full redraw"
    assert terminal.get_viewport() == ["Line 2", "Line 3", "Line 4", "Line 5", "Line 6"]

    await tui.stop()


@pytest.mark.tonio
async def test_appends_after_a_shrink_without_another_full_redraw_once_the_viewport_is_reset():
    terminal = VirtualTerminal(20, 5)
    tui = TuiMainScreen(terminal)
    component = TestComponent()
    tui.add_child(component)

    component.lines = [f"Line {i}" for i in range(8)]
    await tui.start()
    await terminal.wait_for_render()

    initial_redraws = tui.full_redraws

    component.lines = ["Line 0", "Line 1"]
    tui.request_render()
    await terminal.wait_for_render()

    assert tui.full_redraws > initial_redraws, "Shrink should reset the viewport with a full redraw"
    redraws_after_shrink = tui.full_redraws

    component.lines = ["Line 0", "Line 1", "Line 2"]
    tui.request_render()
    await terminal.wait_for_render()

    assert tui.full_redraws == redraws_after_shrink, "Append should stay on the differential path"
    assert terminal.get_viewport() == ["Line 0", "Line 1", "Line 2", "", ""]

    await tui.stop()


@pytest.mark.tonio
async def test_clears_stale_content_when_max_lines_rendered_was_inflated_by_a_transient_component():
    terminal = VirtualTerminal(40, 10)
    tui = TuiMainScreen(terminal)
    chat = TestComponent()
    editor = TestComponent()
    tui.add_child(chat)
    tui.add_child(editor)

    long_chat = [f"Chat {i}" for i in range(15)]
    short_chat = [f"Chat {i}" for i in range(12)]
    editor_lines = ["Editor 0", "Editor 1", "Editor 2"]
    selector_lines = [f"Selector {i}" for i in range(8)]

    chat.lines = long_chat
    editor.lines = editor_lines
    await tui.start()
    await terminal.wait_for_render()

    editor.lines = selector_lines
    tui.request_render()
    await terminal.wait_for_render()

    editor.lines = editor_lines
    tui.request_render()
    await terminal.wait_for_render()

    redraws_before_switch = tui.full_redraws
    chat.lines = short_chat
    tui.request_render()
    await terminal.wait_for_render()

    assert tui.full_redraws > redraws_before_switch, "Branch switch should trigger a full redraw"

    viewport = terminal.get_viewport()
    for i in range(10):
        line = viewport[i]
        assert "Chat 12" not in line, f'Stale "Chat 12" at viewport row {i}'
        assert "Chat 13" not in line, f'Stale "Chat 13" at viewport row {i}'
        assert "Chat 14" not in line, f'Stale "Chat 14" at viewport row {i}'

    assert viewport == [
        "Chat 5",
        "Chat 6",
        "Chat 7",
        "Chat 8",
        "Chat 9",
        "Chat 10",
        "Chat 11",
        "Editor 0",
        "Editor 1",
        "Editor 2",
    ]

    await tui.stop()


# TUI Kitty image handling (Image-component cases, deferred from the renderer slice)


@contextmanager
def kitty_capabilities():
    set_capabilities({"images": "kitty", "trueColor": True, "hyperlinks": True})
    set_cell_dimensions({"widthPx": 10, "heightPx": 10})
    try:
        yield
    finally:
        reset_capabilities_cache()
        set_cell_dimensions({"widthPx": 9, "heightPx": 18})


@pytest.mark.tonio
async def test_clears_reserved_kitty_image_rows_before_drawing_appended_image_placements():
    with kitty_capabilities():
        terminal = LoggingVirtualTerminal(40, 10)
        tui = TuiMainScreen(terminal)
        component = TestComponent()
        tui.add_child(component)

        component.lines = ["before"]
        await tui.start()
        await terminal.wait_for_render()
        terminal.clear_writes()

        image = Image(
            "AAAA",
            "image/png",
            {"fallbackColor": lambda value: value},
            {"maxWidthCells": 2},
            {"widthPx": 20, "heightPx": 20},
        )
        image_lines = image.render(40)
        image_sequence = image_lines[0]
        component.lines = ["before", *image_lines, "after"]
        tui.request_render()
        await terminal.wait_for_render()

        writes = terminal.get_writes()
        assert f"\x1b[2K\r\n\x1b[2K\x1b[1A{image_sequence}\x1b[1B" in writes, (
            "reserved rows should be cleared before the image placement is drawn"
        )
        assert f"{image_sequence}\r\n\x1b[2K" not in writes, (
            "reserved row clears must not run after the image placement is drawn"
        )

        await tui.stop()


@pytest.mark.tonio
async def test_falls_back_to_full_redraw_when_kitty_image_pre_clear_would_scroll():
    with kitty_capabilities():
        terminal = LoggingVirtualTerminal(40, 2)
        tui = TuiMainScreen(terminal)
        component = TestComponent()
        tui.add_child(component)

        component.lines = ["before"]
        await tui.start()
        await terminal.wait_for_render()
        redraws_before_image = tui.full_redraws
        terminal.clear_writes()

        image = Image(
            "AAAA",
            "image/png",
            {"fallbackColor": lambda value: value},
            {"maxWidthCells": 3},
            {"widthPx": 30, "heightPx": 30},
        )
        component.lines = ["before", *image.render(40), "after"]
        tui.request_render()
        await terminal.wait_for_render()

        assert tui.full_redraws > redraws_before_image, "unsafe image pre-clear should force a full redraw"
        assert "\x1b[2J" in terminal.get_writes(), "fallback should clear and fully redraw"

        await tui.stop()


@pytest.mark.tonio
async def test_reserves_kitty_image_rows_before_drawing_during_full_redraw_fallbacks():
    with kitty_capabilities():
        terminal = LoggingVirtualTerminal(40, 5)
        tui = TuiMainScreen(terminal)
        component = TestComponent()
        tui.add_child(component)

        component.lines = ["l0", "l1", "l2", "l3", "l4"]
        await tui.start()
        await terminal.wait_for_render()
        redraws_before_image = tui.full_redraws
        terminal.clear_writes()

        image = Image(
            "AAAA",
            "image/png",
            {"fallbackColor": lambda value: value},
            {"maxWidthCells": 3},
            {"widthPx": 30, "heightPx": 30},
        )
        image_lines = image.render(40)
        image_sequence = image_lines[0]
        component.lines = ["l0", "l1", "l2", "l3", "l4", *image_lines, "after"]
        tui.request_render()
        await terminal.wait_for_render()

        writes = terminal.get_writes()
        assert tui.full_redraws > redraws_before_image, "scrolling image append should force a full redraw"
        assert f"\r\n\r\n\x1b[2A{image_sequence}\x1b[2B" in writes, (
            "full redraw should reserve visible image rows before drawing the placement"
        )
        assert f"{image_sequence}\r\n\x1b[0m" not in writes, (
            "full redraw must not write reserved padding rows after drawing the placement"
        )

        await tui.stop()


@pytest.mark.tonio
async def test_does_not_use_cursor_up_placement_for_kitty_images_taller_than_the_viewport():
    with kitty_capabilities():
        terminal = LoggingVirtualTerminal(40, 5)
        tui = TuiMainScreen(terminal)
        component = TestComponent()
        tui.add_child(component)

        component.lines = ["before"]
        await tui.start()
        await terminal.wait_for_render()
        terminal.clear_writes()

        image = Image(
            "AAAA",
            "image/png",
            {"fallbackColor": lambda value: value},
            {"maxWidthCells": 6},
            {"widthPx": 60, "heightPx": 60},
        )
        image_lines = image.render(40)
        image_sequence = image_lines[0]
        assert len(image_lines) > terminal.rows, "test image should exceed the viewport height"

        component.lines = ["before", *image_lines, "after"]
        tui.request_render(True)
        await terminal.wait_for_render()

        writes = terminal.get_writes()
        assert image_sequence in writes, "image placement should be drawn"
        assert f"\x1b[{len(image_lines) - 1}A{image_sequence}" not in writes, (
            "taller-than-viewport images must keep the #4461 first-row placement path"
        )

        await tui.stop()


# single-writer discipline: the render loop is the only task writing terminal
# bytes while it runs (PLAN "cursor writes through the render loop")


class CursorCallCountingTerminal(LoggingVirtualTerminal):
    """Counts the sync cursor methods, which bypass the render loop's writes."""

    def __init__(self, columns: int = 80, rows: int = 24) -> None:
        super().__init__(columns, rows)
        self.sync_cursor_calls = 0

    def hide_cursor(self) -> None:
        self.sync_cursor_calls += 1
        super().hide_cursor()

    def show_cursor(self) -> None:
        self.sync_cursor_calls += 1
        super().show_cursor()


@pytest.mark.tonio
async def test_overlay_and_cursor_changes_write_cursor_state_only_from_the_render_loop():
    """pi hides the cursor synchronously from showOverlay/hideOverlay/
    setShowHardwareCursor. Tasks run on several OS threads here, so those
    bytes could land inside a frame; the cursor state is instead carried by
    every frame's tail, written by the render loop alone."""
    terminal = CursorCallCountingTerminal(40, 10)
    tui = TuiMainScreen(terminal, show_hardware_cursor=True)
    component = TestComponent()
    component.lines = ["content"]
    tui.add_child(component)
    await tui.start()
    await terminal.wait_for_render()
    # start() hides the cursor before the loop exists; only what follows counts.
    terminal.sync_cursor_calls = 0

    overlay = TestComponent()
    overlay.lines = ["overlay"]
    since = terminal.frames
    handle = tui.show_overlay(overlay, {"width": 10})
    await terminal.wait_for_render(since, timeout=1.0)
    assert terminal.get_writes().endswith("\x1b[?25l"), "frame tail must carry the cursor state"

    since = terminal.frames
    handle.hide()
    await terminal.wait_for_render(since, timeout=1.0)

    since = terminal.frames
    tui.set_show_hardware_cursor(False)
    tui.request_render(True)  # unchanged content renders as a cursor-only write, not a frame
    await terminal.wait_for_render(since, timeout=1.0)
    assert terminal.get_writes().endswith("\x1b[?25l")

    assert terminal.sync_cursor_calls == 0, "no task other than the render loop may write cursor bytes"
    await tui.stop()


# The ownership contract (the island): mutation and render are owner work


@pytest.mark.tonio
async def test_posted_mutation_lands_before_the_frame_it_requests():
    """The island's replacement for the old `post_before_render` hook: a
    mutation posted to the UI owner runs before the frame its
    `request_render()` schedules, never in between that frame's component
    renders — mutation and render are serialized on one task."""
    terminal = VirtualTerminal(40, 10)
    tui = TuiMainScreen(terminal)
    component = InputComponent()
    component.lines = ["before"]
    tui.add_child(component)
    await tui.start()
    await terminal.wait_for_render()

    since = terminal.frames
    render_count = component.render_count

    async def mutate() -> None:
        assert component.render_count == render_count
        component.lines = ["after"]
        tui.request_render()

    tui.input_owner.post(mutate)
    await terminal.wait_for_render(since, timeout=1.0)

    assert component.render_count > render_count
    assert component.lines == ["after"]
    assert any("after" in line for line in terminal.get_viewport())
    await tui.stop()


@pytest.mark.tonio
async def test_renders_requested_from_other_tasks_funnel_onto_the_owner():
    """Renders requested from arbitrary tasks and input through the terminal
    both land as owner work — frames and key effects keep flowing."""
    terminal = VirtualTerminal(40, 10)
    tui = TuiMainScreen(terminal)
    component = InputComponent()
    component.lines = ["initial"]
    tui.add_child(component)
    tui.set_focus(component)
    await tui.start()
    await terminal.wait_for_render()

    since = terminal.frames

    async def request_from_elsewhere() -> None:
        tui.request_render(True)

    async with tonio.scope() as scope:
        scope.spawn(request_from_elsewhere())
    await terminal.wait_for_render(since, timeout=1.0)
    assert terminal.frames > since

    since = terminal.frames
    await terminal.send_input("typed")
    await terminal.wait_for_render(since, timeout=1.0)
    assert component.lines == ["typed"]
    await tui.stop()


@pytest.mark.tonio
async def test_render_error_is_handed_to_the_installed_handler():
    terminal = VirtualTerminal(40, 10)
    tui = TuiMainScreen(terminal)
    component = TestComponent()
    tui.add_child(component)
    await tui.start()
    await terminal.wait_for_render()

    failed = tonio.Event()
    seen: list = []

    async def on_error(error) -> None:
        seen.append(error)
        failed.set()

    tui.set_render_error_handler(on_error)

    def broken_render(_width):
        raise RuntimeError("torn cache")

    component.render = broken_render
    tui.request_render()
    await failed.wait(1.0)

    assert failed.is_set()
    assert isinstance(seen[0], RuntimeError)
    await tui.stop()


@pytest.mark.tonio
async def test_next_frame_is_computed_while_the_previous_one_is_still_on_the_wire():
    # A slow terminal (SSH) must pace the loop, not serialize compute behind
    # flush: the frame after a blocked write is rendered before that write
    # completes, and no more than that (one-slot pipeline).
    terminal = VirtualTerminal(40, 10)
    release = tonio.Event()
    blocked = tonio.Event()
    slow_writes: list[str] = []
    original_write = terminal.write

    async def slow_write(data: str) -> None:
        if slow_writes or release.is_set():
            await original_write(data)
            return
        slow_writes.append(data)
        blocked.set()
        await release.wait(None)
        await original_write(data)

    tui = TuiMainScreen(terminal)
    component = InputComponent()
    component.lines = ["one"]
    tui.add_child(component)
    await tui.start()
    await terminal.wait_for_render()

    terminal.write = slow_write
    seen: dict[str, tonio.Event] = {"two": tonio.Event(), "three": tonio.Event()}
    original_render = component.render

    def render(width):
        lines = original_render(width)
        if lines and lines[0] in seen:
            seen[lines[0]].set()
        return lines

    component.render = render
    component.lines = ["two"]
    tui.request_render()
    await blocked.wait(1.0)
    assert blocked.is_set()
    await seen["two"].wait(1.0)
    # The frame holding "two" is stuck in the writer; a further frame must
    # still be computed while it waits.
    component.lines = ["three"]
    tui.request_render()
    await seen["three"].wait(1.0)
    assert seen["three"].is_set()
    assert not release.is_set()

    release.set()
    await tui.stop()
    assert any("three" in line for line in terminal.get_viewport())
