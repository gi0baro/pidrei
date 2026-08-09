"""Mirror of pi tui test/overlay-options.test.ts, overlay-short-content.test.ts,
tui-overlay-style-leak.test.ts, regression-overlay-cjk-boundary.test.ts and the
overlay case from tab-width.test.ts."""

import pytest

from pidrei_tui.tui import composite_tui_line
from pidrei_tui.tui_main_screen import TuiMainScreen
from pidrei_tui.utils import extract_segments, slice_by_column, visible_width

from .virtual_terminal import LoggingVirtualTerminal, VirtualTerminal


class StaticOverlay:
    def __init__(self, lines, requested_width=None):
        self.lines = lines
        self.requested_width = requested_width

    def render(self, width):
        # Store the width we were asked to render at for verification
        self.requested_width = width
        return self.lines

    def invalidate(self):
        pass


class EmptyContent:
    def render(self, width):
        return []

    def invalidate(self):
        pass


class StaticLines:
    def __init__(self, lines):
        self.lines = lines

    def render(self, width):
        return self.lines

    def invalidate(self):
        pass


async def render_and_flush(tui, terminal):
    """Render and wait until the frame on screen reflects the current state.

    Two round-trips, not one. Waiting for a single frame is not enough: if the
    render loop is already inside `_do_render()` when the test mutates focus or
    overlay order, it writes a *stale* frame, the counter advances and the wait
    returns early — the assertion then reads the previous frame. The second
    request cannot start until the first has completed, so the frame it
    produces is guaranteed to have begun after the mutation.

    This replaced `await tonio.sleep(0.05)`, which waited for nothing at all
    and was the cause of a long-standing load-dependent flake across the focus
    and overlay suites.
    """
    for _ in range(2):
        before = terminal.frames
        tui.request_render(True)
        await terminal.wait_for_render(before)


# width overflow protection


@pytest.mark.tonio
async def test_truncates_overlay_lines_that_exceed_declared_width():
    terminal = VirtualTerminal(80, 24)
    tui = TuiMainScreen(terminal)
    # Overlay declares width 20 but renders lines much wider
    overlay = StaticOverlay(["X" * 100])

    tui.add_child(EmptyContent())
    tui.show_overlay(overlay, {"width": 20})
    await tui.start()
    await render_and_flush(tui, terminal)

    # Should not crash, and no line should exceed terminal width
    for line in terminal.get_viewport():
        assert line is not None
    await tui.stop()


@pytest.mark.tonio
async def test_handles_overlay_with_complex_ansi_sequences_without_crashing():
    terminal = VirtualTerminal(80, 24)
    tui = TuiMainScreen(terminal)
    # Simulate complex ANSI content like the crash log showed
    complex_line = (
        "\x1b[48;2;40;50;40m \x1b[38;2;128;128;128mSome styled content\x1b[39m\x1b[49m"
        "\x1b]8;;http://example.com\x07link\x1b]8;;\x07" + " more content " * 10
    )
    overlay = StaticOverlay([complex_line, complex_line, complex_line])

    tui.add_child(EmptyContent())
    tui.show_overlay(overlay, {"width": 60})
    await tui.start()
    await render_and_flush(tui, terminal)

    # Should not crash
    assert len(terminal.get_viewport()) > 0
    await tui.stop()


@pytest.mark.tonio
async def test_handles_overlay_composited_on_styled_base_content():
    terminal = VirtualTerminal(80, 24)
    tui = TuiMainScreen(terminal)

    class StyledContent:
        def render(self, width):
            styled_line = "\x1b[1m\x1b[38;2;255;0;0m" + "X" * width + "\x1b[0m"
            return [styled_line, styled_line, styled_line]

        def invalidate(self):
            pass

    overlay = StaticOverlay(["OVERLAY"])

    tui.add_child(StyledContent())
    tui.show_overlay(overlay, {"width": 20, "anchor": "center"})
    await tui.start()
    await render_and_flush(tui, terminal)

    # Should not crash and overlay should be visible
    viewport = terminal.get_viewport()
    assert any("OVERLAY" in line for line in viewport), "Overlay should be visible"
    await tui.stop()


@pytest.mark.tonio
async def test_handles_wide_characters_at_overlay_boundary():
    terminal = VirtualTerminal(80, 24)
    tui = TuiMainScreen(terminal)
    # Wide chars (each takes 2 columns) at the edge of declared width
    overlay = StaticOverlay(["中文日本語한글テスト漢字"])

    tui.add_child(EmptyContent())
    tui.show_overlay(overlay, {"width": 15})  # Odd width to potentially hit boundary
    await tui.start()
    await render_and_flush(tui, terminal)

    assert len(terminal.get_viewport()) > 0
    await tui.stop()


@pytest.mark.tonio
async def test_handles_overlay_positioned_at_terminal_edge():
    terminal = VirtualTerminal(80, 24)
    tui = TuiMainScreen(terminal)
    # Overlay positioned at right edge with content that exceeds declared width
    overlay = StaticOverlay(["X" * 50])

    tui.add_child(EmptyContent())
    # Position at col 60 with width 20 - should fit exactly at right edge
    tui.show_overlay(overlay, {"col": 60, "width": 20})
    await tui.start()
    await render_and_flush(tui, terminal)

    assert len(terminal.get_viewport()) > 0
    await tui.stop()


@pytest.mark.tonio
async def test_handles_overlay_on_base_content_with_osc_sequences():
    terminal = VirtualTerminal(80, 24)
    tui = TuiMainScreen(terminal)

    class HyperlinkContent:
        def render(self, width):
            link = "\x1b]8;;file:///path/to/file.ts\x07file.ts\x1b]8;;\x07"
            line = f"See {link} for details " + "X" * (width - 30)
            return [line, line, line]

        def invalidate(self):
            pass

    overlay = StaticOverlay(["OVERLAY-TEXT"])

    tui.add_child(HyperlinkContent())
    tui.show_overlay(overlay, {"anchor": "center", "width": 20})
    await tui.start()
    await render_and_flush(tui, terminal)

    # Should not crash - this was the original bug scenario
    assert len(terminal.get_viewport()) > 0
    await tui.stop()


# width percentage


@pytest.mark.tonio
async def test_renders_overlay_at_percentage_of_terminal_width():
    terminal = VirtualTerminal(100, 24)
    tui = TuiMainScreen(terminal)
    overlay = StaticOverlay(["test"])

    tui.add_child(EmptyContent())
    tui.show_overlay(overlay, {"width": "50%"})
    await tui.start()
    await render_and_flush(tui, terminal)

    assert overlay.requested_width == 50
    await tui.stop()


@pytest.mark.tonio
async def test_respects_min_width_when_width_percent_results_in_smaller_width():
    terminal = VirtualTerminal(100, 24)
    tui = TuiMainScreen(terminal)
    overlay = StaticOverlay(["test"])

    tui.add_child(EmptyContent())
    tui.show_overlay(overlay, {"width": "10%", "minWidth": 30})
    await tui.start()
    await render_and_flush(tui, terminal)

    assert overlay.requested_width == 30
    await tui.stop()


# anchor positioning


@pytest.mark.tonio
async def test_positions_overlay_at_top_left():
    terminal = VirtualTerminal(80, 24)
    tui = TuiMainScreen(terminal)
    overlay = StaticOverlay(["TOP-LEFT"])

    tui.add_child(EmptyContent())
    tui.show_overlay(overlay, {"anchor": "top-left", "width": 10})
    await tui.start()
    await render_and_flush(tui, terminal)

    viewport = terminal.get_viewport()
    assert viewport[0].startswith("TOP-LEFT"), f"Expected TOP-LEFT at start, got: {viewport[0]}"
    await tui.stop()


@pytest.mark.tonio
async def test_positions_overlay_at_bottom_right():
    terminal = VirtualTerminal(80, 24)
    tui = TuiMainScreen(terminal)
    overlay = StaticOverlay(["BTM-RIGHT"])

    tui.add_child(EmptyContent())
    tui.show_overlay(overlay, {"anchor": "bottom-right", "width": 10})
    await tui.start()
    await render_and_flush(tui, terminal)

    viewport = terminal.get_viewport()
    last_row = viewport[23]
    assert "BTM-RIGHT" in last_row, f"Expected BTM-RIGHT on last row, got: {last_row}"
    assert last_row.rstrip().endswith("BTM-RIGHT"), f"Expected BTM-RIGHT at end, got: {last_row}"
    await tui.stop()


@pytest.mark.tonio
async def test_positions_overlay_at_top_center():
    terminal = VirtualTerminal(80, 24)
    tui = TuiMainScreen(terminal)
    overlay = StaticOverlay(["CENTERED"])

    tui.add_child(EmptyContent())
    tui.show_overlay(overlay, {"anchor": "top-center", "width": 10})
    await tui.start()
    await render_and_flush(tui, terminal)

    viewport = terminal.get_viewport()
    first_row = viewport[0]
    assert "CENTERED" in first_row, f"Expected CENTERED on first row, got: {first_row}"
    # Check it's roughly centered (col 35 for width 10 in 80 col terminal)
    col_index = first_row.find("CENTERED")
    assert 30 <= col_index <= 40, f"Expected centered, got col {col_index}"
    await tui.stop()


# margin


@pytest.mark.tonio
async def test_clamps_negative_margins_to_zero():
    terminal = VirtualTerminal(80, 24)
    tui = TuiMainScreen(terminal)
    overlay = StaticOverlay(["NEG-MARGIN"])

    tui.add_child(EmptyContent())
    # Negative margins should be treated as 0
    tui.show_overlay(
        overlay,
        {"anchor": "top-left", "width": 12, "margin": {"top": -5, "left": -10, "right": 0, "bottom": 0}},
    )
    await tui.start()
    await render_and_flush(tui, terminal)

    viewport = terminal.get_viewport()
    # Should be at row 0, col 0 (negative margins clamped to 0)
    assert viewport[0].startswith("NEG-MARGIN"), f"Expected NEG-MARGIN at start of row 0, got: {viewport[0]}"
    await tui.stop()


@pytest.mark.tonio
async def test_respects_margin_as_number():
    terminal = VirtualTerminal(80, 24)
    tui = TuiMainScreen(terminal)
    overlay = StaticOverlay(["MARGIN"])

    tui.add_child(EmptyContent())
    tui.show_overlay(overlay, {"anchor": "top-left", "width": 10, "margin": 5})
    await tui.start()
    await render_and_flush(tui, terminal)

    viewport = terminal.get_viewport()
    # Should be on row 5 (not 0) due to margin
    assert "MARGIN" not in viewport[0], "Should not be on row 0"
    assert "MARGIN" not in viewport[4], "Should not be on row 4"
    assert "MARGIN" in viewport[5], f"Expected MARGIN on row 5, got: {viewport[5]}"
    # Should start at col 5 (not 0)
    assert viewport[5].find("MARGIN") == 5, f"Expected col 5, got {viewport[5].find('MARGIN')}"
    await tui.stop()


@pytest.mark.tonio
async def test_respects_margin_object():
    terminal = VirtualTerminal(80, 24)
    tui = TuiMainScreen(terminal)
    overlay = StaticOverlay(["MARGIN"])

    tui.add_child(EmptyContent())
    tui.show_overlay(
        overlay, {"anchor": "top-left", "width": 10, "margin": {"top": 2, "left": 3, "right": 0, "bottom": 0}}
    )
    await tui.start()
    await render_and_flush(tui, terminal)

    viewport = terminal.get_viewport()
    assert "MARGIN" in viewport[2], f"Expected MARGIN on row 2, got: {viewport[2]}"
    assert viewport[2].find("MARGIN") == 3, f"Expected col 3, got {viewport[2].find('MARGIN')}"
    await tui.stop()


# offset


@pytest.mark.tonio
async def test_applies_offset_x_and_offset_y_from_anchor_position():
    terminal = VirtualTerminal(80, 24)
    tui = TuiMainScreen(terminal)
    overlay = StaticOverlay(["OFFSET"])

    tui.add_child(EmptyContent())
    tui.show_overlay(overlay, {"anchor": "top-left", "width": 10, "offsetX": 10, "offsetY": 5})
    await tui.start()
    await render_and_flush(tui, terminal)

    viewport = terminal.get_viewport()
    assert "OFFSET" in viewport[5], f"Expected OFFSET on row 5, got: {viewport[5]}"
    assert viewport[5].find("OFFSET") == 10, f"Expected col 10, got {viewport[5].find('OFFSET')}"
    await tui.stop()


# percentage positioning


@pytest.mark.tonio
async def test_positions_with_row_percent_and_col_percent():
    terminal = VirtualTerminal(80, 24)
    tui = TuiMainScreen(terminal)
    overlay = StaticOverlay(["PCT"])

    tui.add_child(EmptyContent())
    # 50% should center both ways
    tui.show_overlay(overlay, {"width": 10, "row": "50%", "col": "50%"})
    await tui.start()
    await render_and_flush(tui, terminal)

    viewport = terminal.get_viewport()
    found_row = next((i for i, line in enumerate(viewport) if "PCT" in line), -1)
    # Should be roughly centered vertically (row ~11-12 for 24 row terminal)
    assert 10 <= found_row <= 13, f"Expected centered row, got {found_row}"
    await tui.stop()


@pytest.mark.tonio
async def test_row_percent_0_positions_at_top():
    terminal = VirtualTerminal(80, 24)
    tui = TuiMainScreen(terminal)
    overlay = StaticOverlay(["TOP"])

    tui.add_child(EmptyContent())
    tui.show_overlay(overlay, {"width": 10, "row": "0%"})
    await tui.start()
    await render_and_flush(tui, terminal)

    viewport = terminal.get_viewport()
    assert "TOP" in viewport[0], f"Expected TOP on row 0, got: {viewport[0]}"
    await tui.stop()


@pytest.mark.tonio
async def test_row_percent_100_positions_at_bottom():
    terminal = VirtualTerminal(80, 24)
    tui = TuiMainScreen(terminal)
    overlay = StaticOverlay(["BOTTOM"])

    tui.add_child(EmptyContent())
    tui.show_overlay(overlay, {"width": 10, "row": "100%"})
    await tui.start()
    await render_and_flush(tui, terminal)

    viewport = terminal.get_viewport()
    assert "BOTTOM" in viewport[23], f"Expected BOTTOM on last row, got: {viewport[23]}"
    await tui.stop()


# maxHeight


@pytest.mark.tonio
async def test_truncates_overlay_to_max_height():
    terminal = VirtualTerminal(80, 24)
    tui = TuiMainScreen(terminal)
    overlay = StaticOverlay(["Line 1", "Line 2", "Line 3", "Line 4", "Line 5"])

    tui.add_child(EmptyContent())
    tui.show_overlay(overlay, {"maxHeight": 3})
    await tui.start()
    await render_and_flush(tui, terminal)

    content = "\n".join(terminal.get_viewport())
    assert "Line 1" in content, "Should include Line 1"
    assert "Line 2" in content, "Should include Line 2"
    assert "Line 3" in content, "Should include Line 3"
    assert "Line 4" not in content, "Should NOT include Line 4"
    assert "Line 5" not in content, "Should NOT include Line 5"
    await tui.stop()


@pytest.mark.tonio
async def test_truncates_overlay_to_max_height_percent():
    terminal = VirtualTerminal(80, 10)
    tui = TuiMainScreen(terminal)
    # 10 lines in a 10 row terminal with 50% maxHeight should show 5 lines
    overlay = StaticOverlay(["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10"])

    tui.add_child(EmptyContent())
    tui.show_overlay(overlay, {"maxHeight": "50%"})
    await tui.start()
    await render_and_flush(tui, terminal)

    content = "\n".join(terminal.get_viewport())
    assert "L1" in content, "Should include L1"
    assert "L5" in content, "Should include L5"
    assert "L6" not in content, "Should NOT include L6"
    await tui.stop()


# absolute positioning


@pytest.mark.tonio
async def test_row_and_col_override_anchor():
    terminal = VirtualTerminal(80, 24)
    tui = TuiMainScreen(terminal)
    overlay = StaticOverlay(["ABSOLUTE"])

    tui.add_child(EmptyContent())
    # Even with bottom-right anchor, row/col should win
    tui.show_overlay(overlay, {"anchor": "bottom-right", "row": 3, "col": 5, "width": 10})
    await tui.start()
    await render_and_flush(tui, terminal)

    viewport = terminal.get_viewport()
    assert "ABSOLUTE" in viewport[3], f"Expected ABSOLUTE on row 3, got: {viewport[3]}"
    assert viewport[3].find("ABSOLUTE") == 5, f"Expected col 5, got {viewport[3].find('ABSOLUTE')}"
    await tui.stop()


# stacked overlays


@pytest.mark.tonio
async def test_renders_multiple_overlays_with_later_ones_on_top():
    terminal = VirtualTerminal(80, 24)
    tui = TuiMainScreen(terminal)

    tui.add_child(EmptyContent())

    overlay1 = StaticOverlay(["FIRST-OVERLAY"])
    tui.show_overlay(overlay1, {"anchor": "top-left", "width": 20})

    overlay2 = StaticOverlay(["SECOND"])
    tui.show_overlay(overlay2, {"anchor": "top-left", "width": 10})

    await tui.start()
    await render_and_flush(tui, terminal)

    viewport = terminal.get_viewport()
    # Second overlay should be visible (on top)
    assert "SECOND" in viewport[0], f"Expected SECOND on row 0, got: {viewport[0]}"
    await tui.stop()


@pytest.mark.tonio
async def test_handles_overlays_at_different_positions_without_interference():
    terminal = VirtualTerminal(80, 24)
    tui = TuiMainScreen(terminal)

    tui.add_child(EmptyContent())

    overlay1 = StaticOverlay(["TOP-LEFT"])
    tui.show_overlay(overlay1, {"anchor": "top-left", "width": 15})

    overlay2 = StaticOverlay(["BTM-RIGHT"])
    tui.show_overlay(overlay2, {"anchor": "bottom-right", "width": 15})

    await tui.start()
    await render_and_flush(tui, terminal)

    viewport = terminal.get_viewport()
    assert "TOP-LEFT" in viewport[0], f"Expected TOP-LEFT on row 0, got: {viewport[0]}"
    assert "BTM-RIGHT" in viewport[23], f"Expected BTM-RIGHT on row 23, got: {viewport[23]}"
    await tui.stop()


@pytest.mark.tonio
async def test_properly_hides_overlays_in_stack_order():
    terminal = VirtualTerminal(80, 24)
    tui = TuiMainScreen(terminal)

    tui.add_child(EmptyContent())

    overlay1 = StaticOverlay(["FIRST"])
    tui.show_overlay(overlay1, {"anchor": "top-left", "width": 10})

    overlay2 = StaticOverlay(["SECOND"])
    tui.show_overlay(overlay2, {"anchor": "top-left", "width": 10})

    await tui.start()
    await render_and_flush(tui, terminal)

    viewport = terminal.get_viewport()
    assert "SECOND" in viewport[0], "SECOND should be visible initially"

    # Hide top overlay
    tui.hide_overlay()
    await render_and_flush(tui, terminal)

    viewport = terminal.get_viewport()
    assert "FIRST" in viewport[0], "FIRST should be visible after hiding SECOND"

    await tui.stop()


# TUI overlay with short content (overlay-short-content.test.ts)


@pytest.mark.tonio
async def test_renders_overlay_when_content_is_shorter_than_terminal_height():
    # Terminal has 24 rows, but content only has 3 lines
    terminal = VirtualTerminal(80, 24)
    tui = TuiMainScreen(terminal)

    tui.add_child(StaticLines(["Line 1", "Line 2", "Line 3"]))

    # Show overlay centered - should be around row 10 in a 24-row terminal
    tui.show_overlay(StaticOverlay(["OVERLAY_TOP", "OVERLAY_MID", "OVERLAY_BOT"]))

    await tui.start()
    await terminal.wait_for_render()

    viewport = terminal.get_viewport()
    assert any("OVERLAY" in line for line in viewport), (
        "Overlay should be visible when content is shorter than terminal"
    )

    await tui.stop()


# TUI overlay compositing (tui-overlay-style-leak.test.ts)


@pytest.mark.tonio
async def test_does_not_leak_styles_when_a_trailing_reset_sits_beyond_the_last_visible_column_no_overlay():
    width = 20
    base_line = "\x1b[3m" + "X" * width + "\x1b[23m"

    terminal = VirtualTerminal(width, 6)
    tui = TuiMainScreen(terminal)
    tui.add_child(StaticLines([base_line, "INPUT"]))
    await tui.start()
    await render_and_flush(tui, terminal)
    assert terminal.get_cell_italic(1, 0) == 0
    await tui.stop()


@pytest.mark.tonio
async def test_does_not_leak_styles_when_overlay_slicing_drops_trailing_sgr_resets():
    width = 20
    base_line = "\x1b[3m" + "X" * width + "\x1b[23m"

    terminal = VirtualTerminal(width, 6)
    tui = TuiMainScreen(terminal)
    tui.add_child(StaticLines([base_line, "INPUT"]))

    tui.show_overlay(StaticOverlay(["OVR"]), {"row": 0, "col": 5, "width": 3})
    await tui.start()
    await render_and_flush(tui, terminal)

    assert terminal.get_cell_italic(1, 0) == 0
    await tui.stop()


# overlay CJK boundary regression (regression-overlay-cjk-boundary.test.ts)


def test_excludes_a_wide_grapheme_from_before_when_overlay_starts_inside_it():
    segments = extract_segments("abcd让EFGH", 5, 9, 11, True)

    assert segments["before"] == "abcd"
    assert segments["beforeWidth"] == 4
    assert visible_width(segments["before"]) == segments["beforeWidth"]
    assert segments["after"] == "H"
    assert segments["afterWidth"] == 1


def test_keeps_ascii_before_segment_behavior_at_the_same_boundary():
    segments = extract_segments("abcdG EFGH", 5, 9, 11, True)

    assert segments["before"] == "abcdG"
    assert segments["beforeWidth"] == 5
    assert visible_width(segments["before"]) == segments["beforeWidth"]


def test_composites_an_overlay_at_the_requested_column_when_it_starts_inside_a_wide_grapheme():
    out = composite_tui_line("abcd让EFGH", "│XX│", 5, 4, 20)
    prefix = slice_by_column(out, 0, 5, True)
    overlay = slice_by_column(out, 5, 4, True)

    assert ("让" in out) is False
    assert visible_width(out) == 20
    assert visible_width(prefix) == 5
    assert visible_width(overlay) == 4
    assert ("│XX│" in overlay) is True


def test_composites_an_overlay_when_it_starts_at_a_wide_grapheme_boundary():
    out = composite_tui_line("abcd让EFGH", "│XX│", 4, 4, 20)
    overlay = slice_by_column(out, 4, 4, True)

    assert ("让" in out) is False
    assert visible_width(out) == 20
    assert visible_width(overlay) == 4
    assert ("│XX│" in overlay) is True


# tab width overlay case (tab-width.test.ts)


class FullViewportContent:
    def render(self, width):
        return [line.ljust(width) for line in ["base 0", "base 1", "base 2"]]

    def invalidate(self):
        pass


class TabStatusOverlay:
    def render(self, width):
        return ["\tX"]

    def invalidate(self):
        pass


@pytest.mark.tonio
async def test_keeps_tab_containing_overlays_on_one_physical_terminal_row():
    terminal = LoggingVirtualTerminal(16, 3)
    tui = TuiMainScreen(terminal)
    tui.add_child(FullViewportContent())
    tui.show_overlay(TabStatusOverlay(), {"width": 4, "row": 1, "col": 4})
    await tui.start()

    await terminal.wait_for_render()
    # pi asserts padEnd-preserved trailing spaces; pyte cannot distinguish
    # written blanks, so the viewport is right-stripped here.
    assert terminal.get_viewport() == ["base 0", "base   X", "base 2"]
    assert "\t" not in terminal.get_writes()

    await tui.stop()
