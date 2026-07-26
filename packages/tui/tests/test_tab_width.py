"""Mirror of pi tui test/tab-width.test.ts.

The overlay compositing case ("keeps tab-containing overlays on one physical
terminal row") lands with the TUI renderer stage.
"""

from pidrei_tui.utils import extract_segments, normalize_terminal_output, slice_with_width, visible_width


def test_keeps_slice_helper_widths_consistent_with_visible_width():
    text = "out 192M\t.pidrei/skill-tests/results-ha"
    sliced, width = slice_with_width(text, 0, 10, True)

    assert sliced == "out 192M"
    assert width == 8
    assert visible_width(sliced) == width


def test_keeps_overlay_segment_widths_consistent_with_visible_width():
    text = "out 192M\t.pidrei/skill-tests/results-ha"
    segments = extract_segments(text, 10, 13, 10, True)

    assert segments["before"] == "out 192M"
    assert segments["beforeWidth"] == 8
    assert visible_width(segments["before"]) == segments["beforeWidth"]

    tab_fits = extract_segments(text, 11, 13, 10, True)
    assert tab_fits["before"] == "out 192M\t"
    assert tab_fits["beforeWidth"] == 11
    assert visible_width(tab_fits["before"]) == tab_fits["beforeWidth"]


def test_keeps_tabs_inside_terminal_control_sequences_byte_identical():
    control_sequences = [
        "\x1b]8;;https://example.test/a\tb\x07",
        "\x1b]0;window\ttitle\x1b\\",
        "\x1b_payload\tdata\x1b\\",
    ]

    for control_sequence in control_sequences:
        assert normalize_terminal_output(f"{control_sequence}label\ttext") == f"{control_sequence}label   text"
