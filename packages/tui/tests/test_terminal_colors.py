"""Mirror of pi tui test/terminal-colors.test.ts (parse functions).

The TUI.query_terminal_colors cases land with the TUI renderer stage in
test_tui_queries.py.
"""

from pidrei_tui.terminal_colors import parse_osc_color_response, parse_terminal_color_scheme_report


def test_parses_color_scheme_reports():
    assert parse_terminal_color_scheme_report("\x1b[?997;1n") == "dark"
    assert parse_terminal_color_scheme_report("\x1b[?997;2n") == "light"
    assert parse_terminal_color_scheme_report("\x1b[?997;2n\x1b[?997;1n\x1b[?997;1n") == "dark"
    assert parse_terminal_color_scheme_report("\x1b[?997;1n\x1b[?997;2n\x1b[?997;2n") == "light"
    assert parse_terminal_color_scheme_report("\x1b[?997;3n") is None
    assert parse_terminal_color_scheme_report("\x1b[?996n") is None
    assert parse_terminal_color_scheme_report("x\x1b[?997;1n") is None


def test_parses_osc_10_11_and_4_replies():
    assert parse_osc_color_response("\x1b]10;rgb:ffff/ffff/ffff\x07") == {
        "target": "foreground",
        "rgb": {"r": 255, "g": 255, "b": 255},
    }
    assert parse_osc_color_response("\x1b]4;13;#ff0080\x1b\\") == {"target": 13, "rgb": {"r": 255, "g": 0, "b": 128}}
    assert parse_osc_color_response("\x1b]4;1;bogus\x07") == {"target": 1, "rgb": None}
    assert parse_osc_color_response("\x1b]12;#ffffff\x07") is None
