"""Mirror of pi tui test/terminal-colors.test.ts (parse functions).

The TUI.query_terminal_background_color cases land with the TUI renderer
stage in test_tui_queries.py.
"""

from pidrei_tui.terminal_colors import parse_osc11_background_color, parse_terminal_color_scheme_report


def test_parses_16_bit_osc11_rgb_responses():
    assert parse_osc11_background_color("\x1b]11;rgb:0000/8000/ffff\x07") == {"r": 0, "g": 128, "b": 255}


def test_parses_osc11_hex_responses():
    assert parse_osc11_background_color("\x1b]11;#ffffff\x1b\\") == {"r": 255, "g": 255, "b": 255}
    assert parse_osc11_background_color("\x1b]11;#000000\x07") == {"r": 0, "g": 0, "b": 0}


def test_rejects_non_strict_osc11_responses():
    assert parse_osc11_background_color("x\x1b]11;#ffffff\x07") is None
    assert parse_osc11_background_color("\x1b]10;#ffffff\x07") is None
    assert parse_osc11_background_color("\x1b]11;#ffffff\x07x") is None


def test_parses_color_scheme_reports():
    assert parse_terminal_color_scheme_report("\x1b[?997;1n") == "dark"
    assert parse_terminal_color_scheme_report("\x1b[?997;2n") == "light"
    assert parse_terminal_color_scheme_report("\x1b[?997;2n\x1b[?997;1n\x1b[?997;1n") == "dark"
    assert parse_terminal_color_scheme_report("\x1b[?997;1n\x1b[?997;2n\x1b[?997;2n") == "light"
    assert parse_terminal_color_scheme_report("\x1b[?997;3n") is None
    assert parse_terminal_color_scheme_report("\x1b[?996n") is None
    assert parse_terminal_color_scheme_report("x\x1b[?997;1n") is None
