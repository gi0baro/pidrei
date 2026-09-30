"""Mirror of pi tui test/visible-width.test.ts."""

from pidrei_tui.utils import visible_width


def test_measures_styled_ascii_without_counting_escape_sequences():
    assert visible_width("\x1b[38;5;4mhello\x1b[39m world") == 11
    assert visible_width("\x1b]8;;https://example.com\x07link\x1b]8;;\x07") == 4
    assert visible_width("\x1b]133;A\x1b\\prompt") == 6
    assert visible_width("\x1b_pi:c\x07cursor") == 6


def test_counts_tabs_as_three_columns_in_styled_text():
    assert visible_width("\x1b[1ma\tb\x1b[22m") == 5


def test_measures_styled_non_ascii_text():
    assert visible_width("\x1b[31m日本\x1b[39m ok") == 7
    assert visible_width("\x1b[31m─→\x1b[39m") == 2


def test_treats_unterminated_escape_sequences_as_zero_width_control_characters():
    assert visible_width("\x1b[31") == 3
    assert visible_width("a\x1b") == 1
