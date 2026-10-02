"""Mirror of pi tui test/regression-slice-by-column-ansi-order.test.ts."""

from pidrei_tui.utils import slice_by_column


# https://github.com/earendil-works/pi/issues/10169
def test_keeps_a_reset_at_the_slice_start_after_earlier_style_codes():
    line = "\x1b[32mfoo\x1b[39m bar"
    assert slice_by_column(line, 3, 4, True) == "\x1b[32m\x1b[39m bar"


def test_does_not_leak_color_into_text_after_a_highlighted_token():
    line = "Another \x1b[35malpha\x1b[39m line with \x1b[35mbeta\x1b[39m later."
    after = slice_by_column(line, 13, 100, True)
    assert after == "\x1b[35m\x1b[39m line with \x1b[35mbeta\x1b[39m later."
