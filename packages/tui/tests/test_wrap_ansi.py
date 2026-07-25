"""Mirror of pi tui test/wrap-ansi.test.ts."""

import re

from pidrei_tui.utils import visible_width, wrap_text_with_ansi


# underline styling


def test_does_not_apply_underline_style_before_the_styled_text():
    underline_on = "\x1b[4m"
    underline_off = "\x1b[24m"
    url = "https://example.com/very/long/path/that/will/wrap"
    text = f"read this thread {underline_on}{url}{underline_off}"

    wrapped = wrap_text_with_ansi(text, 40)

    # First line should NOT contain underline code - it's just "read this thread"
    assert wrapped[0] == "read this thread"

    # Second line should start with underline, have URL content
    assert wrapped[1].startswith(underline_on) is True
    assert "https://" in wrapped[1]


def test_does_not_have_whitespace_before_underline_reset_code():
    underline_on = "\x1b[4m"
    underline_off = "\x1b[24m"
    text_with_underlined_trailing_space = f"{underline_on}underlined text here {underline_off}more"

    wrapped = wrap_text_with_ansi(text_with_underlined_trailing_space, 18)

    assert f" {underline_off}" not in wrapped[0]


def test_does_not_bleed_underline_to_padding():
    underline_on = "\x1b[4m"
    underline_off = "\x1b[24m"
    url = "https://example.com/very/long/path/that/will/definitely/wrap"
    text = f"prefix {underline_on}{url}{underline_off} suffix"

    wrapped = wrap_text_with_ansi(text, 30)

    # Middle lines (with underlined content) should end with underline-off, not full reset
    for line in wrapped[1:-1]:
        if underline_on in line:
            # Should end with underline off, NOT full reset
            assert line.endswith(underline_off) is True
            assert line.endswith("\x1b[0m") is False


# background color preservation


def test_preserves_background_color_across_wrapped_lines_without_full_reset():
    bg_blue = "\x1b[44m"
    reset = "\x1b[0m"
    text = f"{bg_blue}hello world this is blue background text{reset}"

    wrapped = wrap_text_with_ansi(text, 15)

    # Each line should have background color
    for line in wrapped:
        assert bg_blue in line

    # Middle lines should NOT end with full reset (kills background for padding)
    for line in wrapped[:-1]:
        assert line.endswith("\x1b[0m") is False


def test_resets_underline_but_preserves_background_when_wrapping_underlined_text_inside_background():
    underline_on = "\x1b[4m"
    underline_off = "\x1b[24m"
    reset = "\x1b[0m"

    text = f"\x1b[41mprefix {underline_on}UNDERLINED_CONTENT_THAT_WRAPS{underline_off} suffix{reset}"

    wrapped = wrap_text_with_ansi(text, 20)

    # All lines should have background color 41 (either as \x1b[41m or combined like \x1b[4;41m)
    for line in wrapped:
        has_bg_color = "[41m" in line or ";41m" in line or "[41;" in line
        assert has_bg_color

    # Lines with underlined content should use underline-off at end, not full reset
    for line in wrapped[:-1]:
        # If this line has underline on, it should end with underline off (not full reset)
        if ("[4m" in line or "[4;" in line or ";4m" in line) and underline_off not in line:
            assert line.endswith(underline_off) is True
            assert line.endswith("\x1b[0m") is False


# basic wrapping


def test_handles_lf_crlf_and_cr_line_endings():
    assert wrap_text_with_ansi("first\nsecond\r\nthird\rfourth", 80) == [
        "first",
        "second",
        "third",
        "fourth",
    ]


def test_preserves_ansi_state_across_crlf_and_cr_line_endings():
    red = "\x1b[31m"
    reset = "\x1b[0m"

    assert wrap_text_with_ansi(f"{red}first\r\nsecond\rthird{reset}", 80) == [
        f"{red}first",
        f"{red}second",
        f"{red}third{reset}",
    ]


def test_wraps_plain_text_correctly():
    text = "hello world this is a test"
    wrapped = wrap_text_with_ansi(text, 10)

    assert len(wrapped) > 1
    for line in wrapped:
        assert visible_width(line) <= 10


def test_breaks_cjk_runs_at_grapheme_boundaries_after_latin_text():
    text = "This is an example 中文汉字测试段落内容中文汉字测试段落内容."
    wrapped = wrap_text_with_ansi(text, 40)

    assert wrapped == ["This is an example 中文汉字测试段落内容", "中文汉字测试段落内容."]
    for line in wrapped:
        assert visible_width(line) <= 40


def test_preserves_color_codes_when_wrapping_cjk_runs():
    red = "\x1b[31m"
    reset = "\x1b[0m"
    text = f"{red}This is an example 中文汉字测试段落内容中文汉字测试段落内容.{reset}"
    wrapped = wrap_text_with_ansi(text, 40)

    assert len(wrapped) == 2
    assert wrapped[0] == f"{red}This is an example 中文汉字测试段落内容"
    assert wrapped[1] == f"{red}中文汉字测试段落内容.{reset}"
    for line in wrapped:
        assert visible_width(line) <= 40


def test_ignores_osc_133_semantic_markers_in_visible_width():
    text = "\x1b]133;A\x07hello\x1b]133;B\x07"
    assert visible_width(text) == 5


def test_ignores_osc_sequences_terminated_with_st_in_visible_width():
    text = "\x1b]133;A\x1b\\hello\x1b]133;B\x1b\\"
    assert visible_width(text) == 5


def test_treats_isolated_regional_indicators_as_width_2():
    assert visible_width("🇨") == 2
    assert visible_width("🇨🇳") == 2


def test_truncates_trailing_whitespace_that_exceeds_width():
    two_spaces_wrapped_to_width_1 = wrap_text_with_ansi("  ", 1)
    assert visible_width(two_spaces_wrapped_to_width_1[0]) <= 1


def test_preserves_color_codes_across_wraps():
    red = "\x1b[31m"
    reset = "\x1b[0m"
    text = f"{red}hello world this is red{reset}"

    wrapped = wrap_text_with_ansi(text, 10)

    # Each continuation line should start with red code
    for line in wrapped[1:]:
        assert line.startswith(red) is True

    # Middle lines should not end with full reset
    for line in wrapped[:-1]:
        assert line.endswith("\x1b[0m") is False


# OSC 8 hyperlinks


def test_re_emits_osc8_open_at_the_start_of_continuation_lines():
    # A hyperlink whose text is long enough to wrap
    url = "https://example.com"
    # OSC 8 open + text that is 10 visible chars + OSC 8 close
    input_text = f"\x1b]8;;{url}\x1b\\0123456789\x1b]8;;\x1b\\"
    lines = wrap_text_with_ansi(input_text, 6)

    # Every line that contains visible text from inside the hyperlink
    # should start with the OSC 8 open sequence (or be preceded by it).
    for line in lines:
        # If the line has visible content it must begin with the OSC 8 re-open
        # OR it is the line where the close appeared with no following content.
        stripped = re.sub(r"\x1b\]8;;[^\x1b\x07]*\x1b\\", "", line)
        stripped = re.sub(r"\x1b\[[0-9;]*m", "", stripped)
        if stripped.strip():
            assert line.startswith(f"\x1b]8;;{url}\x1b\\") or f"\x1b]8;;{url}\x1b\\" in line, (
                f"Line {line!r} has visible text but no OSC 8 re-open"
            )


def test_closes_osc8_before_each_line_break():
    url = "https://example.com"
    input_text = f"\x1b]8;;{url}\x1b\\0123456789\x1b]8;;\x1b\\"
    lines = wrap_text_with_ansi(input_text, 6)

    for line in lines[:-1]:
        # Every non-final line that is inside a hyperlink should end with the close
        if f"\x1b]8;;{url}\x1b\\" in line:
            assert line.endswith("\x1b]8;;\x1b\\"), (
                f"Non-final line {line!r} is inside a hyperlink but does not close it"
            )


def test_preserves_bel_terminators_when_wrapping_oauth_style_hyperlinks():
    url = "https://example.com/oauth/" + "a" * 32
    input_text = f"\x1b]8;;{url}\x07{url}\x1b]8;;\x07"
    lines = wrap_text_with_ansi(input_text, 20)

    assert len(lines) > 1
    for line in lines:
        assert f"\x1b]8;;{url}\x07" in line, f"Line {line!r} does not reopen the hyperlink with BEL"
        assert f"\x1b]8;;{url}\x1b\\" not in line, f"Line {line!r} reopens the hyperlink with ST"
    for line in lines[:-1]:
        assert line.endswith("\x1b]8;;\x07"), f"Line {line!r} does not close the hyperlink with BEL"


def test_does_not_emit_osc8_sequences_on_lines_that_are_outside_the_hyperlink():
    url = "https://example.com"
    input_text = f"before \x1b]8;;{url}\x1b\\link\x1b]8;;\x1b\\ after"
    lines = wrap_text_with_ansi(input_text, 80)

    # With width 80 everything fits on one line; there should be exactly one
    # OSC 8 open and one OSC 8 close.
    assert len(lines) == 1
    open_count = len(re.findall(r"\x1b\]8;;https:[^\x1b]+\x1b\\", lines[0]))
    close_count = len(re.findall(r"\x1b\]8;;\x1b\\", lines[0]))
    assert open_count == 1
    assert close_count == 1
