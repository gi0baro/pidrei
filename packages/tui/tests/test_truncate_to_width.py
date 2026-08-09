"""Mirror of pi tui test/truncate-to-width.test.ts."""

from pidrei_tui.utils import normalize_terminal_output, truncate_to_width, visible_width


# truncateToWidth


def test_keeps_output_within_width_for_very_large_unicode_input():
    text = "🙂界" * 100_000
    truncated = truncate_to_width(text, 40, "…")

    assert visible_width(truncated) <= 40
    assert truncated.endswith("…\x1b[0m") is True


def test_preserves_ansi_styling_for_kept_text_and_resets_before_and_after_ellipsis():
    text = "\x1b[31m" + "hello " * 1000 + "\x1b[0m"
    truncated = truncate_to_width(text, 20, "…")

    assert visible_width(truncated) <= 20
    assert ("\x1b[31m" in truncated) is True
    assert truncated.endswith("\x1b[0m…\x1b[0m") is True


def test_closes_a_bel_terminated_osc8_link_when_truncating_its_label():
    open_link = "\x1b]8;;https://example.com\x07"
    close_link = "\x1b]8;;\x07"
    text = f"{open_link}some-longer-label-here{close_link}"

    assert truncate_to_width(text, 15) == f"{open_link}some-longer-{close_link}\x1b[0m...\x1b[0m"


def test_handles_malformed_ansi_escape_prefixes_without_hanging():
    text = "abc\x1bnot-ansi " + "🙂" * 1000
    truncated = truncate_to_width(text, 20, "…")

    assert visible_width(truncated) <= 20


def test_clips_wide_ellipsis_safely_and_brackets_it_with_resets():
    assert truncate_to_width("abcdef", 1, "🙂") == ""
    assert truncate_to_width("abcdef", 2, "🙂") == "\x1b[0m🙂\x1b[0m"
    assert visible_width(truncate_to_width("abcdef", 2, "🙂")) <= 2


def test_returns_the_original_text_when_it_already_fits_even_if_ellipsis_is_too_wide():
    assert truncate_to_width("a", 2, "🙂") == "a"
    assert truncate_to_width("界", 2, "🙂") == "界"


def test_pads_truncated_output_to_requested_width():
    truncated = truncate_to_width("🙂界🙂界🙂界", 8, "…", True)
    assert visible_width(truncated) == 8


def test_adds_a_trailing_reset_when_truncating_without_an_ellipsis():
    truncated = truncate_to_width("\x1b[31m" + "hello" * 100, 10, "")
    assert visible_width(truncated) <= 10
    assert truncated.endswith("\x1b[0m") is True


def test_keeps_a_contiguous_prefix_instead_of_skipping_a_wide_grapheme_and_resuming_later():
    truncated = truncate_to_width("🙂\t界 \x1b_abc\x07", 7, "…", True)
    assert truncated == "🙂\t\x1b[0m…\x1b[0m "


# visibleWidth


def test_counts_tabs_inline_and_skips_ansi_inline():
    assert visible_width("\t\x1b[31m界\x1b[0m") == 5


def test_counts_indic_conjunct_spacing_code_points_within_grapheme_clusters():
    assert visible_width("र्क") == 2
    assert visible_width("नेटवर्क") == 5
    assert visible_width("सर्वाधिकार सुरक्षित। ऑर्डर पर क्लिक करें") == 33
    assert visible_width("র্ক") == 2
    assert visible_width("ર્ક") == 2
    assert visible_width("ର୍କ") == 2
    assert visible_width("ర్క") == 2
    assert visible_width("ര്‍ക") == 2


def test_keeps_ordinary_combining_marks_zero_width():
    assert visible_width("é") == 1
    assert visible_width("čřžůú") == 5
    assert visible_width("שָׁ") == 1
    assert visible_width("بّ") == 1
    assert visible_width("རྐ") == 1
    assert visible_width("ᜠ᜴") == 1
    assert visible_width("가〮") == 2
    assert visible_width("가〯") == 2


def test_keeps_cjk_and_japanese_width_accounting_unchanged():
    assert visible_width("网络") == 4
    assert visible_width("ネットワーク") == 12
    assert visible_width("が") == 2
    assert visible_width("が") == 2


def test_counts_myanmar_marks_that_terminals_allocate_cells_for():
    assert visible_width("ကာ") == 2
    assert visible_width("ကေ") == 2
    assert visible_width("က်") == 2
    assert visible_width("ကျ") == 2
    assert visible_width("ကြ") == 2
    assert visible_width("ကဳ") == 2
    assert visible_width("ကဴ") == 2
    assert visible_width("ကဵ") == 2
    assert visible_width("ကး") == 2
    assert visible_width("ကို") == 1
    assert visible_width("က္") == 1


def test_keeps_thai_and_lao_am_clusters_at_their_normal_cell_width():
    assert visible_width("ำ") == 1
    assert visible_width("ຳ") == 1
    assert visible_width("กำ") == 2
    assert visible_width("ກຳ") == 2


def test_normalizes_thai_and_lao_am_vowels_only_for_terminal_output():
    assert normalize_terminal_output("ำ") == "ํา"
    assert normalize_terminal_output("ຳ") == "ໍາ"
    assert visible_width(normalize_terminal_output("ำabc")) == visible_width("ำabc")
    assert visible_width(normalize_terminal_output("ຳabc")) == visible_width("ຳabc")
