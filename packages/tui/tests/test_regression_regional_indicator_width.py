"""Mirror of pi tui test/regression-regional-indicator-width.test.ts."""

from pidrei_tui.utils import visible_width, wrap_text_with_ansi


def test_treats_partial_flag_grapheme_as_full_width_to_avoid_streaming_render_drift():
    # Repro context:
    # During streaming, "🇨🇳" often appears as an intermediate "🇨" first.
    # If "🇨" is measured as width 1 while terminal renders it as width 2,
    # differential rendering can drift and leave stale characters on screen.
    partial_flag = "🇨"
    list_line = "      - 🇨"

    assert visible_width(partial_flag) == 2
    assert visible_width(list_line) == 10


def test_wraps_intermediate_partial_flag_list_line_before_overflow():
    # Width 9 cannot fit "      - 🇨" if 🇨 is width 2 (8 + 2 = 10).
    # This must wrap to avoid terminal auto-wrap mismatch.
    wrapped = wrap_text_with_ansi("      - 🇨", 9)

    assert len(wrapped) == 2
    assert visible_width(wrapped[0] or "") == 7
    assert visible_width(wrapped[1] or "") == 2


def test_treats_all_regional_indicator_singleton_graphemes_as_width_2():
    for cp in range(0x1F1E6, 0x1F1FF + 1):
        regional_indicator = chr(cp)
        assert visible_width(regional_indicator) == 2, f"Expected {regional_indicator} (U+{cp:X}) to be width 2"


def test_keeps_full_flag_pairs_at_width_2():
    for flag in ["🇯🇵", "🇺🇸", "🇬🇧", "🇨🇳", "🇩🇪", "🇫🇷"]:
        assert visible_width(flag) == 2, f"Expected {flag} to be width 2"


def test_keeps_common_streaming_emoji_intermediates_at_stable_width():
    for sample in ["👍", "👍🏻", "✅", "⚡", "⚡️", "👨", "👨‍💻", "🏳️‍🌈"]:
        assert visible_width(sample) == 2, f"Expected {sample} to be width 2"
