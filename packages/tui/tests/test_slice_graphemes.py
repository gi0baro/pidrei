"""Mirror of pi tui test/slice-graphemes.test.ts."""

from pidrei_tui.utils import get_grapheme_cell_range, slice_with_width


# #9549: slicing splits standalone characters without the grapheme segmenter; clusters must stay intact.


def test_slices_box_drawing_arrows_and_cjk_by_cell():
    line = "│ a → 中文 ✓ │"
    assert slice_with_width(line, 0, 6, True) == ("│ a → ", 6)
    assert slice_with_width(line, 6, 4, True) == ("中文", 4)
    assert slice_with_width(line, 7, 3, True) == ("文", 2)
    assert get_grapheme_cell_range(line, 7) == (6, 8)


def test_keeps_clusters_that_join_standalone_characters():
    # A ZWJ joins the heart to the preceding emoji; the combining accent joins its base letter.
    family = "x👨‍❤y"
    assert slice_with_width(family, 1, 2, True)[0] == "👨‍❤"
    accented = "→é→"
    assert slice_with_width(accented, 1, 1, True) == ("é", 1)
    assert get_grapheme_cell_range(accented, 2) == (2, 3)
