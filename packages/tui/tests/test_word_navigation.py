"""Mirror of pi tui test/word-navigation.test.ts."""

from pidrei_tui.word_navigation import find_word_backward, find_word_forward


# findWordBackward


def test_backward_basic_words_hello_world():
    text = "hello world"
    assert find_word_backward(text, 11) == 6
    assert find_word_backward(text, 6) == 0


def test_backward_dotted_foo_bar():
    text = "foo.bar"
    assert find_word_backward(text, 7) == 4
    assert find_word_backward(text, 4) == 3
    assert find_word_backward(text, 3) == 0


def test_backward_colon_foo_bar():
    text = "foo:bar"
    assert find_word_backward(text, 7) == 4
    assert find_word_backward(text, 4) == 3
    assert find_word_backward(text, 3) == 0


def test_backward_path_to_file():
    text = "path/to/file"
    assert find_word_backward(text, 12) == 8
    assert find_word_backward(text, 8) == 7
    # "/to" is one word-like segment with "/" as punctuation boundary
    assert find_word_backward(text, 7) == 5
    assert find_word_backward(text, 5) == 4
    assert find_word_backward(text, 4) == 0


def test_backward_cjk_mixed():
    text = "你好世界 test"
    assert find_word_backward(text, len(text)) == 5
    # Intl.Segmenter treats CJK dictionary words as word-like segments
    assert find_word_backward(text, 5) == 2
    assert find_word_backward(text, 2) == 0


def test_backward_whitespace_at_boundaries():
    text = "  hello  "
    assert find_word_backward(text, 9) == 2
    assert find_word_backward(text, 2) == 0


def test_backward_punctuation_run_foo_bar():
    text = "foo...bar"
    assert find_word_backward(text, 9) == 6
    assert find_word_backward(text, 6) == 3
    assert find_word_backward(text, 3) == 0


def test_backward_cursor_at_0_returns_0():
    assert find_word_backward("hello", 0) == 0


# findWordForward


def test_forward_basic_words_hello_world():
    text = "hello world"
    assert find_word_forward(text, 0) == 5
    assert find_word_forward(text, 5) == 11


def test_forward_dotted_foo_bar():
    text = "foo.bar"
    assert find_word_forward(text, 0) == 3
    assert find_word_forward(text, 3) == 4
    assert find_word_forward(text, 4) == 7


def test_forward_colon_foo_bar():
    text = "foo:bar"
    assert find_word_forward(text, 0) == 3
    assert find_word_forward(text, 3) == 4
    assert find_word_forward(text, 4) == 7


def test_forward_path_to_file():
    text = "path/to/file"
    assert find_word_forward(text, 0) == 4
    assert find_word_forward(text, 4) == 5
    assert find_word_forward(text, 5) == 7
    assert find_word_forward(text, 7) == 8
    assert find_word_forward(text, 8) == 12


def test_forward_cjk_mixed():
    text = "你好世界 test"
    first_end = find_word_forward(text, 0)
    assert first_end > 0
    assert first_end <= 4
    # Walk to end
    pos = 0
    while pos < len(text):
        next_pos = find_word_forward(text, pos)
        if next_pos == pos:
            break
        pos = next_pos
    assert pos == len(text)


def test_forward_whitespace_at_boundaries():
    text = "  hello  "
    assert find_word_forward(text, 0) == 7
    assert find_word_forward(text, 7) == 9


def test_forward_punctuation_run_foo_bar():
    text = "foo...bar"
    assert find_word_forward(text, 0) == 3
    assert find_word_forward(text, 3) == 6
    assert find_word_forward(text, 6) == 9


def test_forward_cursor_at_end_returns_end():
    assert find_word_forward("hello", 5) == 5


# atomic segments

_MARKER = "[paste #1 +5 lines]"
_TEXT = f"hello {_MARKER} world"


def _is_atomic(segment: str) -> bool:
    return segment == _MARKER


# The functions slice text before calling segment(), so we map each expected
# substring to its pre-split segments.
_SEGMENT_MAP = {
    _TEXT: [
        {"segment": "hello", "index": 0, "isWordLike": True},
        {"segment": " ", "index": 5, "isWordLike": False},
        {"segment": _MARKER, "index": 6, "isWordLike": True},
        {"segment": " ", "index": 25, "isWordLike": False},
        {"segment": "world", "index": 26, "isWordLike": True},
    ],
    _TEXT[:26]: [
        {"segment": "hello", "index": 0, "isWordLike": True},
        {"segment": " ", "index": 5, "isWordLike": False},
        {"segment": _MARKER, "index": 6, "isWordLike": True},
        {"segment": " ", "index": 25, "isWordLike": False},
    ],
    _TEXT[6:]: [
        {"segment": _MARKER, "index": 0, "isWordLike": True},
        {"segment": " ", "index": 19, "isWordLike": False},
        {"segment": "world", "index": 20, "isWordLike": True},
    ],
}


def _segment(text: str) -> list[dict]:
    return _SEGMENT_MAP.get(text, [])


def test_atomic_backward_skips_word_then_stops_before_atomic_marker():
    assert find_word_backward(_TEXT, len(_TEXT), segment=_segment, is_atomic_segment=_is_atomic) == 26


def test_atomic_backward_skips_whitespace_then_atomic_marker_as_one_unit():
    assert find_word_backward(_TEXT, 26, segment=_segment, is_atomic_segment=_is_atomic) == 6


def test_atomic_forward_skips_atomic_marker_as_one_unit():
    assert find_word_forward(_TEXT, 6, segment=_segment, is_atomic_segment=_is_atomic) == 6 + len(_MARKER)
