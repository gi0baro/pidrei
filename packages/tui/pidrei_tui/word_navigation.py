"""Word navigation for cursor movement (port of pi tui ``word-navigation.ts``).

``find_word_backward``/``find_word_forward`` are pure functions; the optional
``segment``/``is_atomic_segment`` hooks mirror pi's ``WordNavigationOptions``
(custom segmenters return iterables of ``{"segment", "index", "isWordLike"}``
records, e.g. for paste markers treated as single units).
"""

from .utils import PUNCTUATION_REGEX, get_word_segmenter, is_whitespace_char


__all__ = ["find_word_backward", "find_word_forward"]


def find_word_backward(text: str, cursor: int, *, segment=None, is_atomic_segment=None) -> int:
    """Find the cursor position after moving one word backward from *cursor*.

    Skips trailing whitespace, then stops at the next word/punctuation boundary.
    """
    if cursor <= 0:
        return 0

    text_before_cursor = text[:cursor]
    segments = (
        list(segment(text_before_cursor)) if segment is not None else get_word_segmenter().segment(text_before_cursor)
    )
    new_cursor = cursor

    def is_atomic(value: str) -> bool:
        return is_atomic_segment is not None and is_atomic_segment(value)

    # Skip trailing whitespace
    while segments and not is_atomic(segments[-1]["segment"]) and is_whitespace_char(segments[-1]["segment"]):
        new_cursor -= len(segments.pop()["segment"])

    if not segments:
        return new_cursor

    last = segments[-1]

    if is_atomic(last["segment"]):
        # Skip one atomic segment.
        new_cursor -= len(last["segment"])
    elif last["isWordLike"]:
        # Skip inside one word-like segment, preserving ASCII punctuation boundaries.
        value = last["segment"]
        matches = list(PUNCTUATION_REGEX.finditer(value))
        if not matches:
            new_cursor -= len(value)
        else:
            last_match = matches[-1]
            new_cursor -= len(value) - (last_match.start() + len(last_match.group(0)))
    else:
        # Skip non-word non-whitespace run (punctuation)
        while (
            segments
            and not is_atomic(segments[-1]["segment"])
            and not segments[-1]["isWordLike"]
            and not is_whitespace_char(segments[-1]["segment"])
        ):
            new_cursor -= len(segments.pop()["segment"])

    return new_cursor


def find_word_forward(text: str, cursor: int, *, segment=None, is_atomic_segment=None) -> int:
    """Find the cursor position after moving one word forward from *cursor*.

    Skips leading whitespace, then stops at the next word/punctuation boundary.
    """
    if cursor >= len(text):
        return len(text)

    text_after_cursor = text[cursor:]
    segments = segment(text_after_cursor) if segment is not None else get_word_segmenter().segment(text_after_cursor)
    iterator = iter(segments)
    current = next(iterator, None)
    new_cursor = cursor

    def is_atomic(value: str) -> bool:
        return is_atomic_segment is not None and is_atomic_segment(value)

    # Skip leading whitespace
    while current is not None and not is_atomic(current["segment"]) and is_whitespace_char(current["segment"]):
        new_cursor += len(current["segment"])
        current = next(iterator, None)

    if current is None:
        return new_cursor

    if is_atomic(current["segment"]):
        # Skip one atomic segment.
        new_cursor += len(current["segment"])
    elif current["isWordLike"]:
        # Skip inside one word-like segment, preserving ASCII punctuation boundaries.
        match = PUNCTUATION_REGEX.search(current["segment"])
        new_cursor += match.start() if match is not None else len(current["segment"])
    else:
        # Skip non-word non-whitespace run (punctuation)
        while (
            current is not None
            and not is_atomic(current["segment"])
            and not current["isWordLike"]
            and not is_whitespace_char(current["segment"])
        ):
            new_cursor += len(current["segment"])
            current = next(iterator, None)

    return new_cursor
