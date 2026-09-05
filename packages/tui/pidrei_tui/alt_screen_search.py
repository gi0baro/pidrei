"""Transcript search over rendered alt-screen content (port of pi `alt-screen-search.ts`).

The corpus is the whitespace-collapsed searchable text plus one span per
indexed run mapping a text range back to `{row, startCol, endCol}` cells;
matches are re-grouped into per-row segments for highlighting. Printable
ASCII lines are indexed one non-space run at a time (columns are linear in
the text offset); other lines are segmented per grapheme. pi maps UTF-16 code
units, Python maps code points — both are self-consistent with their regex
engines' match offsets.
"""

import re
import sys
from dataclasses import dataclass, field

import grapheme as grapheme_lib

from .components.input import Input
from .keybindings import get_keybindings
from .utils import strip_terminal_sequences, truncate_to_width, visible_width


_WHITESPACE_RE = re.compile(r"^\s+$")
_PRINTABLE_ASCII_RE = re.compile(r"^[\x20-\x7e]*$")


@dataclass(slots=True)
class AltScreenSearchSegment:
    row: int
    start_col: int
    end_col: int


@dataclass(slots=True)
class AltScreenSearchMatch:
    segments: list[AltScreenSearchSegment] = field(default_factory=list)


@dataclass(slots=True)
class _SearchSourceSpan:
    text_start: int
    text_end: int
    row: int
    start_col: int
    end_col: int
    linear_columns: bool


@dataclass(slots=True)
class _SearchCorpus:
    text: str
    spans: list[_SearchSourceSpan]


def _build_search_corpus(lines: list[str]) -> _SearchCorpus:
    chunks: list[str] = []
    spans: list[_SearchSourceSpan] = []
    text_length = 0
    pending_separator = False

    def append_separator() -> None:
        nonlocal text_length, pending_separator
        if not pending_separator:
            return
        chunks.append(" ")
        text_length += 1
        pending_separator = False

    for row, raw_line in enumerate(lines):
        line = strip_terminal_sequences(raw_line or "")
        column = 0

        # Rendered transcripts are overwhelmingly ASCII. Index complete non-space
        # runs at once instead of segmenting and allocating one mapping per cell.
        if _PRINTABLE_ASCII_RE.match(line):
            index = 0
            length = len(line)
            while index < length:
                if line[index] == " ":
                    if text_length > 0:
                        pending_separator = True
                    column += 1
                    index += 1
                    continue
                end = index + 1
                while end < length and line[end] != " ":
                    end += 1
                append_separator()
                text = line[index:end]
                chunks.append(text)
                spans.append(
                    _SearchSourceSpan(
                        text_start=text_length,
                        text_end=text_length + len(text),
                        row=row,
                        start_col=column,
                        end_col=column + len(text),
                        linear_columns=True,
                    )
                )
                text_length += len(text)
                column += len(text)
                index = end
        else:
            for text in grapheme_lib.graphemes(line):
                width = visible_width(text)
                if _WHITESPACE_RE.match(text):
                    if text_length > 0:
                        pending_separator = True
                    column += width
                    continue
                append_separator()
                chunks.append(text)
                spans.append(
                    _SearchSourceSpan(
                        text_start=text_length,
                        text_end=text_length + len(text),
                        row=row,
                        start_col=column,
                        end_col=column + width,
                        linear_columns=False,
                    )
                )
                text_length += len(text)
                column += width
        if text_length > 0:
            pending_separator = True

    return _SearchCorpus(text="".join(chunks), spans=spans)


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip()


def _find_search_corpus_matches(corpus: _SearchCorpus, normalized_query: str) -> list[AltScreenSearchMatch]:
    if not normalized_query:
        return []
    expression = re.compile(re.escape(normalized_query), re.IGNORECASE)
    matches: list[AltScreenSearchMatch] = []
    spans = corpus.spans
    span_index = 0

    for match in expression.finditer(corpus.text):
        start = match.start()
        end = start + len(match.group(0))
        while span_index < len(spans) and spans[span_index].text_end <= start:
            span_index += 1

        segments: list[AltScreenSearchSegment] = []
        for index in range(span_index, len(spans)):
            span = spans[index]
            if span.text_start >= end:
                break
            if span.text_end <= start:
                continue
            if span.linear_columns:
                start_col = span.start_col + max(start, span.text_start) - span.text_start
                end_col = span.start_col + min(end, span.text_end) - span.text_start
            else:
                start_col = span.start_col
                end_col = span.end_col
            previous = segments[-1] if segments else None
            if previous is not None and previous.row == span.row and start_col <= previous.end_col:
                previous.end_col = max(previous.end_col, end_col)
            else:
                segments.append(AltScreenSearchSegment(row=span.row, start_col=start_col, end_col=end_col))
        while span_index < len(spans) and spans[span_index].text_end <= end:
            span_index += 1
        if segments:
            matches.append(AltScreenSearchMatch(segments=segments))

    return matches


@dataclass(slots=True)
class AltScreenSearchResult:
    matches: list[AltScreenSearchMatch]
    changed: bool


class AltScreenSearchIndex:
    """Cache the searchable corpus and matches while rendered transcript lines remain unchanged."""

    def __init__(self) -> None:
        self._source_lines: list[str] | None = None
        self._corpus: _SearchCorpus | None = None
        self._normalized_query: str | None = None
        self._matches: list[AltScreenSearchMatch] = []

    def search(self, lines: list[str], query: str) -> AltScreenSearchResult:
        source_changed = self._source_lines is None or len(self._source_lines) != len(lines)
        if not source_changed and self._source_lines is not None:
            for index, line in enumerate(lines):
                if self._source_lines[index] != line:
                    source_changed = True
                    break
        if source_changed or self._corpus is None:
            self._source_lines = list(lines)
            self._corpus = _build_search_corpus(lines)

        normalized_query = _normalize_query(query)
        changed = source_changed or normalized_query != self._normalized_query
        if changed:
            self._normalized_query = normalized_query
            self._matches = _find_search_corpus_matches(self._corpus, normalized_query)
        return AltScreenSearchResult(matches=self._matches, changed=changed)


def find_alt_screen_search_matches(lines: list[str], query: str) -> list[AltScreenSearchMatch]:
    normalized_query = _normalize_query(query)
    return _find_search_corpus_matches(_build_search_corpus(lines), normalized_query) if normalized_query else []


def get_alt_screen_search_match_key(match: AltScreenSearchMatch) -> str:
    if not match.segments:
        return ""
    first = match.segments[0]
    last = match.segments[-1]
    return f"{first.row}:{first.start_col}:{last.row}:{last.end_col}"


def _format_key(key: str | None) -> str:
    if not key:
        return "Unbound"
    parts = []
    for part in key.split("+"):
        if sys.platform == "darwin" and part.lower() == "alt":
            parts.append("Option")
        else:
            parts.append(part[:1].upper() + part[1:])
    return "+".join(parts)


class AltScreenSearchComponent:
    def __init__(self, on_query_change, navigation_button_style=None) -> None:
        self._input = Input(
            {
                "prompt": " ",
                "placeholder": "Find in transcript",
                "placeholderStyle": lambda text: f"\x1b[2m{text}\x1b[22m",
            }
        )
        self._on_query_change = on_query_change
        self._navigation_button_style = navigation_button_style or (lambda text, _hovered: text)
        self._result_count = 0
        self._result_index = -1
        self._previous_button_start = -1
        self._previous_button_end = -1
        self._next_button_start = -1
        self._next_button_end = -1
        self._hovered_navigation_direction: int | None = None
        self._focused = False

    @property
    def focused(self) -> bool:
        return self._focused

    @focused.setter
    def focused(self, value: bool) -> None:
        self._focused = value
        self._input.focused = value

    def set_result(self, index: int, count: int) -> None:
        self._result_index = index
        self._result_count = count

    def get_navigation_direction_at(self, row: int, column: int) -> int | None:
        """-1 over the previous button, 1 over the next button, else None."""
        if row != 2:
            return None
        if self._previous_button_start <= column < self._previous_button_end:
            return -1
        if self._next_button_start <= column < self._next_button_end:
            return 1
        return None

    def set_hovered_navigation_direction(self, direction: int | None) -> bool:
        if direction == self._hovered_navigation_direction:
            return False
        self._hovered_navigation_direction = direction
        return True

    async def handle_input(self, data: str) -> None:
        previous = self._input.get_value()
        await self._input.handle_input(data)
        query = self._input.get_value()
        if query != previous:
            self._on_query_change(query)

    def invalidate(self) -> None:
        self._input.invalidate()

    def render(self, width: int) -> list[str]:
        safe_width = max(1, width)
        inner_width = max(0, safe_width - 2)
        keybindings = get_keybindings()
        previous_keys = keybindings.get_keys("tui.altScreen.searchPrevious")
        next_keys = keybindings.get_keys("tui.altScreen.searchNext")
        previous_key = _format_key(previous_keys[0] if previous_keys else None)
        next_key = _format_key(next_keys[0] if next_keys else None)
        query = self._input.get_value()
        if not query:
            result = ""
        elif self._result_count == 0:
            result = "No matches"
        else:
            result = f"{self._result_index + 1}/{self._result_count}"
        result_space = max(0, inner_width - 3)
        visible_result = truncate_to_width(result, result_space, "")
        result_text = f"\x1b[2m {visible_result} \x1b[22m" if visible_result else ""
        input_width = max(0, inner_width - visible_width(result_text))
        rendered_input = self._input.render(max(1, input_width))
        input_line = truncate_to_width(rendered_input[0] if rendered_input else "", input_width, "")
        input_padding = " " * max(0, input_width - visible_width(input_line))
        content = f"{input_line}{input_padding}{result_text}"

        previous_button = f"↑ {previous_key}"
        next_button = f"↓ {next_key}"
        separator = " · "
        outer_gap_width = 1
        available_controls_width = max(0, inner_width - outer_gap_width * 2 - 1)
        controls_width = visible_width(previous_button) + visible_width(separator) + visible_width(next_button)
        if controls_width > available_controls_width:
            previous_button = "↑"
            next_button = "↓"
            separator = " "
            controls_width = visible_width(previous_button) + visible_width(separator) + visible_width(next_button)
        show_buttons = controls_width <= available_controls_width
        rendered_buttons = (
            self._navigation_button_style(previous_button, self._hovered_navigation_direction == -1)
            + separator
            + self._navigation_button_style(next_button, self._hovered_navigation_direction == 1)
            if show_buttons
            else ""
        )
        outer_gaps_width = outer_gap_width * 2 if show_buttons else 0
        right_rule_width = 1 if rendered_buttons and inner_width > controls_width + outer_gaps_width else 0
        left_rule_width = max(
            0, inner_width - (controls_width if show_buttons else 0) - outer_gaps_width - right_rule_width
        )
        previous_start = 1 + left_rule_width + outer_gap_width
        self._previous_button_start = previous_start if show_buttons else -1
        self._previous_button_end = previous_start + visible_width(previous_button) if show_buttons else -1
        self._next_button_start = self._previous_button_end + visible_width(separator) if show_buttons else -1
        self._next_button_end = self._next_button_start + visible_width(next_button) if show_buttons else -1

        if safe_width == 1:
            return ["┌", "│", "└"]
        gap = " " if rendered_buttons else ""
        return [
            f"┌{'─' * inner_width}┐",
            f"│{content}│",
            f"└{'─' * left_rule_width}{gap}{rendered_buttons}{gap}{'─' * right_rule_width}┘",
        ]
