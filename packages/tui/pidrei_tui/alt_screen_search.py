"""Transcript search over rendered alt-screen content (port of pi `alt-screen-search.ts`).

The corpus maps every character of the whitespace-collapsed searchable text back
to its source `{row, startCol, endCol}` cell span; matches are re-grouped into
per-row segments for highlighting. pi maps UTF-16 code units, Python maps code
points — both are self-consistent with their regex engines' match offsets.
"""

import re
from dataclasses import dataclass, field

import grapheme as grapheme_lib

from .components.input import Input
from .utils import strip_terminal_sequences, truncate_to_width, visible_width


_WHITESPACE_RE = re.compile(r"^\s+$")


@dataclass(slots=True)
class AltScreenSearchSegment:
    row: int
    start_col: int
    end_col: int


@dataclass(slots=True)
class AltScreenSearchMatch:
    segments: list[AltScreenSearchSegment] = field(default_factory=list)


@dataclass(slots=True)
class _SearchCorpus:
    text: str = ""
    source: list[AltScreenSearchSegment | None] = field(default_factory=list)


def _append_mapped_text(text: str, span: AltScreenSearchSegment | None, corpus: _SearchCorpus) -> None:
    corpus.text += text
    for _ in text:
        corpus.source.append(span)


def _build_search_corpus(lines: list[str]) -> _SearchCorpus:
    corpus = _SearchCorpus()
    pending_separator = False

    for row, raw_line in enumerate(lines):
        line = strip_terminal_sequences(raw_line or "")
        column = 0
        for text in grapheme_lib.graphemes(line):
            width = visible_width(text)
            if _WHITESPACE_RE.match(text):
                if len(corpus.text) > 0:
                    pending_separator = True
                column += width
                continue
            if pending_separator:
                _append_mapped_text(" ", None, corpus)
                pending_separator = False
            _append_mapped_text(text, AltScreenSearchSegment(row=row, start_col=column, end_col=column + width), corpus)
            column += width
        if len(corpus.text) > 0:
            pending_separator = True

    return corpus


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip()


def find_alt_screen_search_matches(lines: list[str], query: str) -> list[AltScreenSearchMatch]:
    normalized_query = _normalize_query(query)
    if not normalized_query:
        return []

    corpus = _build_search_corpus(lines)
    expression = re.compile(re.escape(normalized_query), re.IGNORECASE)
    matches: list[AltScreenSearchMatch] = []

    for match in expression.finditer(corpus.text):
        start = match.start()
        end = start + len(match.group(0))
        segments: list[AltScreenSearchSegment] = []
        for index in range(start, end):
            span = corpus.source[index]
            if span is None:
                continue
            previous = segments[-1] if segments else None
            if previous is not None and previous.row == span.row and span.start_col <= previous.end_col:
                previous.end_col = max(previous.end_col, span.end_col)
            else:
                segments.append(AltScreenSearchSegment(row=span.row, start_col=span.start_col, end_col=span.end_col))
        if segments:
            matches.append(AltScreenSearchMatch(segments=segments))

    return matches


def get_alt_screen_search_match_key(match: AltScreenSearchMatch) -> str:
    if not match.segments:
        return ""
    first = match.segments[0]
    last = match.segments[-1]
    return f"{first.row}:{first.start_col}:{last.row}:{last.end_col}"


class AltScreenSearchComponent:
    def __init__(self, on_query_change) -> None:
        self._input = Input()
        self._on_query_change = on_query_change
        self._result_count = 0
        self._result_index = -1
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
        label = " Find transcript"
        query = self._input.get_value()
        if not query:
            status = ""
        elif self._result_count == 0:
            status = "No matches "
        else:
            status = f"{self._result_index + 1}/{self._result_count} "
        label_width = visible_width(label)
        status_width = visible_width(status)
        gap = " " * max(1, safe_width - label_width - status_width)
        title = truncate_to_width(f"{label}{gap}{status}", safe_width, "")
        padding = " " * max(0, safe_width - visible_width(title))
        return [f"\x1b[7m{title}{padding}\x1b[27m", *self._input.render(safe_width)]
