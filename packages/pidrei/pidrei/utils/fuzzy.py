"""Mirror of pi tui src/fuzzy.ts (the subset needed outside the TUI).

Fuzzy matching utilities. Matches if all query characters appear in order
(not necessarily consecutive). Lower score = better match. The Phase 4 TUI
slice reuses this module.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


_WORD_BOUNDARY_RE = re.compile(r"[\s\-_./:]")
_ALPHA_NUMERIC_RE = re.compile(r"^(?P<letters>[a-z]+)(?P<digits>[0-9]+)$")
_NUMERIC_ALPHA_RE = re.compile(r"^(?P<digits>[0-9]+)(?P<letters>[a-z]+)$")
_TOKEN_SPLIT_RE = re.compile(r"[\s/]+")


@dataclass(slots=True)
class FuzzyMatch:
    matches: bool
    score: float


def fuzzy_match(query: str, text: str) -> FuzzyMatch:
    query_lower = query.lower()
    text_lower = text.lower()

    def match_query(normalized_query: str) -> FuzzyMatch:
        if len(normalized_query) == 0:
            return FuzzyMatch(matches=True, score=0)

        if len(normalized_query) > len(text_lower):
            return FuzzyMatch(matches=False, score=0)

        query_index = 0
        score = 0.0
        last_match_index = -1
        consecutive_matches = 0

        for i, char in enumerate(text_lower):
            if query_index >= len(normalized_query):
                break
            if char == normalized_query[query_index]:
                is_word_boundary = i == 0 or _WORD_BOUNDARY_RE.match(text_lower[i - 1]) is not None

                # Reward consecutive matches
                if last_match_index == i - 1:
                    consecutive_matches += 1
                    score -= consecutive_matches * 5
                else:
                    consecutive_matches = 0
                    # Penalize gaps
                    if last_match_index >= 0:
                        score += (i - last_match_index - 1) * 2

                # Reward word boundary matches
                if is_word_boundary:
                    score -= 10

                # Slight penalty for later matches
                score += i * 0.1

                last_match_index = i
                query_index += 1

        if query_index < len(normalized_query):
            return FuzzyMatch(matches=False, score=0)

        if normalized_query == text_lower:
            score -= 100

        return FuzzyMatch(matches=True, score=score)

    primary_match = match_query(query_lower)
    if primary_match.matches:
        return primary_match

    alpha_numeric_match = _ALPHA_NUMERIC_RE.match(query_lower)
    numeric_alpha_match = _NUMERIC_ALPHA_RE.match(query_lower)
    if alpha_numeric_match:
        swapped_query = f"{alpha_numeric_match.group('digits')}{alpha_numeric_match.group('letters')}"
    elif numeric_alpha_match:
        swapped_query = f"{numeric_alpha_match.group('letters')}{numeric_alpha_match.group('digits')}"
    else:
        swapped_query = ""

    if not swapped_query:
        return primary_match

    swapped_match = match_query(swapped_query)
    if not swapped_match.matches:
        return primary_match

    return FuzzyMatch(matches=True, score=swapped_match.score + 5)


def fuzzy_filter(items: list[Any], query: str, get_text: Callable[[Any], str]) -> list[Any]:
    """Filter and sort items by fuzzy match quality (best matches first).

    Supports whitespace- and slash-separated tokens: all tokens must match.
    """
    if not query.strip():
        return items

    tokens = [t for t in _TOKEN_SPLIT_RE.split(query.strip()) if t]

    if not tokens:
        return items

    results: list[tuple[Any, float]] = []

    for item in items:
        text = get_text(item)
        total_score = 0.0
        all_match = True

        for token in tokens:
            match = fuzzy_match(token, text)
            if match.matches:
                total_score += match.score
            else:
                all_match = False
                break

        if all_match:
            results.append((item, total_score))

    results.sort(key=lambda entry: entry[1])
    return [item for item, _score in results]
