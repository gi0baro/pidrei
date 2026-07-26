"""Shared diff computation utilities for the edit and similar tools (port of pi `tools/edit-diff.ts`).

Diff engine decision (PLAN open decision 3, resolved here): pi uses the JS
`diff` package (Myers). The port adapts stdlib `difflib` instead —
`SequenceMatcher(autojunk=False)` over newline-terminated line tokens produces
jsdiff-shaped parts for the display diff, and `difflib.unified_diff` produces
the unified patch. Both emit valid diffs; hunk content can differ from jsdiff
only where multiple minimal diffs exist.

Other approximations (documented): per-line `trimEnd` in fuzzy normalization
uses `str.rstrip()` (a slight superset of JS WhiteSpace), and the
"\\ No newline at end of file" marker is emitted for the common tail case.
"""

import difflib
import itertools
import re
import unicodedata
from dataclasses import dataclass
from typing import Literal


def detect_line_ending(content: str) -> Literal["\r\n", "\n"]:
    crlf_idx = content.find("\r\n")
    lf_idx = content.find("\n")
    if lf_idx == -1:
        return "\n"
    if crlf_idx == -1:
        return "\n"
    return "\r\n" if crlf_idx < lf_idx else "\n"


def normalize_to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def restore_line_endings(text: str, ending: Literal["\r\n", "\n"]) -> str:
    return text.replace("\n", "\r\n") if ending == "\r\n" else text


def normalize_for_fuzzy_match(text: str) -> str:
    """Normalize text for fuzzy matching. Applies progressive transformations:

    - Strip trailing whitespace from each line
    - Normalize smart quotes to ASCII equivalents
    - Normalize Unicode dashes/hyphens to ASCII hyphen
    - Normalize special Unicode spaces to regular space
    """
    normalized = unicodedata.normalize("NFKC", text)
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    # Smart single quotes → '
    normalized = re.sub(r"[‘’‚‛]", "'", normalized)
    # Smart double quotes → "
    normalized = re.sub(r"[“”„‟]", '"', normalized)
    # Various dashes/hyphens → -
    normalized = re.sub(r"[‐‑‒–—―−]", "-", normalized)
    # Special spaces → regular space
    normalized = re.sub("[\u00a0\u2002-\u200a\u202f\u205f\u3000]", " ", normalized)
    return normalized


def _split_lines_with_endings(content: str) -> list[str]:
    return re.findall(r"[^\n]*\n|[^\n]+", content)


@dataclass(slots=True)
class _LineSpan:
    start: int
    end: int


@dataclass(slots=True)
class TextReplacement:
    match_index: int
    match_length: int
    new_text: str


@dataclass(slots=True)
class _MatchedEdit(TextReplacement):
    edit_index: int = 0


def _get_line_spans(content: str) -> list[_LineSpan]:
    offset = 0
    spans: list[_LineSpan] = []
    for line in _split_lines_with_endings(content):
        span = _LineSpan(start=offset, end=offset + len(line))
        offset = span.end
        spans.append(span)
    return spans


def _get_replacement_line_range(lines: list[_LineSpan], replacement: TextReplacement) -> tuple[int, int]:
    replacement_start = replacement.match_index
    replacement_end = replacement.match_index + replacement.match_length

    start_line = -1
    for index, line in enumerate(lines):
        if line.start <= replacement_start < line.end:
            start_line = index
            break
    if start_line == -1:
        raise Exception("Replacement range is outside the base content.")

    end_line = start_line
    while end_line < len(lines) and lines[end_line].end < replacement_end:
        end_line += 1
    if end_line >= len(lines):
        raise Exception("Replacement range is outside the base content.")

    return start_line, end_line + 1


def _apply_replacements(content: str, replacements: list[TextReplacement], offset: int = 0) -> str:
    result = content
    for replacement in reversed(replacements):
        match_index = replacement.match_index - offset
        result = result[:match_index] + replacement.new_text + result[match_index + replacement.match_length :]
    return result


def apply_replacements_preserving_unchanged_lines(
    original_content: str,
    base_content: str,
    replacements: list[TextReplacement],
) -> str:
    """Apply replacements matched against `base_content` to `original_content`
    while preserving unchanged line blocks from the original.

    Useful when `base_content` is a normalized view of the original. Each
    replacement is widened to the lines it actually touches, those touched
    lines are rewritten from the normalized base, and all other lines are
    copied back from `original_content`.
    """
    original_lines = _split_lines_with_endings(original_content)
    base_lines = _get_line_spans(base_content)
    if len(original_lines) != len(base_lines):
        raise Exception("Cannot preserve unchanged lines because the base content has a different line count.")

    @dataclass(slots=True)
    class _Group:
        start_line: int
        end_line: int
        replacements: list[TextReplacement]

    groups: list[_Group] = []
    for replacement in sorted(replacements, key=lambda entry: entry.match_index):
        start_line, end_line = _get_replacement_line_range(base_lines, replacement)
        current = groups[-1] if groups else None
        if current is not None and start_line < current.end_line:
            current.end_line = max(current.end_line, end_line)
            current.replacements.append(replacement)
            continue
        groups.append(_Group(start_line=start_line, end_line=end_line, replacements=[replacement]))

    original_line_index = 0
    result = ""
    for group in groups:
        result += "".join(original_lines[original_line_index : group.start_line])

        group_start_offset = base_lines[group.start_line].start
        group_end_offset = base_lines[group.end_line - 1].end
        result += _apply_replacements(
            base_content[group_start_offset:group_end_offset],
            group.replacements,
            group_start_offset,
        )
        original_line_index = group.end_line
    result += "".join(original_lines[original_line_index:])

    return result


@dataclass(slots=True)
class FuzzyMatchResult:
    # Whether a match was found.
    found: bool
    # The index where the match starts (in the content that should be used for replacement).
    index: int
    # Length of the matched text.
    match_length: int
    # Whether fuzzy matching was used (False = exact match).
    used_fuzzy_match: bool
    # The content to use for replacement operations. When exact match: original
    # content. When fuzzy match: normalized content.
    content_for_replacement: str


@dataclass(slots=True)
class Edit:
    old_text: str
    new_text: str


@dataclass(slots=True)
class AppliedEditsResult:
    base_content: str
    new_content: str


def fuzzy_find_text(content: str, old_text: str) -> FuzzyMatchResult:
    """Find old_text in content, trying exact match first, then fuzzy match.

    When fuzzy matching is used, the returned content_for_replacement is the
    fuzzy-normalized version of the content.
    """
    exact_index = content.find(old_text)
    if exact_index != -1:
        return FuzzyMatchResult(
            found=True,
            index=exact_index,
            match_length=len(old_text),
            used_fuzzy_match=False,
            content_for_replacement=content,
        )

    # Try fuzzy match - work entirely in normalized space.
    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old_text = normalize_for_fuzzy_match(old_text)
    fuzzy_index = fuzzy_content.find(fuzzy_old_text)

    if fuzzy_index == -1:
        return FuzzyMatchResult(
            found=False, index=-1, match_length=0, used_fuzzy_match=False, content_for_replacement=content
        )

    return FuzzyMatchResult(
        found=True,
        index=fuzzy_index,
        match_length=len(fuzzy_old_text),
        used_fuzzy_match=True,
        content_for_replacement=fuzzy_content,
    )


def strip_bom(content: str) -> tuple[str, str]:
    """Strip UTF-8 BOM if present; returns (bom, text)."""
    if content.startswith("﻿"):
        return "﻿", content[1:]
    return "", content


def _count_occurrences(content: str, old_text: str) -> int:
    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old_text = normalize_for_fuzzy_match(old_text)
    return fuzzy_content.count(fuzzy_old_text)


def _not_found_error(path: str, edit_index: int, total_edits: int) -> Exception:
    if total_edits == 1:
        return Exception(
            f"Could not find the exact text in {path}. "
            "The old text must match exactly including all whitespace and newlines."
        )
    return Exception(
        f"Could not find edits[{edit_index}] in {path}. "
        "The oldText must match exactly including all whitespace and newlines."
    )


def _duplicate_error(path: str, edit_index: int, total_edits: int, occurrences: int) -> Exception:
    if total_edits == 1:
        return Exception(
            f"Found {occurrences} occurrences of the text in {path}. The text must be unique. "
            "Please provide more context to make it unique."
        )
    return Exception(
        f"Found {occurrences} occurrences of edits[{edit_index}] in {path}. Each oldText must be unique. "
        "Please provide more context to make it unique."
    )


def _empty_old_text_error(path: str, edit_index: int, total_edits: int) -> Exception:
    if total_edits == 1:
        return Exception(f"oldText must not be empty in {path}.")
    return Exception(f"edits[{edit_index}].oldText must not be empty in {path}.")


def _no_change_error(path: str, total_edits: int) -> Exception:
    if total_edits == 1:
        return Exception(
            f"No changes made to {path}. The replacement produced identical content. "
            "This might indicate an issue with special characters or the text not existing as expected."
        )
    return Exception(f"No changes made to {path}. The replacements produced identical content.")


def apply_edits_to_normalized_content(normalized_content: str, edits: list[Edit], path: str) -> AppliedEditsResult:
    """Apply one or more exact-text replacements to LF-normalized content.

    All edits are matched against the same original content. Replacements are
    then applied in reverse order so offsets remain stable. If any edit needs
    fuzzy matching, the operation runs in fuzzy-normalized content space and
    then overlays those line-level changes onto the original content so
    unchanged line blocks keep their original bytes.
    """
    normalized_edits = [
        Edit(old_text=normalize_to_lf(edit.old_text), new_text=normalize_to_lf(edit.new_text)) for edit in edits
    ]

    for index, edit in enumerate(normalized_edits):
        if len(edit.old_text) == 0:
            raise _empty_old_text_error(path, index, len(normalized_edits))

    initial_matches = [fuzzy_find_text(normalized_content, edit.old_text) for edit in normalized_edits]
    used_fuzzy_match = any(match.used_fuzzy_match for match in initial_matches)
    replacement_base_content = normalize_for_fuzzy_match(normalized_content) if used_fuzzy_match else normalized_content

    matched_edits: list[_MatchedEdit] = []
    for index, edit in enumerate(normalized_edits):
        match_result = fuzzy_find_text(replacement_base_content, edit.old_text)
        if not match_result.found:
            raise _not_found_error(path, index, len(normalized_edits))

        occurrences = _count_occurrences(replacement_base_content, edit.old_text)
        if occurrences > 1:
            raise _duplicate_error(path, index, len(normalized_edits), occurrences)

        matched_edits.append(
            _MatchedEdit(
                match_index=match_result.index,
                match_length=match_result.match_length,
                new_text=edit.new_text,
                edit_index=index,
            )
        )

    matched_edits.sort(key=lambda entry: entry.match_index)
    for previous, current in itertools.pairwise(matched_edits):
        if previous.match_index + previous.match_length > current.match_index:
            raise Exception(
                f"edits[{previous.edit_index}] and edits[{current.edit_index}] overlap in {path}. "
                "Merge them into one edit or target disjoint regions."
            )

    base_content = normalized_content
    new_content = (
        apply_replacements_preserving_unchanged_lines(normalized_content, replacement_base_content, matched_edits)
        if used_fuzzy_match
        else _apply_replacements(replacement_base_content, matched_edits)
    )

    if base_content == new_content:
        raise _no_change_error(path, len(normalized_edits))

    return AppliedEditsResult(base_content=base_content, new_content=new_content)


# --- diff generation ----------------------------------------------------------


@dataclass(slots=True)
class DiffPart:
    """One change run, shaped like a jsdiff `diffLines` part."""

    value: str
    added: bool = False
    removed: bool = False


def diff_lines(old_content: str, new_content: str) -> list[DiffPart]:
    old_tokens = _split_lines_with_endings(old_content)
    new_tokens = _split_lines_with_endings(new_content)
    matcher = difflib.SequenceMatcher(None, old_tokens, new_tokens, autojunk=False)
    parts: list[DiffPart] = []
    for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if tag == "equal":
            parts.append(DiffPart(value="".join(old_tokens[old_start:old_end])))
        elif tag == "delete":
            parts.append(DiffPart(value="".join(old_tokens[old_start:old_end]), removed=True))
        elif tag == "insert":
            parts.append(DiffPart(value="".join(new_tokens[new_start:new_end]), added=True))
        else:  # replace: jsdiff emits the removed run before the added run
            parts.append(DiffPart(value="".join(old_tokens[old_start:old_end]), removed=True))
            parts.append(DiffPart(value="".join(new_tokens[new_start:new_end]), added=True))
    return parts


def generate_unified_patch(path: str, old_content: str, new_content: str, context_lines: int = 4) -> str:
    """Generate a standard unified patch (pi: jsdiff `createTwoFilesPatch` with file headers only)."""
    old_tokens = _split_lines_with_endings(old_content)
    new_tokens = _split_lines_with_endings(new_content)
    old_missing_newline = bool(old_tokens) and not old_tokens[-1].endswith("\n")
    new_missing_newline = bool(new_tokens) and not new_tokens[-1].endswith("\n")
    old_last = old_tokens[-1] + "\n" if old_missing_newline else (old_tokens[-1] if old_tokens else "")
    new_last = new_tokens[-1] + "\n" if new_missing_newline else (new_tokens[-1] if new_tokens else "")
    if old_missing_newline:
        old_tokens[-1] += "\n"
    if new_missing_newline:
        new_tokens[-1] += "\n"

    lines = list(difflib.unified_diff(old_tokens, new_tokens, fromfile=path, tofile=path, n=context_lines))

    if old_missing_newline or new_missing_newline:
        annotated: list[str] = []
        for index, line in enumerate(lines):
            annotated.append(line)
            is_last_of_kind = all(other != line for other in lines[index + 1 :])
            if not is_last_of_kind:
                continue
            if (
                old_missing_newline
                and line in (f"-{old_last}", f" {old_last}")
                or new_missing_newline
                and line in (f"+{new_last}", f" {new_last}")
            ):
                annotated.append("\\ No newline at end of file\n")
        lines = annotated

    return "".join(lines)


def generate_diff_string(old_content: str, new_content: str, context_lines: int = 4) -> tuple[str, int | None]:
    """Generate a display-oriented diff string with line numbers and context.

    Returns (diff string, first changed line number in the new file).
    """
    parts = diff_lines(old_content, new_content)
    output: list[str] = []

    old_lines = old_content.split("\n")
    new_lines = new_content.split("\n")
    max_line_num = max(len(old_lines), len(new_lines))
    line_num_width = len(str(max_line_num))

    old_line_num = 1
    new_line_num = 1
    last_was_change = False
    first_changed_line: int | None = None

    for index, part in enumerate(parts):
        raw = part.value.split("\n")
        if raw and raw[-1] == "":
            raw.pop()

        if part.added or part.removed:
            # Capture the first changed line (in the new file).
            if first_changed_line is None:
                first_changed_line = new_line_num

            for line in raw:
                if part.added:
                    output.append(f"+{str(new_line_num).rjust(line_num_width)} {line}")
                    new_line_num += 1
                else:
                    output.append(f"-{str(old_line_num).rjust(line_num_width)} {line}")
                    old_line_num += 1
            last_was_change = True
        else:
            # Context lines - only show a few before/after changes.
            next_part_is_change = index < len(parts) - 1 and (parts[index + 1].added or parts[index + 1].removed)
            has_leading_change = last_was_change
            has_trailing_change = next_part_is_change

            if has_leading_change and has_trailing_change:
                if len(raw) <= context_lines * 2:
                    for line in raw:
                        output.append(f" {str(old_line_num).rjust(line_num_width)} {line}")
                        old_line_num += 1
                        new_line_num += 1
                else:
                    leading_lines = raw[:context_lines]
                    trailing_lines = raw[len(raw) - context_lines :]
                    skipped_lines = len(raw) - len(leading_lines) - len(trailing_lines)

                    for line in leading_lines:
                        output.append(f" {str(old_line_num).rjust(line_num_width)} {line}")
                        old_line_num += 1
                        new_line_num += 1

                    output.append(f" {''.rjust(line_num_width)} ...")
                    old_line_num += skipped_lines
                    new_line_num += skipped_lines

                    for line in trailing_lines:
                        output.append(f" {str(old_line_num).rjust(line_num_width)} {line}")
                        old_line_num += 1
                        new_line_num += 1
            elif has_leading_change:
                shown_lines = raw[:context_lines]
                skipped_lines = len(raw) - len(shown_lines)

                for line in shown_lines:
                    output.append(f" {str(old_line_num).rjust(line_num_width)} {line}")
                    old_line_num += 1
                    new_line_num += 1

                if skipped_lines > 0:
                    output.append(f" {''.rjust(line_num_width)} ...")
                    old_line_num += skipped_lines
                    new_line_num += skipped_lines
            elif has_trailing_change:
                skipped_lines = max(0, len(raw) - context_lines)
                if skipped_lines > 0:
                    output.append(f" {''.rjust(line_num_width)} ...")
                    old_line_num += skipped_lines
                    new_line_num += skipped_lines

                for line in raw[skipped_lines:]:
                    output.append(f" {str(old_line_num).rjust(line_num_width)} {line}")
                    old_line_num += 1
                    new_line_num += 1
            else:
                # Skip these context lines entirely.
                old_line_num += len(raw)
                new_line_num += len(raw)

            last_was_change = False

    return "\n".join(output), first_changed_line
