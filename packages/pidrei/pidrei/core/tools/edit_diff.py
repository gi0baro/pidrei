"""Mirror of pi coding-agent src/core/tools/edit-diff.ts: shared diff
computation utilities for the edit and similar tools, and the preview helpers
(compute_edits_diff / compute_edit_diff).

Diff engine decision (PLAN open decision 3, resolved here): pi uses the JS
`diff` package (Myers). The port adapts stdlib `difflib` instead:
`SequenceMatcher(autojunk=False)` over newline-terminated line tokens produces
jsdiff-shaped parts for the display diff, and its grouped opcodes produce the
unified patch hunks (`_unified_patch_hunks`, which numbers them from a line
offset where pi shifts jsdiff `structuredPatch` hunks). Both emit valid diffs;
hunk content can differ from jsdiff only where multiple minimal diffs exist.

Other approximations (documented): per-line `trimEnd` in fuzzy normalization
uses `str.rstrip()` (a slight superset of JS WhiteSpace).
"""

import difflib
import errno
import itertools
import re
import unicodedata
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Literal

from tonio.colored import fs

from ...utils.text import split_bom
from .path_utils import resolve_to_cwd


def normalize_to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def apply_replacements_to_original(original_content: str, replacements: list[TextReplacement]) -> str:
    """Apply replacements matched in `normalize_to_lf(original_content)` to `original_content` itself.

    Text outside the replaced ranges keeps its bytes, including its line endings. Line breaks in
    the new text get the line ending of the line where the replaced range starts. The result,
    normalized to LF, equals the replacements applied to the normalized content.
    """
    # Only CRLF changes length when normalizing: the LF of the n-th CRLF (at original offset p) is at
    # p - n in the normalized text. Adding the number of CRLFs before a normalized offset gives the
    # original offset, and a range that covers that LF covers the whole CRLF.
    crlf_before = 0
    next_crlf = original_content.find("\r\n")

    def to_original(offset: int) -> int:
        nonlocal crlf_before, next_crlf
        while next_crlf != -1 and next_crlf - crlf_before < offset:
            crlf_before += 1
            next_crlf = original_content.find("\r\n", next_crlf + 2)
        return offset + crlf_before

    parts: list[str] = []
    ends_with_cr = False

    def push(part: str) -> None:
        nonlocal ends_with_cr
        if not part:
            return
        # A CR before a seam and an LF after it would merge into one CRLF line break, for example a kept
        # lone CR followed by inserted text that starts with LF, or both sides of a deleted range. Adding
        # an LF turns the CR into a CRLF of its own, so the written file has the line breaks the diff shows.
        if ends_with_cr and part[0] == "\n":
            parts.append("\n")
        parts.append(part)
        ends_with_cr = part[-1] == "\r"

    position = 0
    for replacement in replacements:
        start = to_original(replacement.match_index)
        push(original_content[position:start])
        ending = _get_line_ending_at(original_content, start)
        push(replacement.new_text if ending == "\n" else replacement.new_text.replace("\n", ending))
        position = to_original(replacement.match_index + replacement.match_length)
    push(original_content[position:])
    return "".join(parts)


def _get_line_ending_at(content: str, offset: int) -> Literal["\r\n", "\n", "\r"]:
    """Line ending of the line containing `offset`: its own LF or CRLF, or that of the line before
    for an unterminated last line. A lone CR only counts in files without any LF (classic Mac
    line endings); elsewhere it is more likely a control character inside a line.
    """
    line_feed = content.find("\n", offset)
    if line_feed == -1:
        line_feed = content.rfind("\n", 0, offset + 1)
    if line_feed == -1:
        return "\r" if "\r" in content else "\n"
    # JS reads `content[-1]` as undefined; a Python index would wrap around.
    return "\r\n" if line_feed > 0 and content[line_feed - 1] == "\r" else "\n"


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
    """Replacement of `match_length` characters at `match_index` with `new_text`."""

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


def _widen_replacements_to_lines(
    original_content: str,
    base_content: str,
    replacements: list[TextReplacement],
) -> list[TextReplacement]:
    """Turn replacements matched against `base_content` into replacements of
    `original_content` that keep unchanged line blocks from the original.

    Useful when `base_content` is a normalized view of the original. Each
    replacement is widened to the lines it actually touches, and those touched
    lines are rewritten from the normalized base; all other lines keep their
    original text. The actual replacement ranges drive preservation so
    duplicate normalized lines cannot be aligned to the wrong occurrence.
    """
    original_lines = _get_line_spans(original_content)
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

    widened: list[TextReplacement] = []
    for group in groups:
        group_start_offset = base_lines[group.start_line].start
        group_end_offset = base_lines[group.end_line - 1].end
        original_start = original_lines[group.start_line].start
        widened.append(
            TextReplacement(
                match_index=original_start,
                match_length=original_lines[group.end_line - 1].end - original_start,
                new_text=_apply_replacements(
                    base_content[group_start_offset:group_end_offset],
                    group.replacements,
                    group_start_offset,
                ),
            )
        )
    return widened


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
    # Sorted replacements of `base_content` that turn it into `new_content`
    replacements: list[TextReplacement]


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
    replacements: list[TextReplacement] = (
        _widen_replacements_to_lines(normalized_content, replacement_base_content, matched_edits)
        if used_fuzzy_match
        else matched_edits
    )
    new_content = _apply_replacements(base_content, replacements)

    if base_content == new_content:
        raise _no_change_error(path, len(normalized_edits))

    return AppliedEditsResult(base_content=base_content, new_content=new_content, replacements=replacements)


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


def _format_range_unified(start: int, stop: int) -> str:
    """A unified hunk range from 0-based `[start, stop)`, as `difflib.unified_diff` writes it."""
    beginning = start + 1
    length = stop - start
    if length == 1:
        return f"{beginning}"
    if not length:
        beginning -= 1
    return f"{beginning},{length}"


def _patch_lines(prefix: str, token: str) -> list[str]:
    # Only the last line of a text can lack its line feed: jsdiff marks it, and it differs from the
    # same text with one (a line feed added or removed at the end of the file is a change).
    if token.endswith("\n"):
        return [prefix + token]
    return [f"{prefix}{token}\n", "\\ No newline at end of file\n"]


def _unified_patch_hunks(
    old_text: str, new_text: str, context_lines: int, old_line_offset: int = 0, new_line_offset: int = 0
) -> list[str]:
    """Hunk lines of the unified patch from `old_text` to `new_text`, numbered as if the texts
    started after `old_line_offset` and `new_line_offset` lines: what pi gets by shifting jsdiff
    `structuredPatch` hunks. The grouping is `difflib.unified_diff`'s; the matcher is the display
    diff's (`diff_lines`), so the patch and the display diff describe the same alignment.
    """
    old_tokens = _split_lines_with_endings(old_text)
    new_tokens = _split_lines_with_endings(new_text)
    matcher = difflib.SequenceMatcher(None, old_tokens, new_tokens, autojunk=False)
    lines: list[str] = []
    for group in matcher.get_grouped_opcodes(context_lines):
        first, last = group[0], group[-1]
        old_range = _format_range_unified(first[1] + old_line_offset, last[2] + old_line_offset)
        new_range = _format_range_unified(first[3] + new_line_offset, last[4] + new_line_offset)
        lines.append(f"@@ -{old_range} +{new_range} @@\n")
        for tag, old_start, old_end, new_start, new_end in group:
            if tag == "equal":
                for token in old_tokens[old_start:old_end]:
                    lines.extend(_patch_lines(" ", token))
                continue
            if tag in ("replace", "delete"):
                for token in old_tokens[old_start:old_end]:
                    lines.extend(_patch_lines("-", token))
            if tag in ("replace", "insert"):
                for token in new_tokens[new_start:new_end]:
                    lines.extend(_patch_lines("+", token))
    return lines


def _format_patch(path: str, hunk_lines: list[str]) -> str:
    """The patch text with file headers only (pi: `formatPatch` with `FILE_HEADERS_ONLY`)."""
    if not hunk_lines:
        return ""
    return f"--- {path}\n+++ {path}\n" + "".join(hunk_lines)


def generate_unified_patch(path: str, old_content: str, new_content: str, context_lines: int = 4) -> str:
    """Generate a standard unified patch (pi: jsdiff `createTwoFilesPatch` with file headers only)."""
    return _format_patch(path, _unified_patch_hunks(old_content, new_content, context_lines))


def generate_diff_string(old_content: str, new_content: str, context_lines: int = 4) -> tuple[str, int | None]:
    """Generate a display-oriented diff string with line numbers and context.

    Returns (diff string, first changed line number in the new file).
    """
    line_num_width = len(str(max(_count_lines(old_content), _count_lines(new_content))))
    output, first_changed_line = _format_display_diff(
        diff_lines(old_content, new_content), 1, 1, line_num_width, context_lines
    )
    return "\n".join(output), first_changed_line


def _count_lines(text: str) -> int:
    """Number of lines as `text.split("\\n")` counts them, without building the list."""
    return _count_line_feeds(text, 0, len(text)) + 1


def _count_line_feeds(text: str, start: int, end: int) -> int:
    return text.count("\n", start, end)


def _format_display_diff(
    parts: list[DiffPart],
    old_start_line: int,
    new_start_line: int,
    line_num_width: int,
    context_lines: int,
) -> tuple[list[str], int | None]:
    """Format diff line parts as display diff lines, numbering from the given 1-based old and new lines."""
    output: list[str] = []

    old_line_num = old_start_line
    new_line_num = new_start_line
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

    return output, first_changed_line


@dataclass(frozen=True, slots=True)
class EditDiff:
    """Display diff and unified patch of an edit, as returned in the edit tool's details."""

    diff: str
    patch: str
    first_changed_line: int | None


# Unchanged lines diffed on each side of a replacement. Well above the 4 context lines shown, so the
# matcher aligns a change within its window as it would over the whole file.
DIFF_WINDOW_MARGIN_LINES = 32


def generate_edit_diff(
    path: str,
    base_content: str,
    new_content: str,
    replacements: list[TextReplacement],
    context_lines: int = 4,
) -> EditDiff:
    """Diff an edit only around its replacements, so time and memory follow the size of the change
    instead of the file. The result matches generate_diff_string and generate_unified_patch over the
    whole contents, except that the matcher may align runs of repeated lines differently within a window.
    """
    line_num_width = len(str(max(_count_lines(base_content), _count_lines(new_content))))
    skip_marker = f" {''.rjust(line_num_width)} ..."
    output: list[str] = []
    hunk_lines: list[str] = []
    first_changed_line: int | None = None
    for window in _get_diff_windows(base_content, replacements):
        old_text = base_content[window.old_start : window.old_end]
        new_text = new_content[window.new_start : window.new_end]
        shown, shown_first_changed_line = _format_display_diff(
            diff_lines(old_text, new_text),
            window.old_line + 1,
            window.new_line + 1,
            line_num_width,
            context_lines,
        )
        if first_changed_line is None:
            first_changed_line = shown_first_changed_line
        # Both windows around a run of unchanged lines mark it as skipped; the whole-file diff shows one marker.
        if output and output[-1] == skip_marker and shown and shown[0] == skip_marker:
            shown.pop(0)
        output.extend(shown)
        hunk_lines.extend(_unified_patch_hunks(old_text, new_text, context_lines, window.old_line, window.new_line))
    return EditDiff(
        diff="\n".join(output), patch=_format_patch(path, hunk_lines), first_changed_line=first_changed_line
    )


@dataclass(slots=True)
class _DiffWindow:
    """Matching ranges of the old and new contents around one or more replacements: offsets and 0-based first lines."""

    old_start: int
    old_end: int
    old_line: int
    new_start: int
    new_end: int
    new_line: int


def _line_start_before(text: str, offset: int) -> int:
    """Start of the line that contains `offset`."""
    return 0 if offset == 0 else text.rfind("\n", 0, offset) + 1


def _get_diff_windows(base_content: str, replacements: list[TextReplacement]) -> list[_DiffWindow]:
    """Whole-line windows with margin around the replacements, which are sorted and disjoint; overlapping windows merge."""
    windows: list[_DiffWindow] = []
    # Offset and line deltas of the replacements before the current one, and a line-counting cursor.
    offset_delta = 0
    line_delta = 0
    cursor_offset = 0
    cursor_line = 0
    for replacement in replacements:
        match_end = replacement.match_index + replacement.match_length
        start = _line_start_before(base_content, replacement.match_index)
        for _ in range(DIFF_WINDOW_MARGIN_LINES):
            if start <= 0:
                break
            start = _line_start_before(base_content, start - 1)
        end = match_end
        for i in range(DIFF_WINDOW_MARGIN_LINES + 1):
            if end >= len(base_content):
                break
            line_feed = base_content.find("\n", max(match_end - 1, replacement.match_index) if i == 0 else end)
            end = len(base_content) if line_feed == -1 else line_feed + 1
        replacement_offset_delta = len(replacement.new_text) - replacement.match_length
        replacement_line_delta = _count_line_feeds(
            replacement.new_text, 0, len(replacement.new_text)
        ) - _count_line_feeds(base_content, replacement.match_index, match_end)
        previous = windows[-1] if windows else None
        if previous is not None and start <= previous.old_end:
            previous.old_end = max(previous.old_end, end)
            previous.new_end = previous.old_end + offset_delta + replacement_offset_delta
        else:
            cursor_line += _count_line_feeds(base_content, cursor_offset, start)
            cursor_offset = start
            windows.append(
                _DiffWindow(
                    old_start=start,
                    old_end=end,
                    old_line=cursor_line,
                    new_start=start + offset_delta,
                    new_end=end + offset_delta + replacement_offset_delta,
                    new_line=cursor_line + line_delta,
                )
            )
        offset_delta += replacement_offset_delta
        line_delta += replacement_line_delta
    return windows


@dataclass(slots=True)
class EditDiffResult:
    diff: str
    first_changed_line: int | None


@dataclass(slots=True)
class EditDiffError:
    error: str


def _error_code_message(error: Exception) -> str:
    code = _errno_code(error)
    return f"Error code: {code}" if code else str(error)


def _errno_code(error: Exception) -> str | None:
    if isinstance(error, OSError) and error.errno is not None:
        return errno.errorcode.get(error.errno)
    return None


async def compute_edits_diff(path: str, edits: list[Edit], cwd: str) -> EditDiffResult | EditDiffError:
    """Compute the diff for one or more edit operations without applying them.
    Used for preview rendering before the tool executes."""
    absolute_path = resolve_to_cwd(path, cwd)

    try:
        # Check if file exists and is readable
        try:
            raw_content = await fs.Path(absolute_path).read_text(encoding="utf-8", newline="")
        except OSError as error:
            return EditDiffError(error=f"Could not edit file: {path}. {_error_code_message(error)}.")

        # Strip BOM before matching (LLM won't include invisible BOM in oldText)
        _bom, content = split_bom(raw_content)
        normalized_content = normalize_to_lf(content)
        applied = apply_edits_to_normalized_content(normalized_content, edits, path)

        # Generate the diff
        edit_diff = generate_edit_diff(path, applied.base_content, applied.new_content, applied.replacements)
        return EditDiffResult(diff=edit_diff.diff, first_changed_line=edit_diff.first_changed_line)
    except Exception as error:
        return EditDiffError(error=str(error))


def compute_edit_diff(path: str, old_text: str, new_text: str, cwd: str) -> Awaitable[EditDiffResult | EditDiffError]:
    """Compute the diff for a single edit operation without applying it."""
    return compute_edits_diff(path, [Edit(old_text=old_text, new_text=new_text)], cwd)
