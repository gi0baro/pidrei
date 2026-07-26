"""Shared truncation utilities for tool outputs (port of pi `harness/utils/truncate.ts`).

Truncation is based on two independent limits - whichever is hit first wins:
- Line limit (default: 2000 lines)
- Byte limit (default: 50KB)

Never returns partial lines (except bash tail truncation edge case).

Byte math mirrors JS `Buffer.byteLength`: strings are measured as UTF-16 code
units, adjacent surrogate chars count as one 4-byte pair, and unpaired
surrogates count 3 bytes (they encode to U+FFFD). Python strings can carry
lone surrogates, so the port walks characters with the same pairing rules.
"""

from dataclasses import dataclass
from typing import Literal


DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024  # 50KB
GREP_MAX_LINE_LENGTH = 500  # Max chars per grep match line


@dataclass(slots=True)
class TruncationResult:
    # The truncated content.
    content: str
    # Whether truncation occurred.
    truncated: bool
    # Which limit was hit: "lines", "bytes", or None if not truncated.
    truncated_by: Literal["lines", "bytes"] | None
    # Total number of lines in the original content.
    total_lines: int
    # Total number of bytes in the original content.
    total_bytes: int
    # Number of complete lines in the truncated output.
    output_lines: int
    # Number of bytes in the truncated output.
    output_bytes: int
    # Whether the last line was partially truncated (only for tail truncation edge case).
    last_line_partial: bool
    # Whether the first line exceeded the byte limit (for head truncation).
    first_line_exceeds_limit: bool
    # The max lines limit that was applied.
    max_lines: int
    # The max bytes limit that was applied.
    max_bytes: int


@dataclass(slots=True)
class TruncatedLine:
    text: str
    was_truncated: bool


def _is_high_surrogate(code: int) -> bool:
    return 0xD800 <= code <= 0xDBFF


def _is_low_surrogate(code: int) -> bool:
    return 0xDC00 <= code <= 0xDFFF


def utf8_byte_length(content: str) -> int:
    """UTF-8 byte length under JS `Buffer.byteLength` semantics."""
    length = 0
    index = 0
    size = len(content)
    while index < size:
        code = ord(content[index])
        if code <= 0x7F:
            length += 1
        elif code <= 0x7FF:
            length += 2
        elif _is_high_surrogate(code) and index + 1 < size and _is_low_surrogate(ord(content[index + 1])):
            length += 4
            index += 1
        elif code <= 0xFFFF:
            # Includes unpaired surrogates, which encode to U+FFFD (3 bytes).
            length += 3
        else:
            # Astral code point: a surrogate pair in UTF-16.
            length += 4
        index += 1
    return length


def _split_lines_for_counting(content: str) -> list[str]:
    if not content:
        return []
    lines = content.split("\n")
    if content.endswith("\n"):
        lines.pop()
    return lines


def _replace_unpaired_surrogates(content: str) -> str:
    output: list[str] = []
    index = 0
    size = len(content)
    while index < size:
        char = content[index]
        code = ord(char)
        if _is_high_surrogate(code):
            if index + 1 < size and _is_low_surrogate(ord(content[index + 1])):
                output.append(char)
                output.append(content[index + 1])
                index += 2
                continue
            output.append("�")
        elif _is_low_surrogate(code):
            output.append("�")
        else:
            output.append(char)
        index += 1
    return "".join(output)


def format_size(size_bytes: float) -> str:
    """Format bytes as human-readable size."""
    if size_bytes < 1024:
        # :g mirrors JS number stringification ("12B", "12.5B").
        return f"{size_bytes:g}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f}MB"


def truncate_head(
    content: str, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES
) -> TruncationResult:
    """Truncate content from the head (keep first N lines/bytes).

    Suitable for file reads where you want to see the beginning. Never returns
    partial lines. If the first line exceeds the byte limit, returns empty
    content with `first_line_exceeds_limit=True`.
    """
    total_bytes = utf8_byte_length(content)
    lines = _split_lines_for_counting(content)
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content,
            truncated=False,
            truncated_by=None,
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=total_lines,
            output_bytes=total_bytes,
            last_line_partial=False,
            first_line_exceeds_limit=False,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    # Check if first line alone exceeds the byte limit.
    if utf8_byte_length(lines[0]) > max_bytes:
        return TruncationResult(
            content="",
            truncated=True,
            truncated_by="bytes",
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=0,
            output_bytes=0,
            last_line_partial=False,
            first_line_exceeds_limit=True,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    # Collect complete lines that fit.
    output_lines: list[str] = []
    output_bytes_count = 0
    truncated_by: Literal["lines", "bytes"] = "lines"

    for index, line in enumerate(lines[:max_lines]):
        line_bytes = utf8_byte_length(line) + (1 if index > 0 else 0)  # +1 for newline
        if output_bytes_count + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        output_lines.append(line)
        output_bytes_count += line_bytes

    # If we exited due to line limit.
    if len(output_lines) >= max_lines and output_bytes_count <= max_bytes:
        truncated_by = "lines"

    output_content = "\n".join(output_lines)
    return TruncationResult(
        content=output_content,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(output_lines),
        output_bytes=utf8_byte_length(output_content),
        last_line_partial=False,
        first_line_exceeds_limit=False,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def truncate_tail(
    content: str, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES
) -> TruncationResult:
    """Truncate content from the tail (keep last N lines/bytes).

    Suitable for bash output where you want to see the end (errors, final
    results). May return a partial first line if the last line of original
    content exceeds the byte limit.
    """
    total_bytes = utf8_byte_length(content)
    lines = _split_lines_for_counting(content)
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content,
            truncated=False,
            truncated_by=None,
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=total_lines,
            output_bytes=total_bytes,
            last_line_partial=False,
            first_line_exceeds_limit=False,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    # Work backwards from the end.
    output_lines: list[str] = []
    output_bytes_count = 0
    truncated_by: Literal["lines", "bytes"] = "lines"
    last_line_partial = False

    for line in reversed(lines):
        if len(output_lines) >= max_lines:
            break
        line_bytes = utf8_byte_length(line) + (1 if output_lines else 0)  # +1 for newline
        if output_bytes_count + line_bytes > max_bytes:
            truncated_by = "bytes"
            # Edge case: if we haven't added ANY lines yet and this line exceeds
            # max_bytes, take the end of the line (partial).
            if not output_lines:
                truncated_line = _truncate_string_to_bytes_from_end(line, max_bytes)
                output_lines.insert(0, truncated_line)
                output_bytes_count = utf8_byte_length(truncated_line)
                last_line_partial = True
            break
        output_lines.insert(0, line)
        output_bytes_count += line_bytes

    # If we exited due to line limit.
    if len(output_lines) >= max_lines and output_bytes_count <= max_bytes:
        truncated_by = "lines"

    output_content = "\n".join(output_lines)
    return TruncationResult(
        content=output_content,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(output_lines),
        output_bytes=utf8_byte_length(output_content),
        last_line_partial=last_line_partial,
        first_line_exceeds_limit=False,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def _truncate_string_to_bytes_from_end(content: str, max_bytes: int) -> str:
    """Truncate a string to fit within a byte limit (from the end).

    Handles multi-byte UTF-8 characters correctly, including surrogate pairs
    split across Python characters.
    """
    if max_bytes <= 0:
        return ""

    output_bytes = 0
    start = len(content)
    needs_replacement = False
    index = len(content)
    while index > 0:
        character_start = index - 1
        code = ord(content[character_start])
        unpaired_surrogate = False
        if _is_low_surrogate(code) and character_start > 0:
            if _is_high_surrogate(ord(content[character_start - 1])):
                character_start -= 1
                character_bytes = 4
            else:
                character_bytes = 3
                unpaired_surrogate = True
        elif 0xD800 <= code <= 0xDFFF:
            character_bytes = 3
            unpaired_surrogate = True
        elif code > 0xFFFF:
            # Astral code point: a surrogate pair in UTF-16.
            character_bytes = 4
        else:
            character_bytes = 1 if code <= 0x7F else (2 if code <= 0x7FF else 3)
        if output_bytes + character_bytes > max_bytes:
            break
        output_bytes += character_bytes
        start = character_start
        needs_replacement = needs_replacement or unpaired_surrogate
        index = character_start

    output = content[start:]
    return _replace_unpaired_surrogates(output) if needs_replacement else output


def truncate_line(line: str, max_chars: int = GREP_MAX_LINE_LENGTH) -> TruncatedLine:
    """Truncate a single line to max characters, adding a [truncated] suffix.

    Used for grep match lines.
    """
    if len(line) <= max_chars:
        return TruncatedLine(text=line, was_truncated=False)
    return TruncatedLine(text=f"{line[:max_chars]}... [truncated]", was_truncated=True)
