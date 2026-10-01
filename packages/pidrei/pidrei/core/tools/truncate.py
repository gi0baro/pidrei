"""Mirror of pi coding-agent src/core/tools/truncate.ts.

pi keeps two byte-identical truncation implementations (agent harness and
coding-agent); the Phase 2 port in pidrei-agent is the single implementation
here, re-exported under the coding-agent module path.
"""

from dataclasses import dataclass

from pidrei_agent.harness.utils.truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    GREP_MAX_LINE_LENGTH,
    TruncatedLine,
    TruncationResult,
    format_size,
    truncate_head,
    truncate_line,
    truncate_tail,
    utf8_byte_length,
)


__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_LINES",
    "GREP_MAX_LINE_LENGTH",
    "MiddleTruncationResult",
    "TruncatedLine",
    "TruncationResult",
    "format_size",
    "truncate_head",
    "truncate_line",
    "truncate_middle",
    "truncate_tail",
    "utf8_byte_length",
]


@dataclass(slots=True, frozen=True)
class MiddleTruncationResult:
    # The start and end of the content with a `…N chars truncated…` marker between them.
    content: str
    truncated: bool
    # Characters left out.
    removed_chars: int
    total_bytes: int
    total_lines: int


def truncate_middle(content: str, max_bytes: int) -> MiddleTruncationResult:
    """Keep the start and the end of `content`, half of `max_bytes` each, and
    replace the middle with a `…N chars truncated…` marker, like Codex does for
    tool output. Cuts only at character boundaries."""
    buf = content.encode("utf-8", "replace")
    lines = content.split("\n") if content else []
    if content.endswith("\n"):
        lines.pop()
    total_lines = len(lines)
    if len(buf) <= max_bytes:
        return MiddleTruncationResult(
            content=content, truncated=False, removed_chars=0, total_bytes=len(buf), total_lines=total_lines
        )

    def is_boundary(index: int) -> bool:
        # Continuation bytes (10xxxxxx) are not character starts.
        return index >= len(buf) or (buf[index] & 0xC0) != 0x80

    head_end = max_bytes // 2
    while head_end > 0 and not is_boundary(head_end):
        head_end -= 1
    tail_start = len(buf) - (max_bytes - max_bytes // 2)
    while tail_start < len(buf) and not is_boundary(tail_start):
        tail_start += 1
    head = buf[:head_end].decode("utf-8", "replace")
    tail = buf[tail_start:].decode("utf-8", "replace")
    removed_chars = len(buf[head_end:tail_start].decode("utf-8", "replace"))
    return MiddleTruncationResult(
        content=f"{head}…{removed_chars} chars truncated…{tail}",
        truncated=True,
        removed_chars=removed_chars,
        total_bytes=len(buf),
        total_lines=total_lines,
    )
