"""Mirror of pi mcp src/output-limit.ts.

The size limit of the model-facing text of MCP results. Longer text keeps
its start and end with the middle cut out, like Codex does, and the caller
may save the full text where the model can read it.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import NamedTuple

from .protocol.content import LlmContent


# Model-facing text of an MCP result beyond this is cut in the middle.
MCP_OUTPUT_MAX_BYTES = 20 * 1024

# Saves data the model cannot get in full, such as the full text of a
# truncated result, and returns where it was saved (a path the model can
# read). The extension includes the dot, for example `.txt`.
type McpOutputSaver = Callable[[str | bytes, str], Awaitable[str]]


class LimitedMcpContent(NamedTuple):
    """Model-facing content within the limit, and where the full text went when it was saved."""

    content: list[LlmContent]
    full_output_path: str | None = None


async def limit_mcp_content(content: list[LlmContent], save_output: McpOutputSaver | None = None) -> LimitedMcpContent:
    """Keep model-facing text within `MCP_OUTPUT_MAX_BYTES`. Longer text
    becomes one text block in Codex's truncation format, followed by the path
    of the file with the full text when `save_output` is given; images follow
    it. Content within the limit is returned as it is."""
    combined = "\n".join(block["text"] for block in content if block["type"] == "text")
    truncation = _truncate_middle(combined, MCP_OUTPUT_MAX_BYTES)
    if truncation is None:
        return LimitedMcpContent(content)
    full_output_path: str | None = None
    where = ""
    if save_output is not None:
        try:
            full_output_path = await save_output(combined, ".txt")
            where = f"\n\n[Full output: {full_output_path} (read it with offset/limit)]"
        except Exception as error:
            where = f"\n\n[Could not save the full output: {error}]"
    tokens = -(-truncation.total_bytes // 4)
    text = (
        f"Warning: truncated output (original token count: {tokens})\n"
        f"Total output lines: {truncation.total_lines}\n\n{truncation.content}{where}"
    )
    images = [block for block in content if block["type"] == "image"]
    return LimitedMcpContent([{"type": "text", "text": text}, *images], full_output_path or None)


@dataclass(frozen=True, slots=True)
class _MiddleTruncation:
    """Text cut in the middle, with the size of the whole text."""

    # The start and the end of the text with a `…N chars truncated…` marker between them.
    content: str
    total_bytes: int
    total_lines: int


def _truncate_middle(text: str, max_bytes: int) -> _MiddleTruncation | None:
    """Keep the start and the end of `text`, half of `max_bytes` each, and
    replace the middle with a `…N chars truncated…` marker. Cuts only at
    character boundaries. None when `text` fits."""
    buf = text.encode("utf-8", "replace")
    if len(buf) <= max_bytes:
        return None

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
    # A final newline ends the last line instead of starting another.
    lines = len(text.split("\n")) - (1 if text.endswith("\n") else 0)
    return _MiddleTruncation(
        content=f"{head}…{removed_chars} chars truncated…{tail}", total_bytes=len(buf), total_lines=lines
    )
