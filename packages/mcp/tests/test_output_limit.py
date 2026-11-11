"""Mirror of pi mcp test/output-limit.test.ts.

The last case also carries the multi-byte case pi removed from
mcp-extension.test.ts with `truncateMiddle` (PiDrei had it in
`test_mcp_extension.py` and `test_tools.py`).
"""

import re

import pytest

from pidrei_mcp import MCP_OUTPUT_MAX_BYTES, LlmContent, limit_mcp_content


_IMAGE: LlmContent = {"type": "image", "data": "AAAA", "mimeType": "image/png"}


def _text_of(content: list[LlmContent]) -> str:
    first = content[0] if content else None
    if first is None or first["type"] != "text":
        raise AssertionError("expected a text block first")
    return first["text"]


@pytest.mark.tonio
async def test_returns_content_within_the_limit_as_it_is_without_saving():
    content: list[LlmContent] = [{"type": "text", "text": "x" * MCP_OUTPUT_MAX_BYTES}, _IMAGE]
    saves = 0

    async def save(_data, _extension) -> str:
        nonlocal saves
        saves += 1
        return "/tmp/full.txt"

    limited = await limit_mcp_content(content, save)
    assert limited == (content, None)
    assert limited.content is content
    assert saves == 0


@pytest.mark.tonio
async def test_cuts_the_middle_of_longer_text_saves_the_full_text_and_keeps_images():
    full = "\n".join(f"line {index + 1}" for index in range(3000))
    saved: list[tuple[str | bytes, str]] = []
    split = full.index("\n", 100)

    async def save(data, extension) -> str:
        saved.append((data, extension))
        return "/tmp/full.txt"

    limited = await limit_mcp_content(
        [{"type": "text", "text": full[:split]}, _IMAGE, {"type": "text", "text": full[split + 1 :]}], save
    )
    # Text blocks are joined with newlines before the limit applies.
    assert saved == [(full, ".txt")]
    assert limited.full_output_path == "/tmp/full.txt"
    assert len(limited.content) == 2
    assert limited.content[1] == _IMAGE
    text = _text_of(limited.content)
    assert re.match(
        rf"Warning: truncated output \(original token count: {-(-len(full) // 4)}\)\n"
        r"Total output lines: 3000\n\nline 1\nline 2\n",
        text,
    )
    assert re.search(r"…\d+ chars truncated…", text)
    assert text.endswith("line 3000\n\n[Full output: /tmp/full.txt (read it with offset/limit)]")
    assert len(text.encode()) < MCP_OUTPUT_MAX_BYTES + 1024


@pytest.mark.tonio
async def test_reports_a_failed_save_and_names_no_file_without_a_saver():
    content: list[LlmContent] = [{"type": "text", "text": "y" * (MCP_OUTPUT_MAX_BYTES + 1)}]

    async def fail(_data, _extension) -> str:
        raise OSError("disk full")

    failed = await limit_mcp_content(content, fail)
    assert failed.full_output_path is None
    assert _text_of(failed.content).endswith("\n\n[Could not save the full output: disk full]")

    unsaved = await limit_mcp_content(content)
    assert unsaved.full_output_path is None
    assert re.fullmatch(r"Warning: truncated output .*…\d+ chars truncated…y+", _text_of(unsaved.content), re.DOTALL)


@pytest.mark.tonio
async def test_cuts_multi_byte_text_only_at_character_boundaries():
    # After the one-byte prefix, characters start at odd bytes, and both cuts (byte 10240 and byte
    # 40004 - 10240) are even: each lands inside a two-byte character.
    text = f"a{'é' * 20_000}end"
    limited = _text_of((await limit_mcp_content([{"type": "text", "text": text}])).content)
    assert "�" not in limited
    assert limited.endswith("end")
    marker = re.search(r"…(\d+) chars truncated…", limited)
    assert marker is not None
    head, tail = limited[limited.index("\n\n") + 2 :].split(marker.group(0))
    assert len(head.encode()) <= MCP_OUTPUT_MAX_BYTES / 2
    assert len(tail.encode()) <= MCP_OUTPUT_MAX_BYTES / 2
    assert len(head) + len(tail) + int(marker.group(1)) == len(text)
