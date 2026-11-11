"""Mirror of pi coding-agent test/read-streaming.test.ts.

pi keeps its pre-streaming text algorithm as the reference; `_reference_read`
is pidrei's pre-streaming text path. pi hooks `TextDecoder` to append to a
file mid-read; here the read's own `_LineSelector` is swapped for a subclass
that appends on its first chunk. pi's single matrix case runs once per text
here, so a slow input shows by name.

Not mirrored:
- "keeps details key order, which session files serialize": the details are
  a dataclass, whose field order is fixed.
- "reads a 200 MB line with a 64 MB heap": it runs the read in a Node child
  process with a capped V8 heap; there is no such heap cap to run under here.
- The fractional cases of the `read_file` fallback (`offset: 2.5, limit: 1`
  and `offset: 1, limit: 1.5`): pi's output there comes from JS array
  indexing with fractional numbers (`Use offset=3.5`), while pidrei floors
  `offset` and `limit` with `int()`.
"""

import os

import pytest

import pidrei.core.tools.read as read_module
from pidrei.core.tools.read import ReadToolDetails, create_read_tool_definition
from pidrei.core.tools.truncate import DEFAULT_MAX_BYTES, format_size, truncate_head, utf8_byte_length
from pidrei_utils.cancel import CancelToken


FILE_PATH = "/pidrei-read-streaming-test-missing/file.txt"


def _reference_read(text: str, path: str, offset, limit):
    """The read tool's text path before it streamed, kept as the reference the new path must match."""
    all_lines = text.split("\n")
    total_file_lines = len(all_lines)
    start_line = max(0, int(offset) - 1) if offset else 0
    start_line_display = start_line + 1
    if start_line >= len(all_lines):
        raise Exception(f"Offset {offset} is beyond end of file ({len(all_lines)} lines total)")
    user_limited_lines = None
    if limit is not None:
        end_line = min(start_line + int(limit), len(all_lines))
        selected_content = "\n".join(all_lines[start_line:end_line])
        user_limited_lines = end_line - start_line
    else:
        selected_content = "\n".join(all_lines[start_line:])
    truncation = truncate_head(selected_content)
    details = None
    if truncation.first_line_exceeds_limit:
        first_line_size = format_size(utf8_byte_length(all_lines[start_line]))
        output_text = (
            f"[Line {start_line_display} is {first_line_size}, exceeds {format_size(DEFAULT_MAX_BYTES)} "
            f"limit. Use bash: sed -n '{start_line_display}p' {path} | head -c {DEFAULT_MAX_BYTES}]"
        )
        details = ReadToolDetails(truncation=truncation)
    elif truncation.truncated:
        end_line_display = start_line_display + truncation.output_lines - 1
        next_offset = end_line_display + 1
        output_text = truncation.content
        if truncation.truncated_by == "lines":
            output_text += (
                f"\n\n[Showing lines {start_line_display}-{end_line_display} of {total_file_lines}. "
                f"Use offset={next_offset} to continue.]"
            )
        else:
            output_text += (
                f"\n\n[Showing lines {start_line_display}-{end_line_display} of {total_file_lines} "
                f"({format_size(DEFAULT_MAX_BYTES)} limit). Use offset={next_offset} to continue.]"
            )
        details = ReadToolDetails(truncation=truncation)
    elif user_limited_lines is not None and start_line + user_limited_lines < len(all_lines):
        remaining = len(all_lines) - (start_line + user_limited_lines)
        next_offset = start_line + user_limited_lines + 1
        output_text = f"{truncation.content}\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"
    else:
        output_text = truncation.content
    return {"text": output_text, "details": details}


def _reference_outcome(text: str, offset, limit) -> dict:
    try:
        return _reference_read(text, FILE_PATH, offset, limit)
    except Exception as error:
        return {"error": str(error)}


async def _outcome(definition, params: dict) -> dict:
    try:
        result = await definition.execute("call", params, None, None, None)
    except Exception as error:
        return {"error": str(error)}
    return {"text": result.content[0].text, "details": result.details}


class _StreamingOperations:
    """Splits the bytes into chunks of `chunk_size`, so characters and lines split at chunk ends."""

    def __init__(self, data: bytes, chunk_size: int):
        self._data = data
        self._chunk_size = chunk_size

    async def read_file(self, _absolute_path: str) -> bytes:
        raise Exception("read_file must not be used when create_read_stream is present")

    async def access(self, _absolute_path: str) -> None:
        return None

    async def create_read_stream(self, _absolute_path: str, *, cancel=None):
        for position in range(0, len(self._data), self._chunk_size):
            yield self._data[position : position + self._chunk_size]


class _ReadFileOperations:
    def __init__(self, data: bytes):
        self._data = data

    async def read_file(self, _absolute_path: str) -> bytes:
        return self._data

    async def access(self, _absolute_path: str) -> None:
        return None


class _StreamingWithReadFileOperations(_StreamingOperations):
    async def read_file(self, _absolute_path: str) -> bytes:
        return self._data


def _read_with(ops, offset, limit):
    definition = create_read_tool_definition(os.getcwd(), operations=ops)
    return _outcome(definition, {"path": FILE_PATH, "offset": offset, "limit": limit})


def _lines(count: int, make) -> str:
    return "\n".join(make(i) for i in range(count))


FIFTY_KB_OF_LINES = f"{_lines(512, lambda _i: 'y' * 99)}y"

# Inputs around the boundaries the notices and truncation depend on.
TEXT_CASES = {
    "empty": "",
    "no trailing newline": "alpha\nbeta\ngamma",
    "trailing newline": "alpha\nbeta\n",
    "CRLF": "alpha\r\nbeta\r\n\r\ngamma\r\n",
    "multi-byte characters and UTF-8 byte order mark": "﻿é€😀\nñ\U00010348字\n😀😀😀",
    "2001 lines": _lines(2001, lambda i: f"line {i}"),
    # truncate_head drops a trailing empty line: the empty line at the line cap must not end the stored lines.
    "2001 lines, an empty line and more": f"{_lines(2001, lambda i: f'line {i}')}\n\nmore",
    "first line over 50KB": f"{'é' * (DEFAULT_MAX_BYTES // 2 + 1)}\nsecond",
    "line over 50KB after short lines": f"a\nb\n{'z' * (60 * 1024)}\nc",
    # 50KB + 1 byte: truncated, yet every line fits, so truncate_head reports truncated_by "lines".
    "exactly 50KB and a trailing newline": f"{FIFTY_KB_OF_LINES}\n",
    "exactly 50KB, an empty line and more": f"{FIFTY_KB_OF_LINES}\n\nmore",
    "multi-byte characters over 50KB": _lines(700, lambda i: f"{i} {'€😀' * 13}"),
}


@pytest.mark.tonio
@pytest.mark.parametrize("name", list(TEXT_CASES))
async def test_gives_the_same_output_as_the_whole_file_algorithm(name):
    text = TEXT_CASES[name]
    data = text.encode()
    total = len(text.split("\n"))
    for offset in (None, 2, 3, total, total + 1):
        for limit in (None, 0, 1, total):
            expected = _reference_outcome(text, offset, limit)
            for chunk_size in (1, 4096):
                label = f"{name}, offset {offset}, limit {limit}, chunks of {chunk_size}"
                assert await _read_with(_StreamingOperations(data, chunk_size), offset, limit) == expected, label


@pytest.mark.tonio
async def test_decodes_multi_byte_characters_split_across_chunks_and_invalid_utf8_like_a_whole_decode():
    data = "aé€😀\n😀b\U00010348\n".encode() + bytes(
        [0x61, 0xC3, 0x0A, 0xE2, 0x82, 0x0A, 0xFF, 0xFE, 0x80, 0xF0, 0x9F, 0x98, 0x0A, 0xED, 0xA0, 0x80]
    )
    for chunk_size in (1, 2, 3):
        assert await _read_with(_StreamingOperations(data, chunk_size), None, None) == _reference_outcome(
            data.decode("utf-8", "replace"), None, None
        )


@pytest.mark.tonio
async def test_uses_read_file_for_operations_without_create_read_stream_and_for_negative_ranges():
    text = "one\ntwo\nthree\nfour"
    for offset, limit in ((None, None), (2, 1), (9, None)):
        assert await _read_with(_ReadFileOperations(text.encode()), offset, limit) == _reference_outcome(
            text, offset, limit
        )
    # List slicing defines this one, as JS array slicing does in pi: limit -1 selects all lines but the last.
    assert await _read_with(_StreamingWithReadFileOperations(text.encode(), 1), None, -1) == _reference_outcome(
        text, None, -1
    )


@pytest.mark.tonio
async def test_reads_local_files_through_the_default_path(tmp_path):
    path = tmp_path / "local.txt"
    text = _lines(3000, lambda i: f"{i} {'w' * (i % 40)}")
    path.write_text(text)
    definition = create_read_tool_definition(str(tmp_path))
    for offset, limit in ((None, None), (1500, 10)):
        result = await _outcome(definition, {"path": str(path), "offset": offset, "limit": limit})
        assert result == _reference_read(text, str(path), offset, limit)
    # pi expects EISDIR from its stream; here the access check opens the path first.
    with pytest.raises(IsADirectoryError):
        await definition.execute("call", {"path": str(tmp_path)}, None, None, None)


@pytest.mark.tonio
async def test_reads_a_local_file_up_to_its_size_when_opened(tmp_path, monkeypatch):
    path = tmp_path / "growing.txt"
    # Larger than one 64 KB chunk, so the append lands before the read reaches the end.
    path.write_text(f"head\n{'x' * 200_000}")

    class _AppendingSelector(read_module._LineSelector):
        appended = False

        def feed(self, chunk: bytes) -> None:
            # Runs on the pool, inside the read's job: the file is open and partly read.
            if not _AppendingSelector.appended:
                _AppendingSelector.appended = True
                with open(path, "a", encoding="utf-8") as file:
                    file.write("\nadded")
            super().feed(chunk)

    monkeypatch.setattr(read_module, "_LineSelector", _AppendingSelector)
    definition = create_read_tool_definition(str(tmp_path))

    result = await definition.execute("call", {"path": str(path), "limit": 1}, None, None, None)

    assert _AppendingSelector.appended
    assert result.content[0].text == "head\n\n[1 more lines in file. Use offset=2 to continue.]"


@pytest.mark.tonio
async def test_stops_consuming_the_stream_when_aborted():
    token = CancelToken()
    chunks_read = 0
    closed = False
    received: list = []

    class _Operations:
        async def read_file(self, _absolute_path: str) -> bytes:
            return b""

        async def access(self, _absolute_path: str) -> None:
            return None

        async def create_read_stream(self, _absolute_path: str, *, cancel=None):
            nonlocal chunks_read, closed
            received.append(cancel)
            try:
                while True:
                    chunks_read += 1
                    if chunks_read == 3:
                        token.cancel()
                    yield b"line\n"
            finally:
                closed = True

    definition = create_read_tool_definition(os.getcwd(), operations=_Operations())

    with pytest.raises(Exception, match="Operation aborted"):
        await definition.execute("call", {"path": FILE_PATH}, token, None, None)

    assert received == [token]
    assert closed is True
    assert chunks_read == 3
