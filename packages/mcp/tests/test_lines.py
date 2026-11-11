"""Mirror of pi mcp test/lines.test.ts, and of the "PipeTransport framing"
cases of test/pipe.test.ts, which specify the shared framing (`PipeTransport`
is not ported) and run here against `JsonRpcLineReader` directly.

"joins a message split across many chunks once" spies on `Buffer.concat`
being called once; it ports as its correctness half only, and the
every-chunk-size case covers the joins.
"""

import pytest

from pidrei_mcp.protocol.jsonrpc import stringify
from pidrei_mcp.transports.lines import JsonRpcLineReader


def _start_reader(**options):
    messages: list = []
    errors: list[str] = []
    reader = JsonRpcLineReader(on_message=messages.append, on_error=lambda error: errors.append(str(error)), **options)
    return reader, messages, errors


def _feed(reader: JsonRpcLineReader, data: bytes, chunk_size: int) -> None:
    for i in range(0, len(data), chunk_size):
        reader.read_stdout(data[i : i + chunk_size])


def _ping(id_: int) -> dict:
    return {"jsonrpc": "2.0", "id": id_, "method": "ping"}


# pipe.test.ts, "PipeTransport framing"


def test_joins_messages_split_across_chunks_and_splits_several_messages_in_one_chunk():
    reader, messages, errors = _start_reader()
    first, second, third = (stringify(_ping(n)) for n in (1, 2, 3))
    reader.read_stdout(first[:10].encode())
    reader.read_stdout(f"{first[10:]}\n{second}".encode())
    reader.read_stdout(f"\n{third}\n".encode())
    assert messages == [_ping(1), _ping(2), _ping(3)]
    assert errors == []


def test_strips_a_trailing_carriage_return_and_skips_blank_lines():
    reader, messages, errors = _start_reader()
    reader.read_stdout(f"\n\r\n   \n{stringify(_ping(1))}\r\n\n".encode())
    assert messages == [_ping(1)]
    assert errors == []


def test_reports_invalid_messages_and_keeps_reading():
    reader, messages, errors = _start_reader()
    reader.read_stdout(f'not json\n{{"jsonrpc":"1.0"}}\n{stringify(_ping(1))}\n'.encode())
    assert messages == [_ping(1)]
    assert len(errors) == 2


def test_drops_messages_over_max_message_bytes_complete_or_not():
    reader, messages, errors = _start_reader(max_message_bytes=48)
    reader.read_stdout(f"{stringify({'jsonrpc': '2.0', 'method': 'x' * 48})}\n{stringify(_ping(1))}\n".encode())
    reader.read_stdout(b"y" * 49)
    reader.read_stdout(f"\n{stringify(_ping(2))}\n".encode())
    assert messages == [_ping(1), _ping(2)]
    assert errors == ["MCP stdio message exceeds 48 bytes", "MCP stdio message exceeds 48 bytes"]


@pytest.mark.tonio
async def test_keeps_a_bounded_stderr_tail_and_passes_every_chunk_to_on_stderr():
    chunks: list[str] = []

    async def on_stderr(chunk: str) -> None:
        chunks.append(chunk)

    reader, _messages, _errors = _start_reader(max_stderr_bytes=8, on_stderr=on_stderr)
    await reader.read_stderr(b"starting\n")
    await reader.read_stderr(b"ready\n")
    assert chunks == ["starting\n", "ready\n"]
    assert reader.stderr == "g\nready\n"


# lines.test.ts


def test_joins_a_message_split_across_many_chunks_when_its_newline_arrives():
    reader, messages, errors = _start_reader()
    text = "x" * (1024 * 1024)
    data = f"{stringify({'jsonrpc': '2.0', 'id': 1, 'result': {'text': text}})}\n".encode()
    _feed(reader, data, 4096)
    assert errors == []
    assert messages == [{"jsonrpc": "2.0", "id": 1, "result": {"text": text}}]


def test_handles_multi_byte_characters_crlf_blank_lines_and_several_messages_per_chunk_at_any_split():
    data = "\n".join(
        [
            stringify({"jsonrpc": "2.0", "id": 1, "result": {"text": "héllo 🌍"}}),
            "",
            f"{stringify({'jsonrpc': '2.0', 'method': 'notifications/x'})}\r",
            "not json",
            stringify({"jsonrpc": "2.0", "id": 2, "result": {}}),
            "",
        ]
    ).encode()
    for chunk_size in range(1, len(data) + 1):
        reader, messages, errors = _start_reader()
        _feed(reader, data, chunk_size)
        reader.end()
        assert messages == [
            {"jsonrpc": "2.0", "id": 1, "result": {"text": "héllo 🌍"}},
            {"jsonrpc": "2.0", "method": "notifications/x"},
            {"jsonrpc": "2.0", "id": 2, "result": {}},
        ], chunk_size
        assert len(errors) == 1, chunk_size


def test_drops_an_oversized_message_and_keeps_reading_the_next_one():
    reader, messages, errors = _start_reader(max_message_bytes=64)
    following = stringify({"jsonrpc": "2.0", "id": 2, "result": {}})
    _feed(reader, b"a" * 50, 10)
    _feed(reader, f"{'a' * 50}\n{following}\n".encode(), 10)
    assert errors[0] == "MCP stdio message exceeds 64 bytes"
    assert messages == [{"jsonrpc": "2.0", "id": 2, "result": {}}]


def test_reports_a_message_without_its_newline_at_the_end():
    reader, _messages, errors = _start_reader()
    _feed(reader, b'{"jsonrpc":"2.0",', 4)
    reader.end()
    assert errors == ["MCP stdio server closed with an incomplete JSON-RPC message"]
