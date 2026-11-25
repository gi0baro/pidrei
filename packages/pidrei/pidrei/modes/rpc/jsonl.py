"""Mirror of pi coding-agent src/modes/rpc/jsonl.ts.

Strict JSONL framing. pi attaches a reader to a Node stream; pidrei's
equivalent is an incremental decoder fed byte chunks (the async read loops
live with their callers), plus a helper that pumps a tonio stream.
"""

import codecs
import json
from collections.abc import Callable


def serialize_json_line(value: object) -> str:
    """Serialize a single strict JSONL record.

    Framing is LF-only. Payload strings may contain other Unicode separators
    such as U+2028 and U+2029. Clients must split records on `\\n` only
    (JSON.stringify leaves those separators unescaped; ensure_ascii=False
    matches that).
    """
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


class JsonlLineDecoder:
    """LF-only JSONL splitter over an incremental byte/str feed.

    This intentionally does not split on additional Unicode separators
    (U+2028/U+2029 are valid inside JSON strings); only `\\n` terminates a
    record, and a single trailing `\\r` is stripped (CRLF input).
    """

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        # The unterminated line, in pieces, joined once when its newline
        # arrives (pi appends to a string, which copies here).
        self._pending: list[str] = []

    @staticmethod
    def _emit(line: str) -> str:
        return line.removesuffix("\r")

    def feed(self, chunk: bytes | str) -> list[str]:
        text = chunk if isinstance(chunk, str) else self._decoder.decode(chunk)
        # Only the new text is scanned, so a long line arriving in many chunks is not rescanned per chunk.
        lines: list[str] = []
        start = 0
        while (newline_index := text.find("\n", start)) != -1:
            line = text[start:newline_index]
            if self._pending:
                self._pending.append(line)
                line = "".join(self._pending)
                self._pending.clear()
            start = newline_index + 1
            lines.append(self._emit(line))
        if start < len(text):
            self._pending.append(text[start:])
        return lines

    def end(self) -> list[str]:
        self._pending.append(self._decoder.decode(b"", final=True))
        line = "".join(self._pending)
        self._pending.clear()
        return [self._emit(line)] if line else []


async def pump_jsonl_lines(stream, on_line: Callable[[str], None]) -> None:
    """Read a tonio byte stream to EOF, emitting strict JSONL lines."""
    decoder = JsonlLineDecoder()
    while True:
        chunk = await stream.receive_some()
        if not chunk:
            break
        for line in decoder.feed(chunk):
            on_line(line)
    for line in decoder.end():
        on_line(line)
