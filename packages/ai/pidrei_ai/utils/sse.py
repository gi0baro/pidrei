"""Mirror of pi ai src/utils/sse.ts: a server-sent events decoder.

pi parses server-sent events by hand instead of relying on provider SDKs.
Every PiDrei adapter that reads SSE (Anthropic, Mistral, OpenAI Codex, and
the Completions, Responses and Google clients pi reads through SDKs) builds
its event handling on `iterate_sse_messages`, pi's generator of that name in
`anthropic-messages.ts`, over `SseDecoder`.

The incomplete last line is kept in pieces and joined once when it ends: pi
appends it to a string, which is free in JavaScript but a copy in Python.
"""

import codecs
import re
from collections.abc import AsyncGenerator, AsyncIterable
from dataclasses import dataclass

from pidrei_utils.cancel import AbortError, CancelToken


@dataclass(slots=True)
class ServerSentEvent:
    """One server-sent event: its `event` field, its `data` lines joined by
    "\\n", and its raw lines for diagnostics."""

    event: str | None
    data: str
    raw: list[str]


# A bare CR, a bare LF or a CRLF ends a line.
_LINE_BREAK = re.compile(r"\r\n?|\n")


class SseDecoder:
    """Decodes server-sent events from a UTF-8 byte stream. Lines end with
    LF, CRLF or CR, and a blank line ends an event. Only new text is scanned,
    and an incomplete line is joined once when it ends, so a large event
    arriving in many chunks costs time linear in its size. One reader owns a
    decoder."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        # The incomplete last line, in pieces.
        self._pending_line: list[str] = []
        # The last text ended with CR, so a leading LF of the next text completes that CRLF.
        self._skip_leading_lf = False
        self._event: str | None = None
        self._data: list[str] = []
        self._raw: list[str] = []

    def push(self, data: bytes) -> list[ServerSentEvent]:
        """Takes the next bytes and returns the events they complete."""
        return self._decode_text(self._decoder.decode(data))

    def end(self) -> list[ServerSentEvent]:
        """Ends the stream and returns the remaining events, including a last
        one the stream ended without a blank line."""
        events = self._decode_text(self._decoder.decode(b"", final=True))
        if self._pending_line:
            self._decode_line("".join(self._pending_line))
        self._pending_line.clear()
        self._skip_leading_lf = False
        if (last := self._flush()) is not None:
            events.append(last)
        return events

    def _decode_text(self, text: str) -> list[ServerSentEvent]:
        events: list[ServerSentEvent] = []
        start = 1 if self._skip_leading_lf and text.startswith("\n") else 0
        # A chunk holding only part of a UTF-8 sequence decodes to nothing and keeps the flag.
        if text:
            self._skip_leading_lf = False
        while (match := _LINE_BREAK.search(text, start)) is not None:
            line = text[start : match.start()]
            if self._pending_line:
                self._pending_line.append(line)
                line = "".join(self._pending_line)
                self._pending_line.clear()
            if (event := self._decode_line(line)) is not None:
                events.append(event)
            if match.group() == "\r" and match.end() == len(text):
                self._skip_leading_lf = True
            start = match.end()
        if start < len(text):
            self._pending_line.append(text[start:])
        return events

    def _decode_line(self, line: str) -> ServerSentEvent | None:
        if line == "":
            return self._flush()
        self._raw.append(line)
        if line.startswith(":"):
            return None
        colon = line.find(":")
        field = line if colon == -1 else line[:colon]
        value = "" if colon == -1 else line[colon + 1 :]
        value = value.removeprefix(" ")
        if field == "event":
            self._event = value
        elif field == "data":
            self._data.append(value)
        return None

    def _flush(self) -> ServerSentEvent | None:
        if not self._event and not self._data:
            self._event = None
            self._raw = []
            return None
        event = ServerSentEvent(event=self._event, data="\n".join(self._data), raw=self._raw)
        self._event = None
        self._data = []
        self._raw = []
        return event


async def iterate_sse_messages(
    body: AsyncIterable[bytes],
    cancel: CancelToken | None = None,
) -> AsyncGenerator[ServerSentEvent]:
    sse = SseDecoder()
    iterator = aiter(body)

    while True:
        if cancel is not None and cancel.cancelled:
            raise AbortError("Request was aborted")

        try:
            chunk = await anext(iterator)
        except StopAsyncIteration:
            break

        for event in sse.push(chunk):
            yield event

    for event in sse.end():
        yield event
