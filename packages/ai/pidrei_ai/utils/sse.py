"""Port of pi's SSE decoder (packages/ai/src/api/anthropic-messages.ts:296-441).

pi parses server-sent events by hand instead of relying on provider SDKs; this
module is the behavioral mirror of that decoder, including its exact line
splitting (bare CR, bare LF, and CRLF all break lines) and its trailing-buffer
flush at end of stream. Adapters build provider-specific event handling on top
of `iterate_sse_messages`.
"""

import codecs
import re
from collections.abc import AsyncGenerator, AsyncIterable
from dataclasses import dataclass, field

from pidrei_utils.cancel import AbortError, CancelToken


@dataclass(slots=True)
class ServerSentEvent:
    event: str | None
    data: str
    raw: list[str]


@dataclass(slots=True)
class _SseDecoderState:
    event: str | None = None
    data: list[str] = field(default_factory=list)
    raw: list[str] = field(default_factory=list)


def _flush_sse_event(state: _SseDecoderState) -> ServerSentEvent | None:
    if not state.event and not state.data:
        return None

    event = ServerSentEvent(event=state.event, data="\n".join(state.data), raw=list(state.raw))
    state.event = None
    state.data = []
    state.raw = []
    return event


def _decode_sse_line(line: str, state: _SseDecoderState) -> ServerSentEvent | None:
    if line == "":
        return _flush_sse_event(state)

    state.raw.append(line)
    if line.startswith(":"):
        return None

    delimiter_index = line.find(":")
    field_name = line if delimiter_index == -1 else line[:delimiter_index]
    value = "" if delimiter_index == -1 else line[delimiter_index + 1 :]
    value = value.removeprefix(" ")

    if field_name == "event":
        state.event = value
    elif field_name == "data":
        state.data.append(value)

    return None


# A bare CR, a bare LF or a CRLF ends a line. A CR that ends the text so far ends
# its line alone, so a CRLF split across chunks reads as two breaks, as in pi.
_LINE_BREAK = re.compile(r"\r\n?|\n")


def _consume_lines(text: str, pending: list[str]) -> list[str]:
    """The lines `text` completes, the first one continuing `pending`; the
    unterminated rest goes to `pending`.

    Lines are cut by searching from an offset and the unterminated line is
    kept in pieces, so a chunk costs its own length: pi's decoder slices the
    buffer per line, which is cheap in JavaScript but copies in Python.
    """
    lines: list[str] = []
    position = 0
    while (match := _LINE_BREAK.search(text, position)) is not None:
        line = text[position : match.start()]
        if pending:
            pending.append(line)
            line = "".join(pending)
            pending.clear()
        lines.append(line)
        position = match.end()
    if position < len(text):
        pending.append(text[position:])
    return lines


async def iterate_sse_messages(
    body: AsyncIterable[bytes],
    cancel: CancelToken | None = None,
) -> AsyncGenerator[ServerSentEvent]:
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    state = _SseDecoderState()
    # The unterminated line, in pieces.
    pending: list[str] = []
    iterator = aiter(body)

    while True:
        if cancel is not None and cancel.cancelled:
            raise AbortError("Request was aborted")

        try:
            chunk = await anext(iterator)
        except StopAsyncIteration:
            break

        for line in _consume_lines(decoder.decode(chunk), pending):
            if (event := _decode_sse_line(line, state)) is not None:
                yield event

    for line in _consume_lines(decoder.decode(b"", final=True), pending):
        if (event := _decode_sse_line(line, state)) is not None:
            yield event

    if pending and (event := _decode_sse_line("".join(pending), state)) is not None:
        yield event

    if (trailing := _flush_sse_event(state)) is not None:
        yield trailing
