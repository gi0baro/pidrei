"""Port of pi's SSE decoder (packages/ai/src/api/anthropic-messages.ts:296-441).

pi parses server-sent events by hand instead of relying on provider SDKs; this
module is the behavioral mirror of that decoder, including its exact line
splitting (bare CR, bare LF, and CRLF all break lines) and its trailing-buffer
flush at end of stream. Adapters build provider-specific event handling on top
of `iterate_sse_messages`.
"""

import codecs
from collections.abc import AsyncGenerator, AsyncIterable
from dataclasses import dataclass, field

from pppi_ai.utils.cancel import AbortError, CancelToken


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


def _next_line_break_index(text: str) -> int:
    carriage_return_index = text.find("\r")
    newline_index = text.find("\n")
    if carriage_return_index == -1:
        return newline_index
    if newline_index == -1:
        return carriage_return_index
    return min(carriage_return_index, newline_index)


def _consume_line(text: str) -> tuple[str, str] | None:
    line_break_index = _next_line_break_index(text)
    if line_break_index == -1:
        return None

    next_index = line_break_index + 1
    if text[line_break_index] == "\r" and next_index < len(text) and text[next_index] == "\n":
        next_index += 1

    return text[:line_break_index], text[next_index:]


async def iterate_sse_messages(
    body: AsyncIterable[bytes],
    cancel: CancelToken | None = None,
) -> AsyncGenerator[ServerSentEvent]:
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    state = _SseDecoderState()
    buffer = ""
    iterator = aiter(body)

    while True:
        if cancel is not None and cancel.cancelled:
            raise AbortError("Request was aborted")

        try:
            chunk = await anext(iterator)
        except StopAsyncIteration:
            break

        buffer += decoder.decode(chunk)
        while (consumed := _consume_line(buffer)) is not None:
            line, buffer = consumed
            if (event := _decode_sse_line(line, state)) is not None:
                yield event

    buffer += decoder.decode(b"", final=True)
    while (consumed := _consume_line(buffer)) is not None:
        line, buffer = consumed
        if (event := _decode_sse_line(line, state)) is not None:
            yield event

    if buffer and (event := _decode_sse_line(buffer, state)) is not None:
        yield event

    if (trailing := _flush_sse_event(state)) is not None:
        yield trailing
