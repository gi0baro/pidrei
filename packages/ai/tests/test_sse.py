"""Mirror of pi ai test/sse.test.ts (the `SseDecoder` cases), with PiDrei's
own cases for `iterate_sse_messages` around them."""

import math
import time

import pytest

from pidrei_ai.utils.sse import ServerSentEvent, SseDecoder, iterate_sse_messages
from pidrei_utils.cancel import AbortError, CancelToken


async def _chunks(*parts: bytes):
    for part in parts:
        yield part


async def _collect(*parts: bytes, cancel=None) -> list[ServerSentEvent]:
    return [event async for event in iterate_sse_messages(_chunks(*parts), cancel)]


@pytest.mark.tonio
async def test_basic_event():
    events = await _collect(b'event: message_start\ndata: {"a":1}\n\n')

    assert events == [
        ServerSentEvent(event="message_start", data='{"a":1}', raw=["event: message_start", 'data: {"a":1}'])
    ]


@pytest.mark.tonio
async def test_multiple_data_lines_joined_with_newline():
    events = await _collect(b"data: first\ndata: second\n\n")

    assert len(events) == 1
    assert events[0].event is None
    assert events[0].data == "first\nsecond"


@pytest.mark.tonio
async def test_comment_lines_are_ignored_but_kept_in_raw():
    events = await _collect(b": keep-alive\ndata: x\n\n")

    assert len(events) == 1
    assert events[0].data == "x"
    assert events[0].raw == [": keep-alive", "data: x"]


@pytest.mark.tonio
async def test_field_without_colon_and_space_stripping():
    # A bare field name has an empty value; only a single leading space is stripped.
    events = await _collect(b"data\n\n", b"data:  two spaces\n\n")

    assert events[0].data == ""
    assert events[1].data == " two spaces"


@pytest.mark.tonio
async def test_line_breaks_cr_lf_crlf_equivalent():
    lf = await _collect(b"event: e\ndata: 1\n\n")
    cr = await _collect(b"event: e\rdata: 1\r\r")
    crlf = await _collect(b"event: e\r\ndata: 1\r\n\r\n")

    assert lf == cr == crlf


@pytest.mark.tonio
async def test_chunk_boundaries_do_not_change_events():
    whole = await _collect(b"event: e\r\ndata: hello\r\n\r\ndata: tail\r\n\r\n")
    split = await _collect(b"event: e\r", b"\ndata: hel", b"lo\r\n", b"\r\ndata: tail\r\n\r\n")

    assert whole == split


def _decode(data: bytes, chunk_size: int) -> list[ServerSentEvent]:
    decoder = SseDecoder()
    events: list[ServerSentEvent] = []
    for index in range(0, len(data), chunk_size):
        events.extend(decoder.push(data[index : index + chunk_size]))
    events.extend(decoder.end())
    return events


def test_gives_the_same_events_for_every_chunking_and_line_ending():
    for separator in ("\n", "\r\n", "\r"):
        text = separator.join([": keepalive", "event: delta", 'data: {"a":', 'data:  "é"}', "", "data: tail"])
        data = text.encode()
        for chunk_size in range(1, len(data) + 1):
            assert _decode(data, chunk_size) == [
                ServerSentEvent(
                    event="delta",
                    data='{"a":\n "é"}',
                    raw=[": keepalive", "event: delta", 'data: {"a":', 'data:  "é"}'],
                ),
                ServerSentEvent(event=None, data="tail", raw=["data: tail"]),
            ], (separator, chunk_size)


def test_does_not_end_an_event_at_a_crlf_split_across_chunks():
    decoder = SseDecoder()
    assert decoder.push(b"data: a\r") == []
    assert decoder.push(b"\ndata: b\r\n\r\n") == [ServerSentEvent(event=None, data="a\nb", raw=["data: a", "data: b"])]


def test_does_not_carry_an_empty_event_name_into_the_next_event():
    decoder = SseDecoder()
    assert decoder.push(b"event:\n\ndata: x\n\n") == [ServerSentEvent(event=None, data="x", raw=["data: x"])]


def test_drops_comment_lines_of_events_without_data():
    decoder = SseDecoder()
    assert decoder.push(b": ping\n\ndata: x\n\n") == [ServerSentEvent(event=None, data="x", raw=["data: x"])]


@pytest.mark.tonio
async def test_trailing_event_without_final_blank_line_is_flushed():
    events = await _collect(b"data: incomplete")

    assert len(events) == 1
    assert events[0].data == "incomplete"


@pytest.mark.tonio
async def test_utf8_split_across_chunks():
    encoded = "data: caffè\n\n".encode()
    events = await _collect(encoded[:8], encoded[8:])

    assert events[0].data == "caffè"


@pytest.mark.tonio
async def test_event_without_explicit_type():
    events = await _collect(b"data: [DONE]\n\n")

    assert events[0].event is None
    assert events[0].data == "[DONE]"


async def _decode_seconds(*parts: bytes) -> float:
    best = math.inf
    for _ in range(2):
        started = time.perf_counter()
        await _collect(*parts)
        best = min(best, time.perf_counter() - started)
    return best


@pytest.mark.tonio
async def test_reads_a_large_chunk_and_a_long_line_in_time_linear_in_their_length():
    # A buffered response arrives as one chunk; slicing the rest off per line copied it whole each time.
    def events(count: int) -> bytes:
        return b"".join(b'event: e\ndata: {"delta":"%d"}\n\n' % index for index in range(count))

    assert len(await _collect(events(50_000))) == 50_000
    assert await _decode_seconds(events(50_000)) / await _decode_seconds(events(5_000)) < 20

    # One long data line arriving in many small chunks: appending each chunk to the line copied it whole.
    def line(size: int) -> list[bytes]:
        text = b"data: " + b"x" * size + b"\n\n"
        return [text[index : index + 64] for index in range(0, len(text), 64)]

    (event,) = await _collect(*line(200_000))
    assert len(event.data) == 200_000
    assert await _decode_seconds(*line(2_000_000)) / await _decode_seconds(*line(200_000)) < 20


@pytest.mark.tonio
async def test_cancelled_token_aborts_iteration():
    cancel = CancelToken()
    cancel.cancel()

    with pytest.raises(AbortError):
        await _collect(b"data: x\n\n", cancel=cancel)
