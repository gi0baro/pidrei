import pytest

from pidrei_ai.utils.cancel import AbortError, CancelToken
from pidrei_ai.utils.sse import ServerSentEvent, iterate_sse_messages


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
    # Splits that keep CRLF pairs intact are invariant.
    whole = await _collect(b"event: e\r\ndata: hello\r\n\r\ndata: tail\r\n\r\n")
    split = await _collect(b"event: e\r\ndata: hel", b"lo\r\n", b"\r\ndata: tail\r\n\r\n")

    assert whole == split


@pytest.mark.tonio
async def test_crlf_split_across_chunks_flushes_early():
    # Mirror of a pi decoder quirk: a chunk-final CR is consumed as a complete
    # line break (the decoder cannot see the LF still in flight), so the LF
    # arriving in the next chunk reads as an empty line and flushes the
    # pending event early.
    events = await _collect(b"event: e\r", b"\ndata: x\r\n\r\n")

    assert [(event.event, event.data) for event in events] == [("e", ""), (None, "x")]


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


@pytest.mark.tonio
async def test_cancelled_token_aborts_iteration():
    cancel = CancelToken()
    cancel.cancel()

    with pytest.raises(AbortError):
        await _collect(b"data: x\n\n", cancel=cancel)
