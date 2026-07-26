"""pidrei-only: the request `@aws-sdk/client-bedrock-runtime` builds for pi.

pi's Bedrock specs mock the SDK away, so nothing upstream covers URL building,
the body/URI split, the build-step ordering, SigV4 signing or the
`vnd.amazon.eventstream` decoding. Here those are ours (`api/bedrock_runtime.py`,
over botocore's sans-io pieces), so they are pinned here.

Frames are encoded by hand in this file — botocore ships a decoder
(`EventStreamBuffer`) but no encoder — which means the assertions check *our*
mapping against *botocore's* framing rather than against another copy of our own.
"""

import binascii
import json
import struct

import pytest

from pidrei_ai.api.bedrock_runtime import (
    BedrockRuntimeClient,
    BedrockRuntimeServiceException,
    ConverseStreamCommand,
    HttpRequest,
    MiddlewareArgs,
    _decode_event,
    _iterate_event_stream,
    _service_exception_from_response,
    _without_model_id,
)


# --- eventstream framing ------------------------------------------------------


def encode_frame(headers: dict[str, str], payload: bytes) -> bytes:
    """One `vnd.amazon.eventstream` frame (all headers as type 7, UTF-8 string)."""
    encoded_headers = b""
    for name, value in headers.items():
        name_bytes = name.encode("utf-8")
        value_bytes = value.encode("utf-8")
        encoded_headers += struct.pack("!B", len(name_bytes)) + name_bytes
        encoded_headers += struct.pack("!B", 7) + struct.pack("!H", len(value_bytes)) + value_bytes

    total_length = 16 + len(encoded_headers) + len(payload)
    prelude = struct.pack("!II", total_length, len(encoded_headers))
    prelude_crc = struct.pack("!I", binascii.crc32(prelude) & 0xFFFFFFFF)
    message = prelude + prelude_crc + encoded_headers + payload
    return message + struct.pack("!I", binascii.crc32(message) & 0xFFFFFFFF)


def event_frame(event_type: str, payload: dict) -> bytes:
    return encode_frame(
        {":message-type": "event", ":event-type": event_type},
        json.dumps(payload).encode("utf-8"),
    )


class _FakeResponse:
    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks

    def iter_bytes(self):
        async def gen():
            for chunk in self._chunks:
                yield chunk

        return gen()

    async def close(self) -> None:
        pass


async def collect(chunks: list[bytes]) -> list[dict]:
    return [event async for event in _iterate_event_stream(_FakeResponse(chunks), None)]


@pytest.mark.tonio
async def test_frames_decode_into_the_sdks_event_union():
    events = await collect(
        [
            event_frame("messageStart", {"role": "assistant"}),
            event_frame("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "hi"}}),
            event_frame("messageStop", {"stopReason": "end_turn"}),
        ]
    )

    assert events == [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "hi"}}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]


@pytest.mark.tonio
async def test_a_frame_split_across_transport_chunks_is_reassembled():
    frame = event_frame("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "split"}})
    events = await collect([frame[:7], frame[7:20], frame[20:]])

    assert events == [{"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "split"}}}]


@pytest.mark.tonio
async def test_a_modelled_stream_exception_is_raised_not_yielded():
    frames = [
        event_frame("messageStart", {"role": "assistant"}),
        event_frame("throttlingException", {"message": "slow down"}),
    ]

    with pytest.raises(BedrockRuntimeServiceException) as excinfo:
        await collect(frames)

    assert excinfo.value.name == "ThrottlingException"
    assert "slow down" in str(excinfo.value)


def test_an_exception_message_type_is_raised_with_its_exception_type():
    frame_headers = {":message-type": "exception", ":exception-type": "ValidationException"}

    class _Event:
        def __init__(self):
            self.headers = frame_headers
            self.payload = json.dumps({"message": "bad input"}).encode()

    with pytest.raises(BedrockRuntimeServiceException) as excinfo:
        _decode_event(_Event())

    assert excinfo.value.name == "ValidationException"
    assert "bad input" in str(excinfo.value)


def test_a_frame_without_an_event_type_is_ignored():
    class _Event:
        def __init__(self):
            self.headers = {":message-type": "event"}
            self.payload = b""

    assert _decode_event(_Event()) is None


# --- request assembly ---------------------------------------------------------


def test_the_model_id_is_a_uri_label_not_a_body_field():
    body = _without_model_id({"modelId": "m", "messages": [], "toolConfig": None})

    assert body == {"messages": []}


def test_the_converse_stream_url_escapes_the_model_id():
    client = BedrockRuntimeClient({"region": "eu-central-1"})

    url = client._url("arn:aws:bedrock:eu-central-1:1:application-inference-profile/x")

    assert url.startswith("https://bedrock-runtime.eu-central-1.amazonaws.com/model/")
    assert url.endswith("/converse-stream")
    assert ":" not in url.split("/model/")[1]


def test_a_custom_endpoint_overrides_the_regional_hostname():
    client = BedrockRuntimeClient({"region": "us-west-2", "endpoint": "https://vpc.example.com/"})

    assert client._url("m").startswith("https://vpc.example.com/model/")


def test_the_region_falls_back_to_us_east_1():
    assert BedrockRuntimeClient({})._url("m").startswith("https://bedrock-runtime.us-east-1.amazonaws.com/")


@pytest.mark.tonio
async def test_build_middleware_runs_in_registration_order():
    client = BedrockRuntimeClient({})
    order: list[str] = []

    def make(tag: str):
        def middleware(next_handler):
            async def handle(args):
                order.append(tag)
                args.request.headers[tag] = "1"
                return await next_handler(args)

            return handle

        return middleware

    client.middleware_stack.add(make("first"), step="build", name="a")
    client.middleware_stack.add(make("second"), step="build", name="b")
    # A non-build step must not run here.
    client.middleware_stack.add(make("finalize"), step="finalizeRequest", name="c")

    args = MiddlewareArgs(request=HttpRequest(method="POST", url="https://x.invalid/", headers={}, body=b""))
    await client.middleware_stack.apply_build(args)

    assert order == ["first", "second"]
    assert sorted(args.request.headers) == ["first", "second"]


@pytest.mark.tonio
async def test_a_bearer_token_replaces_sigv4_signing():
    client = BedrockRuntimeClient({"token": {"token": "secret-token"}, "region": "us-east-1"})
    sent: dict = {}

    async def fake_post(url, *, content, headers, timeout):
        sent["headers"] = headers
        raise _StopSend

    class _StopSend(Exception):
        pass

    from pidrei_ai.utils import http

    original = http.client_for
    http.client_for = lambda *_args, **_kwargs: type("C", (), {"post": staticmethod(fake_post)})()
    try:
        with pytest.raises(_StopSend):
            await client.send(ConverseStreamCommand({"modelId": "m", "messages": []}))
    finally:
        http.client_for = original

    assert sent["headers"]["authorization"] == "Bearer secret-token"
    # No SigV4 headers when the bearer path is taken.
    assert not any(key.lower().startswith("x-amz-") for key in sent["headers"])


# --- errors -------------------------------------------------------------------


def test_a_service_error_takes_its_name_from_the_amzn_errortype_header():
    error = _service_exception_from_response(
        400, {"x-amzn-errortype": "ValidationException:http://internal"}, '{"message": "bad"}'
    )

    assert error.name == "ValidationException"
    assert str(error) == "bad"
    assert error.status == 400


def test_a_service_error_falls_back_to_the_body_type():
    error = _service_exception_from_response(500, {}, '{"__type": "com.amazon#InternalServerException"}')

    assert error.name == "InternalServerException"


def test_a_non_json_error_body_is_kept_verbatim():
    error = _service_exception_from_response(502, {}, "upstream boom")

    assert error.name == "UnknownError"
    assert str(error) == "upstream boom"
    assert error.body == "upstream boom"
