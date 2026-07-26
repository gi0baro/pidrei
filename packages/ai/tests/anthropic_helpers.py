"""Shared helpers for the anthropic adapter test mirrors.

pi's suites capture the request payload by raising from `onPayload` before any
network I/O happens; the stream then terminates with an error event and the
captured params are asserted on.

`capture_request` additionally records the request *headers*. pi's fireworks
suite reads them off a real local HTTP server; here the adapter's own transport
class is swapped for a recording subclass, so the headers come from the real
`_create_client` path (including the cache-retention gate on `session_id`)
without a network layer.
"""

import contextlib
import time
from dataclasses import replace

from pidrei_ai.api import anthropic_messages
from pidrei_ai.api.anthropic_messages import stream_simple
from pidrei_ai.types import Context, Model, SimpleStreamOptions, UserMessage


class PayloadCaptured(Exception):
    def __init__(self):
        super().__init__("payload captured")


def now_ms() -> int:
    return int(time.time() * 1000)


def make_context() -> Context:
    return Context(messages=[UserMessage(content="Hello", timestamp=now_ms())])


async def capture_payload(
    model: Model,
    options: SimpleStreamOptions | None = None,
    context: Context | None = None,
) -> dict:
    captured: list[dict] = []

    def on_payload(payload, _model):
        captured.append(payload)
        raise PayloadCaptured()

    payload_capture_model = replace(model, base_url="http://127.0.0.1:9")
    opts = replace(options if options is not None else SimpleStreamOptions(), api_key="fake-key", on_payload=on_payload)

    await stream_simple(payload_capture_model, context if context is not None else make_context(), opts).result()

    if not captured:
        raise AssertionError("Expected payload to be captured before request failure")
    return captured[0]


@contextlib.contextmanager
def _recording_transport():
    """Record the headers the adapter builds (no yield fixtures: tonio)."""
    recorded: list[dict[str, str]] = []
    original = anthropic_messages._PunkreqAnthropicClient

    class RecordingClient(original):
        def __init__(self, base_url: str, headers: dict[str, str], *args, **kwargs):
            recorded.append(dict(headers))
            super().__init__(base_url, headers, *args, **kwargs)

    anthropic_messages._PunkreqAnthropicClient = RecordingClient
    try:
        yield recorded
    finally:
        anthropic_messages._PunkreqAnthropicClient = original


async def capture_request(
    model: Model,
    options: SimpleStreamOptions | None = None,
    context: Context | None = None,
) -> tuple[dict[str, str], dict]:
    """The (headers, payload) the adapter would put on the wire."""
    with _recording_transport() as recorded:
        payload = await capture_payload(model, options, context)
    if not recorded:
        raise AssertionError("Expected the adapter to build a transport")
    return recorded[0], payload
