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
from pidrei_ai.api.anthropic_messages import AnthropicOptions, stream, stream_simple
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
    options: SimpleStreamOptions | AnthropicOptions | None = None,
    context: Context | None = None,
    *,
    default_api_key: str | None = "fake-key",
) -> dict:
    """The payload the adapter would send.

    `AnthropicOptions` goes through `stream`, everything else through
    `stream_simple` — the same split pi's suites make. A caller-supplied
    `api_key` is kept (the OAuth and Copilot header branches depend on its
    shape); otherwise `default_api_key` stands in — pass `None` to exercise
    header-owned auth with no key at all.
    """
    captured: list[dict] = []

    async def on_payload(payload, _model):
        captured.append(payload)
        raise PayloadCaptured()

    payload_capture_model = replace(model, base_url="http://127.0.0.1:9")
    given = options if options is not None else SimpleStreamOptions()
    opts = replace(given, api_key=given.api_key or default_api_key, on_payload=on_payload)
    run = stream if isinstance(opts, AnthropicOptions) else stream_simple

    await run(payload_capture_model, context if context is not None else make_context(), opts).result()

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
    options: SimpleStreamOptions | AnthropicOptions | None = None,
    context: Context | None = None,
    *,
    default_api_key: str | None = "fake-key",
) -> tuple[dict[str, str], dict]:
    """The (headers, payload) the adapter would put on the wire."""
    with _recording_transport() as recorded:
        payload = await capture_payload(model, options, context, default_api_key=default_api_key)
    if not recorded:
        raise AssertionError("Expected the adapter to build a transport")
    return recorded[0], payload
