"""Shared helpers for the anthropic adapter test mirrors.

pi's suites capture the request payload by raising from `onPayload` before any
network I/O happens; the stream then terminates with an error event and the
captured params are asserted on.
"""

import time
from dataclasses import replace

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
