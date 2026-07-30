"""Mirror of pi's mistral-raw-stop-reason.test.ts."""

import contextlib

import pytest

from pidrei_ai.api import mistral_conversations as mistral
from pidrei_ai.api.mistral_conversations import stream as stream_mistral
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import Context, StreamOptions, UserMessage


MODEL = get_builtin_model("mistral", "devstral-medium-latest")
CONTEXT = Context(messages=[UserMessage(content="hello", timestamp=1)])


@contextlib.contextmanager
def _streaming(finish_reason: str):
    class _Fake:
        def __init__(self, _api_key, _server_url=None, env=None):
            pass

        async def chat_stream(self, _payload, *, headers=None, cancel=None, timeout_ms=None):
            async def events():
                yield {
                    "id": "mistral-response-id",
                    "choices": [{"finishReason": finish_reason, "delta": {}}],
                    "usage": {"promptTokens": 1, "completionTokens": 0, "totalTokens": 1},
                }

            return events()

    original = mistral.MistralClient
    mistral.MistralClient = _Fake
    try:
        yield
    finally:
        mistral.MistralClient = original


async def _result(finish_reason: str):
    with _streaming(finish_reason):
        return await stream_mistral(MODEL, CONTEXT, StreamOptions(api_key="test")).result()


@pytest.mark.tonio
async def test_preserves_raw_mistral_finish_reasons_for_successful_stops():
    message = await _result("stop")

    assert message.stop_reason == "stop"
    assert message.raw_stop_reason == "stop"
    assert message.error_message is None


@pytest.mark.tonio
async def test_preserves_raw_mistral_finish_reasons_for_provider_error_stops():
    message = await _result("error")

    assert message.stop_reason == "error"
    assert message.raw_stop_reason == "error"
    assert message.error_message == "Provider stopped with: error"


@pytest.mark.tonio
async def test_treats_unknown_mistral_finish_reasons_as_provider_error_stops():
    message = await _result("unmapped_error")

    assert message.stop_reason == "error"
    assert message.raw_stop_reason == "unmapped_error"
    assert message.error_message == "Provider stopped with: unmapped_error"
