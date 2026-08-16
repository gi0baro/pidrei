"""Mirror of pi's mistral-raw-stop-reason.test.ts.

pi injects a `fetch` returning a canned SSE response; pidrei's transport seam
is client injection (`MistralOptions.client`), fed the same SSE bytes.
"""

import json

import pytest

from pidrei_ai.api.mistral_conversations import MistralOptions, stream as stream_mistral
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import Context, UserMessage
from tests.mistral_helpers import FakeMistralClient, sse_body


MODEL = get_builtin_model("mistral", "devstral-medium-latest")


def _create_client(finish_reason: str) -> FakeMistralClient:
    event = {
        "id": "mistral-response-id",
        "model": MODEL.id,
        "choices": [{"index": 0, "finish_reason": finish_reason, "delta": {}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
    }
    return FakeMistralClient(body=sse_body([json.dumps(event)]))


async def _result(finish_reason: str):
    context = Context(messages=[UserMessage(content="hello", timestamp=1)])
    options = MistralOptions(api_key="test", client=_create_client(finish_reason))
    return await stream_mistral(MODEL, context, options).result()


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
