"""Mirror of pi's openai-responses-usage-limit.test.ts.

pi answers the request through `fetch`; here the injected client stands in for
the transport. The rejected request raises what `_PunkreqResponsesClient`
raises for pi's 429 body, the failed stream is served as SSE bytes.
"""

import json

import pytest

from pidrei_ai.api.openai_responses import OpenAIApiError, OpenAIResponsesOptions, stream as stream_responses
from pidrei_ai.types import Context, Model, ModelCost, TextContent, UserMessage


USAGE_LIMIT_ERROR = {"code": "subscription_sharing_usage_limit_exceeded", "message": "Usage limit reached."}

MODEL = Model(
    id="gpt-5-mini",
    name="GPT-5 Mini",
    api="openai-responses",
    provider="openai",
    base_url="https://api.openai.com/v1",
    reasoning=True,
    input=["text"],
    cost=ModelCost(),
    context_window=400_000,
    max_tokens=128_000,
)

CONTEXT = Context(system_prompt="", messages=[UserMessage(content=[TextContent(text="hi")], timestamp=0)], tools=[])


class _SseResponse:
    def __init__(self, body: bytes):
        self.status = 200
        self.headers = {"content-type": "text/event-stream"}
        self._body = body

    async def aiter_bytes(self):
        yield self._body


class _Client:
    def __init__(self, *, error: Exception | None = None, body: bytes = b""):
        self._error = error
        self._body = body

    async def create(self, params, *, timeout_ms, cancel):
        if self._error is not None:
            raise self._error
        return _SseResponse(self._body)


async def get_error_message(client: _Client) -> str | None:
    result = await stream_responses(MODEL, CONTEXT, OpenAIResponsesOptions(api_key="test", client=client)).result()
    assert result.stop_reason == "error"
    return result.error_message


@pytest.mark.tonio
async def test_links_to_chatgpt_usage_when_the_request_is_rejected():
    error = {**USAGE_LIMIT_ERROR, "type": "rate_limit_error"}
    rejected = OpenAIApiError(status=429, headers={}, message=f"429 {error['message']}", error=error)

    error_message = await get_error_message(_Client(error=rejected))

    assert "subscription_sharing_usage_limit_exceeded" in error_message
    assert "Check your ChatGPT usage: https://chatgpt.com/settings/usage" in error_message


@pytest.mark.tonio
async def test_links_to_chatgpt_usage_when_the_stream_fails():
    event = {
        "type": "response.failed",
        "sequence_number": 0,
        "response": {"id": "resp_failed", "status": "failed", "error": USAGE_LIMIT_ERROR},
    }
    body = f"event: response.failed\ndata: {json.dumps(event)}\n\n".encode()

    error_message = await get_error_message(_Client(body=body))

    assert "subscription_sharing_usage_limit_exceeded: Usage limit reached." in error_message
    assert "Check your ChatGPT usage: https://chatgpt.com/settings/usage" in error_message
