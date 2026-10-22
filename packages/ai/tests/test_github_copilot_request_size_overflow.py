"""Mirror of pi's github-copilot-request-size-overflow.test.ts.

pi answers the request through a stubbed `fetch` and lets the OpenAI SDK turn
the response into the error; here the stub is a `FetchFunction` returning a
synthetic `http.Response`, and `_PunkreqResponsesClient` builds the error.
"""

import pytest

from pidrei_ai.api.openai_responses import OpenAIResponsesOptions, stream
from pidrei_ai.types import Context, Model, ModelCost, UserMessage
from pidrei_ai.utils.overflow import is_context_overflow
from pidrei_ai.utils.transcript import normalize_context
from pidrei_http import http


MODEL = Model(
    id="copilot-test-model",
    name="Copilot test model",
    api="openai-responses",
    provider="github-copilot",
    base_url="https://copilot.invalid",
    reasoning=True,
    input=["text", "image"],
    cost=ModelCost(),
    context_window=1_000_000,
    max_tokens=128_000,
)


@pytest.mark.tonio
@pytest.mark.parametrize(
    ("status", "nested", "overflow"),
    [(413, True, True), (413, False, True), (400, True, False)],
)
async def test_classifies_http_status_through_the_real_client(status, nested, overflow):
    error = {"message": "failed to parse request", "code": ""}
    requests: list[http.Request] = []

    async def fetch(request, *, env=None):
        requests.append(request)
        return http.Response(
            status,
            headers={"content-type": "application/json"},
            json={"error": error} if nested else error,
            request=request,
        )

    result = await stream(
        MODEL,
        normalize_context(Context(messages=[UserMessage(content="Continue", timestamp=0)])),
        OpenAIResponsesOptions(api_key="test-key", fetch=fetch, max_retries=0),
    ).result()

    assert len(requests) == 1
    assert result.provider == "github-copilot"
    assert result.stop_reason == "error"
    assert result.usage.total_tokens == 0
    assert f"({status})" in result.error_message
    assert "failed to parse request" in result.error_message
    assert is_context_overflow(result, MODEL.context_window) is overflow
