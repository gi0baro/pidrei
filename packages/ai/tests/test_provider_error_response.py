"""Mirror of pi's provider-error-response.test.ts.

#12376: adapters whose client raises on a non-2xx response report HTTP error
responses to `on_response`, like the adapters that read the response themselves.

pi runs a local HTTP server answering 429; here every attempt goes through the
`fetch` request option, which answers with the same 429 response, so no server
crosses a run boundary. The Google case pays one real backoff (its error
carries no `retry-after`), as pi's does.
"""

import json

import pytest

from pidrei_ai.api import (
    anthropic_messages,
    azure_openai_responses,
    google_generative_ai,
    openai_completions,
    openai_responses,
)
from pidrei_ai.types import Context, Model, ModelCost, ProviderResponse, UserMessage
from pidrei_ai.utils.transcript import normalize_context
from pidrei_http import http


CONTEXT = normalize_context(Context(messages=[UserMessage(content="hi", timestamp=0)]))


class RateLimitFetch:
    """Answers every request with 429 and `retry-after: 0`, counting requests."""

    def __init__(self):
        self.requests = 0

    async def __call__(self, request, *, env=None):
        self.requests += 1
        return http.Response(
            429,
            headers={"content-type": "application/json", "retry-after": "0", "x-request-id": "req-1"},
            content=json.dumps(
                {"error": {"message": "rate limited", "type": "rate_limit_error", "code": 429}}
            ).encode(),
            request=request,
        )


def make_model(api: str) -> Model:
    return Model(
        id="test-model",
        name="Test Model",
        api=api,
        provider="test",
        base_url="http://127.0.0.1:9",
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=1000,
        max_tokens=100,
    )


ADAPTERS = [
    ("openai-completions", openai_completions.stream, openai_completions.OpenAICompletionsOptions, True),
    ("openai-responses", openai_responses.stream, openai_responses.OpenAIResponsesOptions, True),
    (
        "azure-openai-responses",
        azure_openai_responses.stream,
        azure_openai_responses.AzureOpenAIResponsesOptions,
        True,
    ),
    ("anthropic-messages", anthropic_messages.stream, anthropic_messages.AnthropicOptions, True),
    # The Google client's error carries no headers, as @google/genai's ApiError.
    ("google-generative-ai", google_generative_ai.stream, google_generative_ai.GoogleOptions, False),
]


@pytest.mark.tonio
@pytest.mark.parametrize(("api", "stream", "options_type", "headers"), ADAPTERS, ids=[entry[0] for entry in ADAPTERS])
async def test_reports_each_failed_attempt_to_on_response(api, stream, options_type, headers):
    fetch = RateLimitFetch()
    responses: list[ProviderResponse] = []

    async def on_response(response, _model):
        responses.append(response)

    result = await stream(
        make_model(api),
        CONTEXT,
        options_type(api_key="test-key", max_retries=1, fetch=fetch, on_response=on_response),
    ).result()

    assert result.stop_reason == "error"
    assert fetch.requests == 2
    assert [response.status for response in responses] == [429, 429]
    if headers:
        assert responses[0].headers["retry-after"] == "0"
        assert responses[0].headers["x-request-id"] == "req-1"
    else:
        assert responses[0].headers == {}
