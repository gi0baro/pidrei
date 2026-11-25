"""Mirror of pi's google-stream-errors.test.ts."""

import json

import pytest

from pidrei_ai.api.google_generative_ai import stream_simple as stream_google_generative_ai
from pidrei_ai.api.google_vertex import stream_simple as stream_google_vertex
from pidrei_ai.types import Context, Model, ModelCost, SimpleStreamOptions, UserMessage
from pidrei_ai.utils.retry import is_retryable_assistant_error
from pidrei_ai.utils.transcript import normalize_context
from pidrei_http import http


CONTEXT = normalize_context(Context(messages=[UserMessage(content="hello", timestamp=1)]))


def create_model(api: str) -> Model:
    return Model(
        id="test-model",
        name="Test Model",
        api=api,
        provider="test-provider",
        base_url="https://upstream.test/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(),
        context_window=10_000,
        max_tokens=1_000,
    )


def create_fetch(body: str):
    async def fetch(request, *, env=None):
        return http.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode(), request=request)

    return fetch


SAFETY_RATINGS = [{"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "probability": "HIGH"}]

ADAPTERS = [
    pytest.param(
        {"blockReason": "PROHIBITED_CONTENT", "safetyRatings": SAFETY_RATINGS},
        "Google prompt blocked: PROHIBITED_CONTENT. Safety rating: HARM_CATEGORY_SEXUALLY_EXPLICIT (HIGH)",
        "Google stream ended without a finish reason after receiving 0 chunks",
        lambda fetch: stream_google_generative_ai(
            create_model("google-generative-ai"), CONTEXT, SimpleStreamOptions(api_key="test-key", fetch=fetch)
        ),
        id="Google Generative AI",
    ),
    pytest.param(
        {
            "blockReason": "PROHIBITED_CONTENT",
            "blockReasonMessage": "Blocked by policy",
            "safetyRatings": SAFETY_RATINGS,
        },
        (
            "Google Vertex prompt blocked: PROHIBITED_CONTENT. Blocked by policy. Safety rating: "
            "HARM_CATEGORY_SEXUALLY_EXPLICIT (HIGH)"
        ),
        "Google Vertex stream ended without a finish reason after receiving 0 chunks",
        lambda fetch: stream_google_vertex(
            create_model("google-vertex"), CONTEXT, SimpleStreamOptions(api_key="test-key", fetch=fetch)
        ),
        id="Google Vertex",
    ),
]


# Regression for #12330.
@pytest.mark.tonio
@pytest.mark.parametrize(("feedback", "expected_block_error", "expected_empty_error", "create_stream"), ADAPTERS)
async def test_distinguishes_a_blocked_prompt_from_an_empty_stream(
    feedback, expected_block_error, expected_empty_error, create_stream
):
    blocked = await create_stream(create_fetch(f"data: {json.dumps({'promptFeedback': feedback})}\n\n")).result()
    empty = await create_stream(create_fetch("")).result()

    assert blocked.stop_reason == "error"
    assert blocked.error_message == expected_block_error
    assert is_retryable_assistant_error(blocked) is False
    assert empty.stop_reason == "error"
    assert empty.error_message == expected_empty_error
    assert is_retryable_assistant_error(empty) is True
