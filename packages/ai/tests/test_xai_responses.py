"""Mirror of pi's xai-responses.test.ts.

pi mocks global `fetch` and reads the outgoing `Request`. pidrei inspects the
same two halves directly: the request body through `xai_provider().stream(...)`
with an injected client (so the provider's narrowing to the Responses API is
exercised, not just the adapter), and the wire headers through
`_create_client(...)`, the pattern test_openai_responses_compat.py established.

DEVIATION: pi's "keeps the SDK User-Agent for non-xAI Responses requests" reads
the User-Agent the OpenAI SDK sets. pidrei's transport has no per-request SDK
header — punkreq sends its own — so the mirror asserts the adapter leaves the
header alone for non-xAI providers instead.
"""

import pytest

from pidrei_ai.api.openai_completions import (
    _create_client as create_completions_client,
    get_compat,
)
from pidrei_ai.api.openai_responses import (
    OpenAIResponsesOptions,
    _create_client as create_responses_client,
)
from pidrei_ai.providers.all import get_builtin_model, get_builtin_models
from pidrei_ai.providers.xai import xai_provider
from pidrei_ai.registry import get_supported_thinking_levels
from pidrei_ai.types import Context, Model, ModelCost, UserMessage
from pidrei_ai.utils.user_agent import get_user_agent
from tests.test_openai_responses import FakeClient, make_model


COMPLETED_EVENTS = [
    {
        "type": "response.completed",
        "sequence_number": 0,
        "response": {
            "id": "resp_xai_test",
            "status": "completed",
            "output": [],
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
                "input_tokens_details": {"cached_tokens": 0},
            },
        },
    }
]


def xai_model(model_id: str) -> Model:
    model = get_builtin_model("xai", model_id)
    assert model is not None, model_id
    return model


async def capture_request(model: Model, context: Context, options: OpenAIResponsesOptions) -> dict:
    """The params the provider puts on the wire for `model`."""
    client = FakeClient(COMPLETED_EVENTS)
    options.client = client
    result = await xai_provider().stream(model, context, options).result()
    assert result.stop_reason == "stop", result.error_message
    assert len(client.requests) == 1
    return client.requests[0]


def test_excludes_retired_and_redundant_models_from_the_builtin_catalog():
    ids = [model.id for model in get_builtin_models("xai")]
    for model_id in (
        "grok-3",
        "grok-3-fast",
        "grok-4.20-0309-non-reasoning",
        "grok-4.20-0309-reasoning",
        "grok-code-fast-1",
    ):
        assert model_id not in ids


@pytest.mark.skip(reason="catalog regen pending — unskip after `make models-data` (PORT_0.84.3 U10)")
def test_routes_every_builtin_xai_model_through_responses():
    for model in get_builtin_models("xai"):
        assert model.api == "openai-responses", model.id
    assert get_supported_thinking_levels(xai_model("grok-4.5")) == ["low", "medium", "high"]
    assert get_supported_thinking_levels(xai_model("grok-4.6")) == ["low", "medium", "high", "xhigh"]
    assert get_supported_thinking_levels(xai_model("grok-4.3")) == ["off", "low", "medium", "high"]
    assert get_supported_thinking_levels(xai_model("grok-build-0.1")) == ["low", "medium", "high"]


@pytest.mark.tonio
async def test_uses_responses_with_bearer_auth_and_xai_compatible_request_fields():
    model = xai_model("grok-4.5")
    context = Context(
        system_prompt="You are a careful coding assistant.",
        messages=[UserMessage(content="hello", timestamp=1)],
    )
    params = await capture_request(
        model,
        context,
        OpenAIResponsesOptions(
            api_key="xai-test-token",
            session_id="pi-session-123",
            cache_retention="long",
            reasoning_effort="medium",
        ),
    )

    client = create_responses_client(model, context, "xai-test-token", None, "pi-session-123")
    assert client._url == "https://api.x.ai/v1/responses"
    assert client._headers["authorization"] == "Bearer xai-test-token"
    assert client._headers["User-Agent"] == get_user_agent()
    assert client._headers["session_id"] == "pi-session-123"

    assert params["model"] == "grok-4.5"
    assert params["store"] is False
    assert params["stream"] is True
    assert params["prompt_cache_key"] == "pi-session-123"
    assert params["reasoning"]["effort"] == "medium"
    assert params["include"] == ["reasoning.encrypted_content"]
    assert "prompt_cache_retention" not in params
    assert any(
        item.get("role") == "developer" and item.get("content") == "You are a careful coding assistant."
        for item in params["input"]
    )


@pytest.mark.tonio
async def test_requests_encrypted_reasoning_without_an_effort_override():
    params = await capture_request(
        xai_model("grok-4.5"),
        Context(messages=[UserMessage(content="hello", timestamp=1)]),
        OpenAIResponsesOptions(api_key="xai-test-token"),
    )

    assert params["model"] == "grok-4.5"
    assert params["store"] is False
    assert params["include"] == ["reasoning.encrypted_content"]
    assert "reasoning" not in params


@pytest.mark.tonio
async def test_uses_responses_for_grok_46_with_xhigh_effort_and_encrypted_reasoning():
    model = xai_model("grok-4.6")
    context = Context(
        system_prompt="You are a careful coding assistant.",
        messages=[UserMessage(content="hello", timestamp=1)],
    )
    params = await capture_request(
        model,
        context,
        OpenAIResponsesOptions(api_key="xai-test-token", reasoning_effort="xhigh"),
    )

    assert create_responses_client(model, context, "xai-test-token", None, None)._url == (
        "https://api.x.ai/v1/responses"
    )
    assert params["model"] == "grok-4.6"
    assert params["store"] is False
    assert params["stream"] is True
    assert params["reasoning"]["effort"] == "xhigh"
    assert params["include"] == ["reasoning.encrypted_content"]


@pytest.mark.tonio
async def test_uses_responses_for_grok_43():
    model = xai_model("grok-4.3")
    context = Context(messages=[UserMessage(content="hello", timestamp=1)])
    params = await capture_request(
        model,
        context,
        OpenAIResponsesOptions(api_key="xai-test-token", reasoning_effort="low"),
    )

    assert create_responses_client(model, context, "xai-test-token", None, None)._url == (
        "https://api.x.ai/v1/responses"
    )
    assert params["model"] == "grok-4.3"
    assert params["store"] is False
    assert params["include"] == ["reasoning.encrypted_content"]
    assert params["reasoning"]["effort"] == "low"


CUSTOM_COMPLETIONS_MODEL = Model(
    id="grok-custom",
    name="Grok Custom",
    api="openai-completions",
    provider="xai",
    base_url="https://api.x.ai/v1",
    reasoning=False,
    input=["text"],
    cost=ModelCost(),
    context_window=128000,
    max_tokens=16384,
)


def completions_user_agent(headers: dict | None = None) -> str:
    context = Context(messages=[UserMessage(content="hello", timestamp=1)])
    client_headers = create_completions_client(
        CUSTOM_COMPLETIONS_MODEL,
        context,
        "xai-test-token",
        headers,
        None,
        get_compat(CUSTOM_COMPLETIONS_MODEL),
    )._headers
    return client_headers["User-Agent"]


def responses_user_agent(headers: dict | None = None) -> str:
    model = make_model(provider="openai", base_url="https://api.openai.com/v1")
    context = Context(messages=[UserMessage(content="hello", timestamp=1)])
    return create_responses_client(model, context, "test-token", headers, None)._headers["User-Agent"]


def test_uses_the_runtime_user_agent_by_default_for_responses_requests():
    assert responses_user_agent() == get_user_agent()


def test_lets_explicit_headers_override_the_default_responses_user_agent():
    assert responses_user_agent({"User-Agent": "custom-agent"}) == "custom-agent"


def test_uses_the_runtime_user_agent_by_default_for_completions_requests():
    assert completions_user_agent() == get_user_agent()


def test_lets_explicit_headers_override_the_default_completions_user_agent():
    assert completions_user_agent({"User-Agent": "custom-agent"}) == "custom-agent"
