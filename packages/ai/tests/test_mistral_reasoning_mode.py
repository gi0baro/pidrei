"""Mirror of pi's mistral-reasoning-mode.test.ts.

pi captures the payload by pointing the model at a dead port and reading
`onPayload` before the request fails. Same capture point here, without
depending on a connection refusal: `on_payload` raises before any transport.

The payload stays camelCase, as pi's does: the snake_case rename happens at
the transport boundary (`to_mistral_wire_payload`).
"""

import pytest

from pidrei_ai.api import mistral_conversations as mistral
from pidrei_ai.api.mistral_conversations import stream_simple as stream_simple_mistral
from pidrei_ai.types import Context, Model, ModelCost, SimpleStreamOptions, UserMessage


captured: list[dict] = []


class _PayloadCaptured(Exception):
    pass


async def _capturing_on_payload(payload, _model):
    captured.append(payload)
    raise _PayloadCaptured("payload captured")


@pytest.fixture(autouse=True)
def _reset():
    captured.clear()


def make_model(model_id: str, reasoning: bool) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="mistral-conversations",
        provider="mistral",
        base_url="http://127.0.0.1:9",
        reasoning=reasoning,
        input=["text"],
        cost=ModelCost(),
        context_window=128000,
        max_tokens=16384,
    )


def make_context() -> Context:
    return Context(messages=[UserMessage(content="Hello", timestamp=1)])


async def capture_payload(model, options: SimpleStreamOptions | None = None) -> dict:
    opts = options or SimpleStreamOptions()
    opts.api_key = "fake-key"
    opts.on_payload = _capturing_on_payload
    await stream_simple_mistral(model, make_context(), opts).result()
    assert captured, "Expected payload to be captured before request failure"
    return captured[0]


@pytest.mark.tonio
async def test_uses_reasoning_effort_for_mistral_small_4():
    payload = await capture_payload(make_model("mistral-small-2603", True), SimpleStreamOptions(reasoning="medium"))

    assert payload["reasoningEffort"] == "high"
    assert "promptMode" not in payload


@pytest.mark.tonio
async def test_omits_reasoning_controls_for_mistral_small_4_when_thinking_is_off():
    payload = await capture_payload(make_model("mistral-small-2603", True))

    assert "reasoningEffort" not in payload
    assert "promptMode" not in payload


@pytest.mark.tonio
async def test_uses_prompt_mode_for_magistral_reasoning_models():
    payload = await capture_payload(
        make_model("magistral-medium-latest", True), SimpleStreamOptions(reasoning="medium")
    )

    assert payload["promptMode"] == "reasoning"
    assert "reasoningEffort" not in payload


# Regression for #9375: Mistral-hosted GLM-5.2 ignores prompt_mode.
@pytest.mark.tonio
async def test_zai_glm_5_2_uses_reasoning_effort_when_thinking_is_enabled():
    payload = await capture_payload(make_model("zai-glm-5-2", True), SimpleStreamOptions(reasoning="medium"))

    assert payload["reasoningEffort"] == "high"
    assert "promptMode" not in payload


@pytest.mark.tonio
async def test_zai_glm_5_2_omits_reasoning_controls_when_thinking_is_off():
    payload = await capture_payload(make_model("zai-glm-5-2", True))

    assert "reasoningEffort" not in payload
    assert "promptMode" not in payload


# Regression for #8700: Medium aliases must use reasoning_effort, not Magistral's prompt_mode.
@pytest.mark.tonio
@pytest.mark.parametrize("model_id", ["mistral-medium-2604", "mistral-medium-latest"])
async def test_medium_uses_reasoning_effort_when_thinking_is_enabled(model_id):
    payload = await capture_payload(make_model(model_id, True), SimpleStreamOptions(reasoning="medium"))

    assert payload["reasoningEffort"] == "high"
    assert "promptMode" not in payload


@pytest.mark.tonio
@pytest.mark.parametrize("model_id", ["mistral-medium-2604", "mistral-medium-latest"])
async def test_medium_omits_reasoning_controls_when_thinking_is_off(model_id):
    payload = await capture_payload(make_model(model_id, True))

    assert "reasoningEffort" not in payload
    assert "promptMode" not in payload


# Regression for #8700: the Medium prefix must still respect the model's reasoning capability.
@pytest.mark.tonio
async def test_omits_reasoning_controls_for_non_reasoning_medium_models():
    payload = await capture_payload(make_model("mistral-medium-2505", False), SimpleStreamOptions(reasoning="medium"))

    assert "reasoningEffort" not in payload
    assert "promptMode" not in payload


@pytest.mark.tonio
async def test_uses_the_session_id_as_prompt_cache_key():
    payload = await capture_payload(
        make_model("mistral-large-latest", False), SimpleStreamOptions(session_id="session-123")
    )

    assert payload["promptCacheKey"] == "session-123"


@pytest.mark.tonio
async def test_omits_prompt_cache_key_when_cache_retention_is_disabled():
    payload = await capture_payload(
        make_model("mistral-large-latest", False),
        SimpleStreamOptions(session_id="session-123", cache_retention="none"),
    )

    assert "promptCacheKey" not in payload


# --- pidrei-only: the rename the transport performs on the way out -------------


def test_the_wire_payload_snake_cases_the_sdks_request_fields():
    wire = mistral.to_mistral_wire_payload(
        {
            "model": "m",
            "maxTokens": 10,
            "promptMode": "reasoning",
            "reasoningEffort": "high",
            "promptCacheKey": "s",
            "toolChoice": "auto",
            "messages": [
                {"role": "tool", "toolCallId": "abc", "content": [{"type": "image_url", "imageUrl": "data:..."}]}
            ],
        }
    )

    assert wire["max_tokens"] == 10
    assert wire["prompt_mode"] == "reasoning"
    assert wire["reasoning_effort"] == "high"
    assert wire["prompt_cache_key"] == "s"
    assert wire["tool_choice"] == "auto"
    assert wire["messages"][0]["tool_call_id"] == "abc"
    assert wire["messages"][0]["content"][0]["image_url"] == "data:..."


def test_caller_controlled_json_is_never_renamed():
    # A tool's schema and a tool call's arguments carry arbitrary user keys;
    # the explicit per-structure tables never descend into them.
    wire = mistral.to_mistral_wire_payload(
        {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "t",
                        "parameters": {"properties": {"maxTokens": {"type": "number"}, "imageUrl": {}}},
                    },
                }
            ],
            "messages": [
                {
                    "role": "assistant",
                    "toolCalls": [{"id": "1", "function": {"name": "t", "arguments": '{"maxTokens": 1}'}}],
                }
            ],
        }
    )

    schema = wire["tools"][0]["function"]["parameters"]["properties"]
    assert "maxTokens" in schema
    assert "imageUrl" in schema
    assert wire["messages"][0]["tool_calls"][0]["function"]["arguments"] == '{"maxTokens": 1}'
