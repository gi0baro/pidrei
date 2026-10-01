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
from pidrei_ai.utils.transcript import normalize_context


captured: list[dict] = []


class _PayloadCaptured(Exception):
    pass


async def _capturing_on_payload(payload, _model):
    captured.append(payload)
    raise _PayloadCaptured("payload captured")


@pytest.fixture(autouse=True)
def _reset():
    captured.clear()


NONE_HIGH_LEVELS = {
    "off": "none",
    "minimal": None,
    "low": None,
    "medium": None,
    "high": "high",
    "xhigh": None,
    "max": None,
}
GLM_5_2_LEVELS = {**NONE_HIGH_LEVELS, "max": "max"}
GLM_5_3_LEVELS = {**NONE_HIGH_LEVELS, "off": None, "low": "low", "max": "max"}


def make_model(model_id: str, reasoning: bool, thinking_level_map: dict | None = None) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="mistral-conversations",
        provider="mistral",
        base_url="http://127.0.0.1:9",
        reasoning=reasoning,
        thinking_level_map=thinking_level_map,
        input=["text"],
        cost=ModelCost(),
        context_window=128000,
        max_tokens=16384,
    )


def make_context() -> Context:
    return normalize_context(Context(messages=[UserMessage(content="Hello", timestamp=1)]))


async def capture_payload(model, options: SimpleStreamOptions | None = None) -> dict:
    opts = options or SimpleStreamOptions()
    opts.api_key = "fake-key"
    opts.on_payload = _capturing_on_payload
    await stream_simple_mistral(model, make_context(), opts).result()
    assert captured, "Expected payload to be captured before request failure"
    return captured[0]


@pytest.mark.tonio
async def test_uses_prompt_mode_for_reasoning_models_without_a_thinking_level_map_magistral():
    payload = await capture_payload(
        make_model("magistral-medium-latest", True), SimpleStreamOptions(reasoning="medium")
    )

    assert payload["promptMode"] == "reasoning"
    assert "reasoningEffort" not in payload


@pytest.mark.tonio
async def test_omits_reasoning_controls_for_magistral_when_thinking_is_off():
    payload = await capture_payload(make_model("magistral-medium-latest", True))

    assert "promptMode" not in payload
    assert "reasoningEffort" not in payload


# Regression for #8700 and #9375: Medium and GLM-5.2 ignore Magistral's prompt_mode.
_EFFORT_MODELS = ["mistral-small-2603", "mistral-medium-latest", "zai-glm-5-2"]


def _effort_map(model_id: str) -> dict:
    return GLM_5_2_LEVELS if model_id == "zai-glm-5-2" else NONE_HIGH_LEVELS


@pytest.mark.tonio
@pytest.mark.parametrize("model_id", _EFFORT_MODELS)
async def test_uses_reasoning_effort_when_thinking_is_enabled(model_id):
    payload = await capture_payload(
        make_model(model_id, True, _effort_map(model_id)), SimpleStreamOptions(reasoning="high")
    )

    assert payload["reasoningEffort"] == "high"
    assert "promptMode" not in payload


@pytest.mark.tonio
@pytest.mark.parametrize("model_id", _EFFORT_MODELS)
async def test_clamps_unsupported_levels_to_a_supported_effort(model_id):
    payload = await capture_payload(
        make_model(model_id, True, _effort_map(model_id)), SimpleStreamOptions(reasoning="low")
    )

    assert payload["reasoningEffort"] == "high"


@pytest.mark.tonio
@pytest.mark.parametrize("model_id", _EFFORT_MODELS)
async def test_sends_reasoning_effort_none_when_thinking_is_off(model_id):
    payload = await capture_payload(make_model(model_id, True, _effort_map(model_id)))

    assert payload["reasoningEffort"] == "none"
    assert "promptMode" not in payload


# Regression for #9678: requested levels must reach Mistral-hosted GLM models.
@pytest.mark.tonio
async def test_sends_max_for_glm_5_2():
    payload = await capture_payload(
        make_model("zai-glm-5-2", True, GLM_5_2_LEVELS), SimpleStreamOptions(reasoning="max")
    )

    assert payload["reasoningEffort"] == "max"


@pytest.mark.tonio
@pytest.mark.parametrize("level", ["low", "high", "max"])
async def test_zai_glm_5_3_sends_reasoning_effort(level):
    payload = await capture_payload(
        make_model("zai-glm-5-3", True, GLM_5_3_LEVELS), SimpleStreamOptions(reasoning=level)
    )

    assert payload["reasoningEffort"] == level
    assert "promptMode" not in payload


@pytest.mark.tonio
async def test_zai_glm_5_3_maps_medium_to_high():
    payload = await capture_payload(
        make_model("zai-glm-5-3", True, GLM_5_3_LEVELS), SimpleStreamOptions(reasoning="medium")
    )

    assert payload["reasoningEffort"] == "high"


# Regression for #8700: reasoning controls must respect the model's reasoning capability.
@pytest.mark.tonio
async def test_omits_reasoning_controls_for_non_reasoning_models():
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
