"""Mirror of pi's mistral-reasoning-mode.test.ts.

pi captures the payload by pointing the model at a dead port and reading
`onPayload` before the request fails. Here the client is stubbed instead — same
capture point, without depending on a connection refusal.

The payload stays camelCase, as pi's does: the snake_case rename happens at the
client boundary (`to_wire_payload`), which is what the SDK does.
"""

import contextlib

import pytest

from pidrei_ai.api import mistral_conversations as mistral
from pidrei_ai.api.mistral_conversations import stream_simple as stream_simple_mistral
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import Context, SimpleStreamOptions, UserMessage


captured: list[dict] = []


class _CapturingClient:
    def __init__(self, _api_key, _server_url=None, env=None):
        pass

    async def chat_stream(self, payload, *, headers=None, cancel=None, timeout_ms=None):
        captured.append(payload)
        raise RuntimeError("payload captured")


@contextlib.contextmanager
def _stubbed_client():
    original = mistral.MistralClient
    mistral.MistralClient = _CapturingClient
    try:
        yield
    finally:
        mistral.MistralClient = original


@pytest.fixture(autouse=True)
def _reset():
    captured.clear()


def make_context() -> Context:
    return Context(messages=[UserMessage(content="Hello", timestamp=1)])


async def capture_payload(model, options: SimpleStreamOptions | None = None) -> dict:
    opts = options or SimpleStreamOptions()
    opts.api_key = "fake-key"
    with _stubbed_client():
        await stream_simple_mistral(model, make_context(), opts).result()
    assert captured, "Expected payload to be captured before request failure"
    return captured[0]


@pytest.mark.tonio
async def test_uses_reasoning_effort_for_mistral_small_4():
    payload = await capture_payload(
        get_builtin_model("mistral", "mistral-small-2603"), SimpleStreamOptions(reasoning="medium")
    )

    assert payload["reasoningEffort"] == "high"
    assert "promptMode" not in payload


@pytest.mark.tonio
async def test_omits_reasoning_controls_for_mistral_small_4_when_thinking_is_off():
    payload = await capture_payload(get_builtin_model("mistral", "mistral-small-2603"))

    assert "reasoningEffort" not in payload
    assert "promptMode" not in payload


@pytest.mark.tonio
async def test_uses_prompt_mode_for_magistral_reasoning_models():
    payload = await capture_payload(
        get_builtin_model("mistral", "magistral-medium-latest"), SimpleStreamOptions(reasoning="medium")
    )

    assert payload["promptMode"] == "reasoning"
    assert "reasoningEffort" not in payload


@pytest.mark.tonio
async def test_uses_reasoning_effort_for_mistral_medium_3_5():
    payload = await capture_payload(
        get_builtin_model("mistral", "mistral-medium-3.5"), SimpleStreamOptions(reasoning="medium")
    )

    assert payload["reasoningEffort"] == "high"
    assert "promptMode" not in payload


@pytest.mark.tonio
async def test_omits_reasoning_controls_for_mistral_medium_3_5_when_thinking_is_off():
    payload = await capture_payload(get_builtin_model("mistral", "mistral-medium-3.5"))

    assert "reasoningEffort" not in payload
    assert "promptMode" not in payload


@pytest.mark.tonio
async def test_uses_the_session_id_as_prompt_cache_key():
    payload = await capture_payload(
        get_builtin_model("mistral", "mistral-large-latest"), SimpleStreamOptions(session_id="session-123")
    )

    assert payload["promptCacheKey"] == "session-123"


@pytest.mark.tonio
async def test_omits_prompt_cache_key_when_cache_retention_is_disabled():
    payload = await capture_payload(
        get_builtin_model("mistral", "mistral-large-latest"),
        SimpleStreamOptions(session_id="session-123", cache_retention="none"),
    )

    assert "promptCacheKey" not in payload


# --- pidrei-only: the rename the SDK performs on the way out -------------------


def test_the_wire_payload_snake_cases_the_sdks_request_fields():
    wire = mistral.to_wire_payload(
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
    # A tool's schema and a tool call's arguments carry arbitrary user keys; a
    # blanket recursive rename would corrupt them.
    wire = mistral.to_wire_payload(
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
