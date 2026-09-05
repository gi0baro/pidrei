"""Mirror of pi's anthropic-mid-conversation-effort.test.ts.

pi captures the payload by throwing from `onPayload`; the beta-header case
reads the header off a fake `fetch`. Here the header comes from a fake
punkreq client swapped in through the `http.client_for` seam — the
`anthropic-beta` header is assembled by the adapter's own transport from the
`betas` request param, exactly where pi's SDK does it.
"""

import json

import pytest

from pidrei_ai.api.anthropic_messages import AnthropicOptions, stream
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import (
    AnthropicMessagesCompat,
    AssistantMessage,
    Context,
    Model,
    ModelCost,
    TextContent,
    ThinkingContent,
    Usage,
    UsageCost,
    UserMessage,
)
from pidrei_ai.utils import http
from tests.anthropic_helpers import PayloadCaptured


def managed_model(provider: str = "anthropic") -> Model:
    return Model(
        id="claude-fable-5-1",
        name="Claude Fable 5.1",
        api="anthropic-messages",
        provider=provider,
        base_url="http://127.0.0.1:9",
        reasoning=True,
        thinking_level_map={
            "off": None,
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "max": "max",
        },
        input=["text"],
        cost=ModelCost(),
        context_window=200000,
        max_tokens=32000,
        compat=AnthropicMessagesCompat(force_adaptive_thinking=True, supports_mid_convo_effort=True),
    )


def assistant(model: Model, level: str | None = None) -> AssistantMessage:
    return AssistantMessage(
        content=[ThinkingContent(thinking="reasoning", thinking_signature="signature"), TextContent(text="answer")],
        api="anthropic-messages",
        provider=model.provider,
        model=model.id,
        provider_thinking_level=level,
        usage=Usage(cost=UsageCost()),
        stop_reason="stop",
        timestamp=1,
    )


async def capture(model: Model, context: Context, effort: str | None = None) -> tuple[dict, AssistantMessage]:
    captured: list[dict] = []

    async def on_payload(value, _model):
        captured.append(value)
        raise PayloadCaptured()

    message = await stream(
        model,
        context,
        AnthropicOptions(
            api_key="test-key", cache_retention="none", thinking_enabled=True, effort=effort, on_payload=on_payload
        ),
    ).result()
    if not captured:
        raise AssertionError("Expected payload capture")
    return captured[0], message


def user(text: str, timestamp: int) -> UserMessage:
    return UserMessage(content=text, timestamp=timestamp)


def effort_messages(payload: dict) -> list[dict]:
    return [message for message in payload["messages"] if message["role"] == "system"]


@pytest.mark.tonio
async def test_reconstructs_an_exact_historical_marker_prefix_and_appends_the_current_marker():
    model = managed_model()
    first_payload, first_message = await capture(model, Context(messages=[user("one", 1)]), "low")
    second_payload, _ = await capture(
        model, Context(messages=[user("one", 1), assistant(model, "low"), user("two", 2)]), "high"
    )

    assert first_payload["messages"] == [
        {"role": "user", "content": "one"},
        {"role": "system", "content": [], "output_config": {"effort": "low"}},
    ]
    assert second_payload["messages"][: len(first_payload["messages"])] == first_payload["messages"]
    assert second_payload["messages"][-1] == {"role": "system", "content": [], "output_config": {"effort": "high"}}
    assert first_payload["output_config"] == {"effort": "high"}
    assert second_payload["output_config"] == {"effort": "high"}
    assert second_payload["thinking"] == {
        "type": "adaptive",
        "display": "summarized",
        "block_binding": {"prefix_mismatch_behavior": "drop_block"},
    }
    assert first_message.provider_thinking_level == "low"


@pytest.mark.tonio
@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
async def test_preserves_native_effort(effort):
    payload, message = await capture(managed_model(), Context(messages=[user("one", 1)]), effort)
    assert effort_messages(payload) == [{"role": "system", "content": [], "output_config": {"effort": effort}}]
    assert message.provider_thinking_level == effort


@pytest.mark.tonio
async def test_defaults_omitted_effort_to_high_and_still_enables_drop_block():
    payload, message = await capture(managed_model(), Context(messages=[user("one", 1)]))
    assert payload["messages"][-1] == {"role": "system", "content": [], "output_config": {"effort": "high"}}
    assert payload["thinking"]["block_binding"]["prefix_mismatch_behavior"] == "drop_block"
    assert message.provider_thinking_level == "high"


@pytest.mark.tonio
async def test_does_not_invent_markers_for_legacy_or_other_provider_assistants():
    model = managed_model()
    legacy = assistant(model)
    other_provider = AssistantMessage(
        content=legacy.content,
        api="anthropic-messages",
        provider="other-provider",
        model=model.id,
        provider_thinking_level="low",
        usage=Usage(cost=UsageCost()),
        stop_reason="stop",
        timestamp=1,
    )
    payload, _ = await capture(
        model,
        Context(messages=[user("one", 1), legacy, user("two", 2), other_provider, user("three", 3)]),
        "medium",
    )
    assert effort_messages(payload) == [{"role": "system", "content": [], "output_config": {"effort": "medium"}}]


@pytest.mark.tonio
async def test_leaves_unsupported_models_on_top_level_effort():
    model = managed_model()
    model.compat = AnthropicMessagesCompat(force_adaptive_thinking=True)
    payload, message = await capture(model, Context(messages=[user("one", 1)]), "low")
    assert payload["messages"] == [{"role": "user", "content": "one"}]
    assert payload["output_config"] == {"effort": "low"}
    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert message.provider_thinking_level is None


class _SseResponse:
    def __init__(self, body: bytes):
        self.status_code = 200
        self.headers = {"content-type": "text/event-stream"}
        self._body = body

    async def iter_bytes(self):
        yield self._body

    async def read(self) -> bytes:
        return self._body

    async def close(self) -> None:
        pass


@pytest.mark.tonio
async def test_sends_the_effort_and_binding_beta_headers():
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_test",
                "model": "claude-fable-5-1",
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        },
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        {"type": "message_stop"},
    ]
    body = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()
    seen_beta_headers: list[str | None] = []

    class FakeClient:
        async def post(self, _url, *, json, headers, timeout):
            seen_beta_headers.append(headers.get("anthropic-beta"))
            return _SseResponse(body)

    original_client_for = http.client_for
    http.client_for = lambda _url, _env=None: FakeClient()
    try:
        result = await stream(
            managed_model(),
            Context(messages=[user("one", 1)]),
            AnthropicOptions(api_key="test-key", cache_retention="none"),
        ).result()
    finally:
        http.client_for = original_client_for

    assert result.stop_reason == "stop"
    assert seen_beta_headers, "expected the transport to send a request"
    beta_header = seen_beta_headers[0] or ""
    assert "mid-conversation-output-config-2026-07-01" in beta_header
    assert "thinking-binding-controls-2026-08-01" in beta_header


def test_generates_exact_model_and_transport_gates():
    direct = get_builtin_model("anthropic", "claude-fable-5-1")
    open_router = get_builtin_model("openrouter", "anthropic/claude-fable-5.1")
    unsupported = get_builtin_model("anthropic", "claude-opus-4-8")
    assert direct.compat.supports_mid_convo_effort is True
    assert "off" in direct.thinking_level_map and direct.thinking_level_map["off"] is None
    assert open_router.api == "anthropic-messages"
    assert open_router.base_url == "https://openrouter.ai/api"
    assert open_router.compat.supports_mid_convo_effort is True
    assert unsupported.compat.supports_mid_convo_effort is None
    assert get_builtin_model("anthropic", "claude-opus-5").compat.allowed_fallback_models is None
