"""Mirror of pi's anthropic-empty-thinking-signature-compat.test.ts.

The Kimi Coding case joins when that provider's catalog lands (PLAN.md).
"""

import pytest

from pidrei_ai.providers.all import get_builtin_model, get_builtin_models
from pidrei_ai.types import (
    AnthropicMessagesCompat,
    AssistantMessage,
    Context,
    Model,
    ModelCost,
    TextContent,
    ThinkingContent,
    Usage,
    UserMessage,
)
from tests.anthropic_helpers import capture_payload, now_ms


def make_model(allow_empty_signature: bool | None = None) -> Model:
    return Model(
        id="mimo-v2.5-pro",
        name="MiMo-V2.5-Pro",
        api="anthropic-messages",
        provider="xiaomi-token-plan-ams",
        base_url="http://127.0.0.1:9/anthropic",
        reasoning=True,
        input=["text"],
        cost=ModelCost(),
        context_window=1048576,
        max_tokens=1024,
        compat=None
        if allow_empty_signature is None
        else AnthropicMessagesCompat(allow_empty_signature=allow_empty_signature),
    )


def make_context(
    thinking_signature: str,
    thinking: str = "internal reasoning",
    provider: str = "xiaomi-token-plan-ams",
    model_id: str = "mimo-v2.5-pro",
    extra_content: list | None = None,
) -> Context:
    assistant = AssistantMessage(
        content=[ThinkingContent(thinking=thinking, thinking_signature=thinking_signature), *(extra_content or [])],
        provider=provider,
        api="anthropic-messages",
        model=model_id,
        timestamp=now_ms(),
        usage=Usage(),
        stop_reason="stop",
    )
    return Context(
        messages=[
            UserMessage(content="first", timestamp=now_ms()),
            assistant,
            UserMessage(content="second", timestamp=now_ms()),
        ]
    )


def assistant_content(payload: dict) -> list[dict]:
    message = next(message for message in payload["messages"] if message["role"] == "assistant")
    return message["content"]


@pytest.mark.tonio
async def test_converts_empty_signature_thinking_to_text_by_default():
    payload = await capture_payload(make_model(), context=make_context(""))
    assert assistant_content(payload) == [{"type": "text", "text": "internal reasoning"}]


@pytest.mark.tonio
async def test_preserves_empty_thinking_text_when_the_signature_is_present():
    payload = await capture_payload(make_model(), context=make_context("signed-thinking", ""))
    assert assistant_content(payload) == [{"type": "thinking", "thinking": "", "signature": "signed-thinking"}]


@pytest.mark.tonio
async def test_preserves_empty_signature_thinking_when_allow_empty_signature_is_enabled():
    payload = await capture_payload(make_model(True), context=make_context(" "))
    assert assistant_content(payload) == [{"type": "thinking", "thinking": "internal reasoning", "signature": ""}]


# Regression for #9676: Vercel AI Gateway emits unsigned thinking for translated models.
def test_allows_empty_thinking_signatures_for_every_vercel_ai_gateway_model():
    models = get_builtin_models("vercel-ai-gateway")
    assert len(models) > 0
    assert all(model.compat is not None and model.compat.allow_empty_signature is True for model in models)


# Regression for #9323: Fireworks emits unsigned thinking that must survive replay.
@pytest.mark.tonio
@pytest.mark.parametrize(
    "model_id",
    [
        "accounts/fireworks/models/deepseek-v4-flash-0731",
        "accounts/fireworks/models/deepseek-v4-flash-vision-exp",
        "accounts/fireworks/models/deepseek-v4-pro-0813",
        "accounts/fireworks/models/qwen3p8-max",
        "accounts/fireworks/models/qwen3p8-2p4t-a95b",
        "accounts/fireworks/models/kimi-k2p6",
    ],
)
async def test_preserves_unsigned_thinking_for_fireworks(model_id):
    model = get_builtin_model("fireworks", model_id)
    assert model.compat.allow_empty_signature is True
    context = make_context("", "internal reasoning", "fireworks", model_id, [TextContent(text="answer")])
    payload = await capture_payload(model, context=context)
    assert assistant_content(payload) == [
        {"type": "thinking", "thinking": "internal reasoning", "signature": ""},
        {"type": "text", "text": "answer"},
    ]


# Regression for #9323: opting into unsigned replay must not change cross-model conversion.
@pytest.mark.tonio
async def test_still_converts_cross_model_fireworks_thinking_to_text():
    model = get_builtin_model("fireworks", "accounts/fireworks/models/deepseek-v4-flash-0731")
    payload = await capture_payload(
        model,
        context=make_context("", "internal reasoning", "fireworks", "accounts/fireworks/models/kimi-k2p6"),
    )
    assert assistant_content(payload) == [{"type": "text", "text": "internal reasoning"}]
