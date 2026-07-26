"""Mirror of pi's anthropic-empty-thinking-signature-compat.test.ts.

The Kimi Coding case joins when that provider's catalog lands (PLAN.md).
"""

import pytest

from pidrei_ai.types import (
    AnthropicMessagesCompat,
    AssistantMessage,
    Context,
    Model,
    ModelCost,
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


def make_context(thinking_signature: str, thinking: str = "internal reasoning") -> Context:
    assistant = AssistantMessage(
        content=[ThinkingContent(thinking=thinking, thinking_signature=thinking_signature)],
        provider="xiaomi-token-plan-ams",
        api="anthropic-messages",
        model="mimo-v2.5-pro",
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
