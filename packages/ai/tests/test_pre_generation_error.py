"""Mirror of pi ai/test/pre-generation-error.test.ts."""

import pytest

from pidrei_ai.api import (
    anthropic_messages,
    azure_openai_responses,
    google_generative_ai,
    mistral_conversations,
    openai_codex_responses,
    openai_completions,
    openai_responses,
)
from pidrei_ai.types import Context, Model, ModelCost, SimpleStreamOptions


def _model(api: str) -> Model:
    return Model(
        id="test-model",
        name="Test",
        api=api,
        provider="test-provider",
        base_url="https://example.invalid",
        reasoning=False,
        input=["text"],
        cost=ModelCost(),
        context_window=1_000,
        max_tokens=100,
    )


async def _expect_pre_generation_error(module, api: str) -> None:
    stream = module.stream_simple(_model(api), Context(messages=[]), SimpleStreamOptions())
    events = [event async for event in stream]
    assert [event.type for event in events] == ["error"]
    result = await stream.result()
    assert result.stop_reason == "error"
    assert result.content == []


@pytest.mark.tonio
async def test_return_an_error_stream_instead_of_throwing_synchronously_when_auth_is_missing():
    await _expect_pre_generation_error(anthropic_messages, "anthropic-messages")
    await _expect_pre_generation_error(azure_openai_responses, "azure-openai-responses")
    await _expect_pre_generation_error(google_generative_ai, "google-generative-ai")
    await _expect_pre_generation_error(mistral_conversations, "mistral-conversations")
    await _expect_pre_generation_error(openai_codex_responses, "openai-codex-responses")
    await _expect_pre_generation_error(openai_completions, "openai-completions")
    await _expect_pre_generation_error(openai_responses, "openai-responses")
