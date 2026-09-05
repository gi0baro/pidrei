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


def _expect_missing_auth_raises(module, api: str) -> None:
    with pytest.raises(Exception, match="No API key for provider: test-provider"):
        module.stream_simple(_model(api), Context(messages=[]), SimpleStreamOptions())


def test_throws_synchronously_when_auth_is_missing():
    _expect_missing_auth_raises(anthropic_messages, "anthropic-messages")
    _expect_missing_auth_raises(azure_openai_responses, "azure-openai-responses")
    _expect_missing_auth_raises(google_generative_ai, "google-generative-ai")
    _expect_missing_auth_raises(mistral_conversations, "mistral-conversations")
    _expect_missing_auth_raises(openai_codex_responses, "openai-codex-responses")
    _expect_missing_auth_raises(openai_completions, "openai-completions")
    _expect_missing_auth_raises(openai_responses, "openai-responses")
