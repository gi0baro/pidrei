"""Mirror of pi's openai-responses-prompt-cache.test.ts.

pi stubs `fetch` and reads the request body back; here the payload comes
straight from `build_params` (`build_request_params`), which is the body pi's
stub parses.
"""

import pytest

from pidrei_ai.api.openai_responses import OpenAIResponsesOptions
from pidrei_ai.types import Context, OpenAIResponsesCompat, UserMessage
from pidrei_ai.utils.transcript import normalize_context
from tests.test_openai_responses import build_request_params, make_model


CONTEXT = normalize_context(Context(messages=[UserMessage(content="hello", timestamp=1)]))


@pytest.fixture(autouse=True)
def _no_cache_retention_env(monkeypatch):
    monkeypatch.delenv("PIDREI_CACHE_RETENTION", raising=False)


def capture_payload(prompt_cache_key_mode=None, cache_retention=None) -> dict:
    model = make_model(
        id="test-model",
        name="Test Model",
        provider="test-proxy",
        base_url="https://proxy.example.com/v1",
        reasoning=False,
        context_window=128000,
        max_tokens=4096,
        compat=OpenAIResponsesCompat(prompt_cache_key_mode=prompt_cache_key_mode),
    )
    return build_request_params(
        model,
        CONTEXT,
        OpenAIResponsesOptions(api_key="test-key", session_id="session-123", cache_retention=cache_retention),
    )


def test_preserves_the_existing_default_behavior():
    assert capture_payload()["prompt_cache_key"] == "session-123"


def test_emits_the_key_when_enabled():
    assert capture_payload("enabled")["prompt_cache_key"] == "session-123"


def test_omits_the_key_when_disabled():
    assert "prompt_cache_key" not in capture_payload("disabled")


def test_lets_cache_retention_none_override_enabled_mode():
    assert "prompt_cache_key" not in capture_payload("enabled", "none")
