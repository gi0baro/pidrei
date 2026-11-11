"""Partial mirror of pi's openai-completions-prompt-cache.test.ts.

Holds the session-affinity cases ported with 0.87.1 and the 1.5.1
`promptCacheKeyMode` cases (the rest of pi's file is a recorded parity gap —
see the classifier's TEST_HOMES). pi fakes the OpenAI SDK and reads the
client's default headers; here the headers come from the adapter's
`_create_client` (precedent: test_github_copilot_headers.py) and the payload
from an `on_payload` capture.
"""

import pytest

from pidrei_ai.api.openai_completions import _create_client, get_compat, stream_simple
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import Context, Model, ModelCost, OpenAICompletionsCompat, SimpleStreamOptions, UserMessage
from pidrei_ai.utils.transcript import normalize_context


CONTEXT = normalize_context(Context(messages=[UserMessage(content="hi", timestamp=0)]))


@pytest.fixture(autouse=True)
def _no_cache_retention_env(monkeypatch):
    # pi's beforeEach deletes PI_CACHE_RETENTION.
    monkeypatch.delenv("PIDREI_CACHE_RETENTION", raising=False)


def create_model(**overrides) -> Model:
    fields = {
        "id": "gpt-4o-mini",
        "name": "GPT-4o mini",
        "api": "openai-completions",
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "reasoning": False,
        "input": ["text"],
        "cost": ModelCost(),
        "context_window": 128000,
        "max_tokens": 16384,
    }
    fields.update(overrides)
    return Model(**fields)


async def capture_request(session_id: str | None, model: Model, cache_retention=None) -> tuple[dict, dict]:
    captured: list[dict] = []

    async def on_payload(payload, _model):
        captured.append(payload)
        raise RuntimeError("payload captured")

    await stream_simple(
        model,
        CONTEXT,
        SimpleStreamOptions(
            api_key="test", session_id=session_id, cache_retention=cache_retention, on_payload=on_payload
        ),
    ).result()
    assert captured, "payload was not captured"
    headers = _create_client(model, CONTEXT, "test", None, session_id, get_compat(model))._headers
    return captured[0], headers


@pytest.mark.tonio
async def test_sends_baseten_session_affinity_for_built_in_catalog_models():
    model = get_builtin_model("baseten", "zai-org/GLM-5.2")
    _, headers = await capture_request("baseten-catalog-session", model)

    assert headers["x-session-affinity"] == "baseten-catalog-session"
    assert headers["x-client-request-id"] == "baseten-catalog-session"


@pytest.mark.tonio
async def test_sends_openrouter_session_affinity_header_by_default_for_built_in_openrouter_models():
    model = get_builtin_model("openrouter", "auto")
    payload, headers = await capture_request("session-openrouter", model)

    assert "session_id" not in payload
    assert "prompt_cache_key" not in payload
    assert headers["x-session-id"] == "session-openrouter"
    assert "session_id" not in headers
    assert "x-client-request-id" not in headers
    assert "x-session-affinity" not in headers


@pytest.mark.tonio
async def test_omits_openrouter_session_affinity_data_when_disabled():
    model = create_model(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        compat=OpenAICompletionsCompat(send_session_affinity_headers=False),
    )
    payload, headers = await capture_request("session-openrouter", model)

    assert "session_id" not in payload
    assert "prompt_cache_key" not in payload
    assert "x-session-id" not in headers


@pytest.mark.tonio
@pytest.mark.parametrize("cache_retention", [None, "short", "long"])
async def test_sends_a_stable_cache_key_through_an_opted_in_proxy(cache_retention):
    model = create_model(
        base_url="https://proxy.example.com/v1",
        compat=OpenAICompletionsCompat(prompt_cache_key_mode="enabled", supports_long_cache_retention=False),
    )
    first, first_headers = await capture_request("session-proxy", model, cache_retention)
    repeated, _ = await capture_request("session-proxy", model, cache_retention)
    other, _ = await capture_request("other-session", model, cache_retention)

    assert first["prompt_cache_key"] == "session-proxy"
    assert repeated["prompt_cache_key"] == first["prompt_cache_key"]
    assert other["prompt_cache_key"] == "other-session"
    assert "prompt_cache_retention" not in first
    assert "session_id" not in first_headers


@pytest.mark.tonio
@pytest.mark.parametrize("mode", [None, "auto"])
async def test_preserves_default_proxy_behavior(mode):
    model = create_model(
        base_url="https://proxy.example.com/v1",
        compat=OpenAICompletionsCompat(prompt_cache_key_mode=mode) if mode else None,
    )
    short, _ = await capture_request("session-proxy", model)
    long, _ = await capture_request("session-proxy", model, "long")

    assert "prompt_cache_key" not in short
    assert long["prompt_cache_key"] == "session-proxy"
    assert long["prompt_cache_retention"] == "24h"


@pytest.mark.tonio
async def test_respects_disabled_caching_on_an_opted_in_proxy(monkeypatch):
    monkeypatch.setenv("PIDREI_CACHE_RETENTION", "long")
    model = create_model(
        base_url="https://proxy.example.com/v1", compat=OpenAICompletionsCompat(prompt_cache_key_mode="enabled")
    )
    payload, _ = await capture_request("session-proxy", model, "none")

    assert "prompt_cache_key" not in payload
    assert "prompt_cache_retention" not in payload


@pytest.mark.tonio
@pytest.mark.parametrize("cache_retention", ["short", "long"])
async def test_allows_disabling_cache_keys(cache_retention):
    model = create_model(compat=OpenAICompletionsCompat(prompt_cache_key_mode="disabled"))
    payload, _ = await capture_request("session-disabled", model, cache_retention)

    assert "prompt_cache_key" not in payload
    assert payload.get("prompt_cache_retention") == ("24h" if cache_retention == "long" else None)


@pytest.mark.tonio
async def test_lets_a_proxy_disable_cache_keys_independently_of_long_retention():
    model = create_model(
        base_url="https://proxy.example.com/v1",
        compat=OpenAICompletionsCompat(prompt_cache_key_mode="disabled", supports_long_cache_retention=True),
    )
    payload, _ = await capture_request("session-proxy", model, "long")

    assert "prompt_cache_key" not in payload
    assert payload["prompt_cache_retention"] == "24h"


@pytest.mark.tonio
async def test_does_not_invent_an_identity_for_an_opted_in_proxy():
    model = create_model(
        base_url="https://proxy.example.com/v1", compat=OpenAICompletionsCompat(prompt_cache_key_mode="enabled")
    )
    payload, _ = await capture_request(None, model)

    assert "prompt_cache_key" not in payload


@pytest.mark.tonio
async def test_clamps_proxy_cache_keys_to_64_unicode_characters():
    model = create_model(
        base_url="https://proxy.example.com/v1", compat=OpenAICompletionsCompat(prompt_cache_key_mode="enabled")
    )
    payload, _ = await capture_request("𐐀" * 67, model)

    assert payload["prompt_cache_key"] == "𐐀" * 64
