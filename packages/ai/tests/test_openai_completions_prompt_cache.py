"""Partial mirror of pi's openai-completions-prompt-cache.test.ts.

Holds the session-affinity cases ported with 0.87.1 (the rest of pi's file is
a recorded parity gap — see the classifier's TEST_HOMES). pi fakes the OpenAI
SDK and reads the client's default headers; here the headers come from the
adapter's `_create_client` (precedent: test_github_copilot_headers.py) and
the payload from an `on_payload` capture.
"""

import pytest

from pidrei_ai.api.openai_completions import _create_client, get_compat, stream_simple
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import Context, Model, ModelCost, OpenAICompletionsCompat, SimpleStreamOptions, UserMessage


CONTEXT = Context(messages=[UserMessage(content="hi", timestamp=0)])


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


async def capture_request(session_id: str, model: Model) -> tuple[dict, dict]:
    captured: list[dict] = []

    async def on_payload(payload, _model):
        captured.append(payload)
        raise RuntimeError("payload captured")

    await stream_simple(
        model, CONTEXT, SimpleStreamOptions(api_key="test", session_id=session_id, on_payload=on_payload)
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
