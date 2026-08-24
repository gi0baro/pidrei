"""Mirror of pi's google-thinking-level-map.test.ts.

pi captures the request through `onPayload` and throws from it to abort the
call; pidrei's Google adapters build their params behind the `GoogleGenAI`
client seam, so this swaps the client the way `test_google_thinking_disable.py`
does and reads the captured `config` directly.
"""

import contextlib
import time
from typing import Any

import pytest

from pidrei_ai.api import google_generative_ai, google_vertex
from pidrei_ai.api.google_shared import resolve_google_thinking_level
from pidrei_ai.types import (
    Context,
    Model,
    ModelCost,
    SimpleStreamOptions,
    ThinkingBudgets,
    ThinkingLevelMap,
    UserMessage,
)


CONTEXT = Context(messages=[UserMessage(content="Hello", timestamp=int(time.time() * 1000))])

_CHUNK = {
    "candidates": [{"content": {"parts": [{"text": "pong"}]}, "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
}


def _model(api: str, provider: str, base_url: str, model_id: str, thinking_level_map: ThinkingLevelMap) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api=api,
        provider=provider,
        base_url=base_url,
        reasoning=True,
        thinking_level_map=thinking_level_map,
        input=["text"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=128000,
        max_tokens=4096,
    )


def google_model(model_id: str, thinking_level_map: ThinkingLevelMap) -> Model:
    return _model("google-generative-ai", "test-google", "https://example.invalid/v1beta", model_id, thinking_level_map)


def vertex_model(model_id: str, thinking_level_map: ThinkingLevelMap) -> Model:
    return _model("google-vertex", "test-vertex", "https://example.invalid/v1", model_id, thinking_level_map)


@contextlib.contextmanager
def _capturing(adapter):
    captured: list[dict[str, Any]] = []

    class _Recorder:
        def __init__(self, _config):
            pass

        async def generate_content_stream(self, params, *, env=None, cancel=None):
            captured.append(params)

            async def _chunks():
                yield _CHUNK

            return _chunks()

    original = adapter.GoogleGenAI
    adapter.GoogleGenAI = _Recorder
    try:
        yield captured
    finally:
        adapter.GoogleGenAI = original


async def _capture_config(
    adapter, model: Model, reasoning: str, thinking_budgets: ThinkingBudgets | None = None
) -> dict[str, Any]:
    options = SimpleStreamOptions(api_key="test", reasoning=reasoning, thinking_budgets=thinking_budgets)
    with _capturing(adapter) as captured:
        await adapter.stream_simple(model, CONTEXT, options).result()
    assert len(captured) == 1
    return captured[0]["config"]


@pytest.mark.tonio
async def test_exhaustively_resolves_supported_logical_levels_and_mapping_values():
    default_expectations = {"off": "high", "minimal": "minimal", "low": "low", "medium": "medium", "high": "high"}
    for level, expected in default_expectations.items():
        assert resolve_google_thinking_level(google_model("gemini-3.7-flash", {}), level) == expected

    mapped_expectations = {
        "minimal": "minimal",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "MINIMAL": "minimal",
        "LOW": "low",
        "MEDIUM": "medium",
        "HIGH": "high",
    }
    for mapped, expected in mapped_expectations.items():
        model = google_model("gemini-3.7-flash", {"high": mapped, "xhigh": mapped, "max": mapped})
        assert resolve_google_thinking_level(model, "high") == expected
        assert resolve_google_thinking_level(model, "xhigh") == expected
        assert resolve_google_thinking_level(model, "max") == expected

    invalid_model = google_model("gemini-3.7-flash", {"xhigh": "extreme"})
    with pytest.raises(
        Exception,
        match="Unsupported Google thinking level mapping for test-google/gemini-3.7-flash: xhigh -> extreme",
    ):
        resolve_google_thinking_level(invalid_model, "xhigh")
    with pytest.raises(
        Exception,
        match="Unsupported Google thinking level mapping for test-google/gemini-3.7-flash: max -> undefined",
    ):
        resolve_google_thinking_level(google_model("gemini-3.7-flash", {}), "max")


@pytest.mark.tonio
@pytest.mark.parametrize("reasoning", ["xhigh", "max"])
async def test_maps_google_generative_ai_extended_levels_to_a_supported_level(reasoning):
    config = await _capture_config(
        google_generative_ai, google_model("gemini-3.7-flash", {"xhigh": "high", "max": "high"}), reasoning
    )

    assert config["thinkingConfig"] == {"includeThoughts": True, "thinkingLevel": "HIGH"}


@pytest.mark.tonio
async def test_honors_uppercase_provider_values_for_standard_google_generative_ai_levels():
    config = await _capture_config(google_generative_ai, google_model("gemini-3.7-flash", {"high": "LOW"}), "high")

    assert config["thinkingConfig"]["thinkingLevel"] == "LOW"


@pytest.mark.tonio
async def test_uses_mapped_google_generative_ai_levels_for_token_budgets():
    config = await _capture_config(
        google_generative_ai,
        google_model("gemini-2.5-flash", {"xhigh": "high"}),
        "xhigh",
        ThinkingBudgets(high=1234),
    )

    assert config["thinkingConfig"]["thinkingBudget"] == 1234


@pytest.mark.tonio
async def test_maps_google_vertex_extended_levels():
    config = await _capture_config(google_vertex, vertex_model("gemini-3.7-flash", {"xhigh": "high"}), "xhigh")

    assert config["thinkingConfig"] == {"includeThoughts": True, "thinkingLevel": "HIGH"}


@pytest.mark.tonio
async def test_uses_mapped_google_vertex_levels_for_token_budgets():
    config = await _capture_config(
        google_vertex, vertex_model("gemini-2.5-flash", {"max": "high"}), "max", ThinkingBudgets(high=4321)
    )

    assert config["thinkingConfig"]["thinkingBudget"] == 4321
