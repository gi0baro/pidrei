"""Mirror of pi's google-thinking-disable.test.ts.

pi's file is entirely credential-gated E2E: it sends a real request with
reasoning off and asserts no thinking events came back. Without GEMINI_API_KEY /
Vertex credentials every case there skips, so the mirror asserts the same intent
one layer down — the payload `stream_simple` actually builds — the way
`test_anthropic_thinking_disable.py` mirrors pi's Anthropic E2E block.

That makes `_get_disabled_thinking_config` covered rather than merely skipped:
its three branches (Gemini 3 Pro cannot turn thinking off, Gemini 3 Flash can
only go MINIMAL, Gemini 2.x takes a zero budget) are the reason pi's E2E cases
exist at all.
"""

import contextlib
import time
from typing import Any

import pytest

from pidrei_ai.api import google_generative_ai, google_vertex
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import Context, SimpleStreamOptions, UserMessage


CONTEXT = Context(
    system_prompt="You are a precise assistant. Follow the requested output format exactly.",
    messages=[UserMessage(content="Say pong.", timestamp=int(time.time() * 1000))],
)

_CHUNK = {
    "candidates": [{"content": {"parts": [{"text": "pong"}]}, "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
}


@contextlib.contextmanager
def _capturing(adapter):
    """Replace the adapter's client with one that records the params it is given."""
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


async def _config_for(adapter, model, options: SimpleStreamOptions | None = None) -> dict[str, Any]:
    with _capturing(adapter) as captured:
        await adapter.stream_simple(model, CONTEXT, options or SimpleStreamOptions(api_key="test-key")).result()
    assert len(captured) == 1
    return captured[0]["config"]


@pytest.mark.tonio
async def test_disables_thinking_with_a_zero_budget_for_gemini_2_5():
    config = await _config_for(google_generative_ai, get_builtin_model("google", "gemini-2.5-flash"))

    assert config["thinkingConfig"] == {"thinkingBudget": 0}
    assert "includeThoughts" not in config["thinkingConfig"]


@pytest.mark.tonio
async def test_disables_thinking_with_the_minimal_level_for_gemini_3_flash():
    config = await _config_for(google_generative_ai, get_builtin_model("google", "gemini-3-flash-preview"))

    assert config["thinkingConfig"] == {"thinkingLevel": "MINIMAL"}


@pytest.mark.tonio
async def test_uses_the_lowest_level_for_gemini_3_1_pro_which_cannot_disable_thinking():
    config = await _config_for(google_generative_ai, get_builtin_model("google", "gemini-3.1-pro-preview"))

    assert config["thinkingConfig"] == {"thinkingLevel": "LOW"}


@pytest.mark.tonio
async def test_disables_thinking_with_the_minimal_level_for_gemma_4():
    config = await _config_for(google_generative_ai, get_builtin_model("google", "gemma-4-31b-it"))

    assert config["thinkingConfig"] == {"thinkingLevel": "MINIMAL"}


@pytest.mark.tonio
async def test_vertex_disables_thinking_with_a_zero_budget_for_gemini_2_5():
    config = await _config_for(
        google_vertex,
        get_builtin_model("google-vertex", "gemini-2.5-flash"),
        SimpleStreamOptions(api_key="test-key"),
    )

    assert config["thinkingConfig"] == {"thinkingBudget": 0}


@pytest.mark.tonio
async def test_vertex_disables_thinking_with_the_minimal_level_for_gemini_3_flash():
    config = await _config_for(
        google_vertex,
        get_builtin_model("google-vertex", "gemini-3-flash-preview"),
        SimpleStreamOptions(api_key="test-key"),
    )

    assert config["thinkingConfig"] == {"thinkingLevel": "MINIMAL"}


@pytest.mark.tonio
async def test_requests_thoughts_and_a_level_when_reasoning_is_enabled():
    config = await _config_for(
        google_generative_ai,
        get_builtin_model("google", "gemini-3-pro-preview"),
        SimpleStreamOptions(api_key="test-key", reasoning="high"),
    )

    assert config["thinkingConfig"] == {"includeThoughts": True, "thinkingLevel": "HIGH"}


@pytest.mark.tonio
async def test_requests_thoughts_and_a_budget_for_budget_based_models():
    config = await _config_for(
        google_generative_ai,
        get_builtin_model("google", "gemini-2.5-pro"),
        SimpleStreamOptions(api_key="test-key", reasoning="low"),
    )

    assert config["thinkingConfig"] == {"includeThoughts": True, "thinkingBudget": 2048}
