"""Mirror of pi's max-thinking.test.ts.

Backfilled with the 0.85.1 sync (the suite dates from pi fbdd4638 and was
never mirrored). pi's `toMatchObject` on the codex thinking-level map is a
subset check, mirrored as one; the Codex Responses payload is captured through
`on_payload`, which fires before any transport is chosen.
"""

import base64
import json
import time

import pytest

from pidrei_ai.api.openai_codex_responses import stream_simple as stream_simple_codex
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.registry import clamp_thinking_level, get_supported_thinking_levels
from pidrei_ai.types import Context, ModelCost, SimpleStreamOptions, UserMessage
from pidrei_ai.utils.transcript import normalize_context
from tests.test_registry import make_model


class PayloadCaptured(Exception):
    pass


def mock_token() -> str:
    payload = base64.b64encode(
        json.dumps({"https://api.openai.com/auth": {"chatgpt_account_id": "acc_test"}}).encode()
    ).decode()
    return f"aaa.{payload}.bbb"


def _assert_matches(actual: dict | None, expected: dict) -> None:
    """pi's `toMatchObject`: every expected entry present and equal."""
    assert actual is not None
    for key, value in expected.items():
        assert actual.get(key) == value, key


def test_is_opt_in_for_ordinary_reasoning_models():
    model = make_model(
        "test",
        "ordinary-reasoning",
        name="Ordinary Reasoning",
        api="openai-completions",
        base_url="https://example.com/v1",
        reasoning=True,
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=128000,
        max_tokens=4096,
    )

    assert get_supported_thinking_levels(model) == ["off", "minimal", "low", "medium", "high"]
    assert clamp_thinking_level(model, "max") == "high"


@pytest.mark.parametrize("model_id", ["gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-luna", "gpt-6-sol"])
def test_exposes_xhigh_and_max_for_openai_codex(model_id):
    model = get_builtin_model("openai-codex", model_id)
    assert model is not None
    _assert_matches(model.thinking_level_map, {"xhigh": "xhigh", "max": "max"})
    assert get_supported_thinking_levels(model) == ["off", "minimal", "low", "medium", "high", "xhigh", "max"]


def test_supports_a_hole_between_high_and_max():
    model = make_model(
        "test",
        "high-and-max",
        name="High and Max",
        api="openai-completions",
        base_url="https://example.com/v1",
        reasoning=True,
        thinking_level_map={"xhigh": None, "max": "max"},
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=128000,
        max_tokens=4096,
    )

    assert get_supported_thinking_levels(model) == ["off", "minimal", "low", "medium", "high", "max"]
    assert clamp_thinking_level(model, "xhigh") == "max"


@pytest.mark.tonio
@pytest.mark.parametrize("model_id", ["gpt-5.6-sol", "gpt-6-astra", "gpt-6-sol", "gpt-6-luna"])
async def test_sends_max_to_the_codex_responses_api(model_id):
    model = get_builtin_model("openai-codex", model_id)
    assert model is not None
    context = normalize_context(
        Context(
            system_prompt="You are a helpful assistant.",
            messages=[UserMessage(content="Hello", timestamp=int(time.time() * 1000))],
        )
    )
    captured: list[dict] = []

    async def on_payload(payload, _model):
        captured.append(payload)
        raise PayloadCaptured("payload captured")

    await stream_simple_codex(
        model,
        context,
        SimpleStreamOptions(api_key=mock_token(), reasoning="max", on_payload=on_payload),
    ).result()

    assert captured
    _assert_matches(captured[0].get("reasoning"), {"effort": "max", "summary": "auto"})
