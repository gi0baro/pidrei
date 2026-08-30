"""Mirror of pi's openrouter-reasoning-options.test.ts.

pi's map helper lives in scripts/openrouter-reasoning-options.ts; pidrei
consolidates it into scripts/generate_models.py next to its models.dev
sibling, so the map cases import the generator script. The payload cases
mirror pi's `onPayload` capture: the callback records the request and raises,
which surfaces as an errored message rather than a sent request.
"""

import importlib.util
from pathlib import Path

import pytest

from pidrei_ai.api.openai_completions import stream_simple
from pidrei_ai.types import Context, Model, ModelCost, OpenAICompletionsCompat, SimpleStreamOptions, UserMessage


def _load_generate_models():
    """Import the sibling generator script (not an installed module)."""
    path = Path(__file__).parents[1] / "scripts" / "generate_models.py"
    spec = importlib.util.spec_from_file_location("pidrei_ai_scripts_generate_models", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generate_models = _load_generate_models()
get_openrouter_thinking_level_map = generate_models.get_openrouter_thinking_level_map

CONTEXT = Context(messages=[UserMessage(content="Hello", timestamp=0)])


def openrouter_model(thinking_level_map: dict | None = None) -> Model:
    return Model(
        id="stealth/ox-alpha",
        name="Ox Alpha",
        api="openai-completions",
        provider="openrouter",
        base_url="https://example.invalid/v1",
        reasoning=True,
        thinking_level_map=thinking_level_map,
        input=["text"],
        cost=ModelCost(),
        context_window=128000,
        max_tokens=4096,
        compat=OpenAICompletionsCompat(thinking_format="openrouter"),
    )


async def capture_payload(model: Model, reasoning: str | None = None) -> dict:
    captured: list[dict] = []

    async def on_payload(payload, _model):
        captured.append(payload)
        raise Exception("payload captured")

    options = SimpleStreamOptions(api_key="test", reasoning=reasoning, on_payload=on_payload)
    await stream_simple(model, CONTEXT, options).result()
    if not captured:
        raise AssertionError("OpenRouter payload was not captured")
    return captured[0]


def test_marks_mandatory_reasoning_and_unsupported_efforts_unavailable():
    assert get_openrouter_thinking_level_map(
        {
            "mandatory": True,
            "default_enabled": True,
            "supported_efforts": ["max", "high", "low"],
            "default_effort": "max",
        }
    ) == {
        "off": None,
        "minimal": None,
        "low": "low",
        "medium": None,
        "high": "high",
        "xhigh": None,
        "max": "max",
    }


def test_still_marks_off_unavailable_when_openrouter_omits_effort_metadata():
    assert get_openrouter_thinking_level_map({"mandatory": True}) == {"off": None}


def test_keeps_off_available_while_restricting_optional_models_to_supported_efforts():
    assert get_openrouter_thinking_level_map(
        {
            "mandatory": False,
            "default_enabled": True,
            "supported_efforts": ["high", "low"],
        }
    ) == {
        "off": "none",
        "minimal": None,
        "low": "low",
        "medium": None,
        "high": "high",
        "xhigh": None,
        "max": None,
    }


def test_does_not_add_metadata_for_optional_models_without_effort_controls():
    assert get_openrouter_thinking_level_map({"mandatory": False}) is None


MANDATORY_MAP = get_openrouter_thinking_level_map({"mandatory": True, "supported_efforts": ["max", "high", "low"]})


@pytest.mark.tonio
async def test_omits_reasoning_when_a_background_call_does_not_request_it():
    payload = await capture_payload(openrouter_model(MANDATORY_MAP))

    assert "reasoning" not in payload


@pytest.mark.tonio
async def test_still_sends_an_explicitly_selected_supported_effort():
    payload = await capture_payload(openrouter_model(MANDATORY_MAP), "low")

    assert payload["reasoning"]["effort"] == "low"


@pytest.mark.tonio
async def test_continues_to_explicitly_disable_reasoning_for_optional_models():
    payload = await capture_payload(openrouter_model())

    assert payload["reasoning"]["effort"] == "none"
