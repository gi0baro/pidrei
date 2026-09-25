"""Mirror of pi's fireworks-model-generation.test.ts.

pi runs the generator script end to end against a mocked models.dev fetch and
reads the fireworks JSON it writes. Here the same pipeline runs in-process:
the Fireworks loader (`_process_fireworks_models`, recording models.dev
reasoning options as `load_models_dev_data` does), then `apply_model_metadata`
(the generator's metadata passes, in their load-bearing order) — everything
the Fireworks entries go through except the fetch and the file write. The
whole `load_models_dev_data` cannot take a Fireworks-only catalog: pidrei's
generator is always-strict, so the Qwen Token Plan allowlist check that pi
gates on `--strict` would reject it.
"""

import importlib.util
from pathlib import Path

import pytest

from pidrei_ai.api.anthropic_messages import stream_simple
from pidrei_ai.models_generated import parse_model_dict
from pidrei_ai.registry import get_supported_thinking_levels
from pidrei_ai.types import Context, SimpleStreamOptions, UserMessage


def _load_generate_models():
    """Import the sibling generator script (not an installed module)."""
    path = Path(__file__).parents[1] / "scripts" / "generate_models.py"
    spec = importlib.util.spec_from_file_location("pidrei_ai_scripts_generate_models", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generate_models = _load_generate_models()


def generate_fireworks_models(options: dict[str, list | None]) -> dict[str, dict]:
    fireworks_models = {
        f"accounts/fireworks/models/{model_id}": {
            "id": model_id,
            "tool_call": True,
            "reasoning": True,
            **({"reasoning_options": reasoning_options} if reasoning_options is not None else {}),
        }
        for model_id, reasoning_options in options.items()
    }
    reasoning_options: dict[str, list] = {}

    def record(provider: str, model_id: str, source: dict) -> None:
        if source.get("reasoning_options") is not None:
            reasoning_options[f"{provider}:{model_id}"] = source["reasoning_options"]

    models = generate_models._process_fireworks_models(fireworks_models, record)
    generate_models.apply_model_metadata(models, reasoning_options)
    return {model["id"]: model for model in models}


# Regression for #9323: import catalog efforts and correct only the known omissions.
def test_combines_upstream_effort_and_toggle_metadata_with_narrow_corrections():
    models = generate_fireworks_models(
        {
            "deepseek-v4-flash-0731": [{"type": "toggle"}, {"type": "effort", "values": ["low", "high", "max"]}],
            "deepseek-v4-flash-vision-exp": [{"type": "toggle"}, {"type": "effort", "values": ["low", "high", "max"]}],
            "deepseek-v4-pro-0813": [{"type": "toggle"}, {"type": "effort", "values": ["high", "max"]}],
            "qwen3p8-max": [{"type": "toggle"}],
            "qwen3p8-2p4t-a95b": [{"type": "effort", "values": ["low", "medium", "xhigh"]}],
            "kimi-k2p6": [{"type": "toggle"}],
        }
    )
    for model_id in ("deepseek-v4-flash-0731", "deepseek-v4-flash-vision-exp", "deepseek-v4-pro-0813"):
        assert models[f"accounts/fireworks/models/{model_id}"]["thinkingLevelMap"] == {
            "off": "none",
            "minimal": None,
            "low": "low",
            "medium": None,
            "high": "high",
            "xhigh": None,
            "max": "max",
        }
    for model_id in ("qwen3p8-max", "qwen3p8-2p4t-a95b"):
        assert models[f"accounts/fireworks/models/{model_id}"]["thinkingLevelMap"] == {
            "off": "none",
            "minimal": None,
            "low": "low",
            "medium": "medium",
            "high": None,
            "xhigh": "xhigh",
            "max": None,
        }
    for model in models.values():
        assert model["api"] == "anthropic-messages"
        assert model["compat"]["allowEmptySignature"] is True
        assert model["compat"].get("forceAdaptiveThinking") is (None if model["id"].endswith("kimi-k2p6") else True)
    assert "thinkingLevelMap" not in models["accounts/fireworks/models/kimi-k2p6"]


# Regression for #9323: new effort-capable models must not require an allowlist update.
@pytest.mark.tonio
async def test_automatically_sends_native_effort_for_newly_cataloged_messages_models():
    models = generate_fireworks_models(
        {"new-reasoner": [{"type": "toggle"}, {"type": "effort", "values": ["low", "max"]}]}
    )
    model = parse_model_dict(models["accounts/fireworks/models/new-reasoner"])
    assert model.api == "anthropic-messages"
    assert model.compat.force_adaptive_thinking is True
    assert get_supported_thinking_levels(model) == ["off", "low", "max"]
    captured: list[dict] = []

    async def on_payload(payload, _model):
        captured.append(payload)
        raise Exception("payload captured")

    await stream_simple(
        model,
        Context(messages=[UserMessage(content="test", timestamp=0)]),
        SimpleStreamOptions(api_key="test-fireworks-key", reasoning="max", on_payload=on_payload),
    ).result()

    assert captured[0]["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert captured[0]["output_config"] == {"effort": "max"}


def test_does_not_infer_adaptive_thinking_from_toggle_budget_or_missing_metadata():
    models = generate_fireworks_models(
        {
            "toggle-only": [{"type": "toggle"}],
            "budget-only": [{"type": "budget_tokens", "min": 1024}],
            "fixed-reasoning": [],
            "missing-metadata": None,
        }
    )
    for model in models.values():
        assert model["api"] == "anthropic-messages"
        assert "forceAdaptiveThinking" not in model["compat"]
        assert "thinkingLevelMap" not in model


# Regression for #9323: full hardcoded maps must not override updated catalog efforts.
def test_prefers_updated_upstream_efforts_over_fixed_maps_and_the_qwen_fallback():
    models = generate_fireworks_models(
        {
            "deepseek-v4-flash-0731": [{"type": "effort", "values": ["high", "max"]}],
            "qwen3p8-max": [{"type": "toggle"}, {"type": "effort", "values": ["medium", "xhigh"]}],
        }
    )
    assert models["accounts/fireworks/models/deepseek-v4-flash-0731"]["thinkingLevelMap"] == {
        "off": None,
        "minimal": None,
        "low": None,
        "medium": None,
        "high": "high",
        "xhigh": None,
        "max": "max",
    }
    assert models["accounts/fireworks/models/qwen3p8-max"]["thinkingLevelMap"] == {
        "off": "none",
        "minimal": None,
        "low": None,
        "medium": "medium",
        "high": None,
        "xhigh": "xhigh",
        "max": None,
    }
