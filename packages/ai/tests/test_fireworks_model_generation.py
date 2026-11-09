"""Mirror of pi's fireworks-model-generation.test.ts.

pi runs the generator script end to end against mocked upstream catalogs and
reads the fireworks JSON it writes. Here the whole catalog build runs
in-process against the same mocked sources (`catalog_generation_utils`):
everything the Fireworks entries go through except the fetch and the file
write.
"""

import pytest

from pidrei_ai.api.anthropic_messages import stream_simple
from pidrei_ai.models_generated import parse_model_dict
from pidrei_ai.registry import get_supported_thinking_levels
from pidrei_ai.types import Context, SimpleStreamOptions, UserMessage
from pidrei_ai.utils.transcript import normalize_context
from tests.catalog_generation_utils import generate_chat_models


def _source_model(model_id: str, reasoning_options: list[dict] | None) -> dict:
    effort = next((option for option in reasoning_options or [] if option["type"] == "effort"), None)
    budget = next((option for option in reasoning_options or [] if option["type"] == "budget_tokens"), None)
    return {
        "id": f"accounts/fireworks/models/{model_id}",
        "type": "chat",
        "name": model_id,
        "modalities": {"input": ["text"], "output": ["text"]},
        "capabilities": {
            "tools": {"supported": True},
            "reasoning": {
                "supported": True,
                **({"toggle": True} if any(option["type"] == "toggle" for option in reasoning_options or []) else {}),
                **({"effort": effort["values"]} if effort else {}),
                # JSON drops pi's undefined min/max.
                **({"budget": {key: budget[key] for key in ("min", "max") if key in budget}} if budget else {}),
            },
        },
        "api": {},
    }


def generate_fireworks_models(capsys, options: dict[str, list | None]) -> dict[str, dict]:
    models = [_source_model(model_id, reasoning_options) for model_id, reasoning_options in options.items()]
    generated = generate_chat_models(
        {"fireworks-ai": {"models": {model["id"]: model for model in models}}}, "fireworks"
    )
    assert capsys.readouterr().err == ""
    return generated


# Regression for #9323: import catalog efforts and correct only the known omissions.
def test_combines_upstream_effort_and_toggle_metadata_with_narrow_corrections(capsys):
    models = generate_fireworks_models(
        capsys,
        {
            "deepseek-v4-flash-0731": [{"type": "toggle"}, {"type": "effort", "values": ["low", "high", "max"]}],
            "deepseek-v4-flash-vision-exp": [{"type": "toggle"}, {"type": "effort", "values": ["low", "high", "max"]}],
            "deepseek-v4-pro-0813": [{"type": "toggle"}, {"type": "effort", "values": ["high", "max"]}],
            "qwen3p8-max": [{"type": "toggle"}],
            "qwen3p8-2p4t-a95b": [{"type": "effort", "values": ["low", "medium", "xhigh"]}],
            "kimi-k2p6": [{"type": "toggle"}],
        },
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
async def test_automatically_sends_native_effort_for_newly_cataloged_messages_models(capsys):
    models = generate_fireworks_models(
        capsys, {"new-reasoner": [{"type": "toggle"}, {"type": "effort", "values": ["low", "max"]}]}
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
        normalize_context(Context(messages=[UserMessage(content="test", timestamp=0)])),
        SimpleStreamOptions(api_key="test-fireworks-key", reasoning="max", on_payload=on_payload),
    ).result()

    assert captured[0]["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert captured[0]["output_config"] == {"effort": "max"}


def test_does_not_infer_adaptive_thinking_from_toggle_budget_or_missing_metadata(capsys):
    models = generate_fireworks_models(
        capsys,
        {
            "toggle-only": [{"type": "toggle"}],
            "budget-only": [{"type": "budget_tokens", "min": 1024}],
            "fixed-reasoning": [],
            "missing-metadata": None,
        },
    )
    for model in models.values():
        assert model["api"] == "anthropic-messages"
        assert "forceAdaptiveThinking" not in model["compat"]
        assert "thinkingLevelMap" not in model


# Regression for #9323: full hardcoded maps must not override updated catalog efforts.
def test_prefers_updated_upstream_efforts_over_fixed_maps_and_the_qwen_fallback(capsys):
    models = generate_fireworks_models(
        capsys,
        {
            "deepseek-v4-flash-0731": [{"type": "effort", "values": ["high", "max"]}],
            "qwen3p8-max": [{"type": "toggle"}, {"type": "effort", "values": ["medium", "xhigh"]}],
        },
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
