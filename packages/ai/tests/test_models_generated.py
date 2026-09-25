"""Tests for the vendored catalog loader (models_generated.py)."""

import pytest

from pidrei_ai.models_generated import MODELS, parse_model_dict
from pidrei_ai.registry import get_supported_thinking_levels
from pidrei_ai.types import AnthropicMessagesCompat, Model, ModelCost, ModelCostTier, OpenAIResponsesCompat


def test_catalog_contains_phase1_providers():
    assert "anthropic" in MODELS
    assert "openai" in MODELS
    assert all(isinstance(model, Model) for models in MODELS.values() for model in models)


def test_anthropic_models_have_expected_shape():
    anthropic = MODELS["anthropic"]
    assert anthropic, "anthropic catalog must not be empty"
    haiku = next(model for model in anthropic if model.id == "claude-haiku-4-5")

    assert haiku.api == "anthropic-messages"
    assert haiku.provider == "anthropic"
    assert haiku.base_url == "https://api.anthropic.com"
    assert haiku.context_window > 0
    assert haiku.max_tokens > 0
    assert isinstance(haiku.compat, AnthropicMessagesCompat)
    assert haiku.compat.supports_strict_tools is True


def test_adaptive_thinking_model_metadata_round_trips():
    fable = next(model for model in MODELS["anthropic"] if model.id == "claude-fable-5")

    assert isinstance(fable.compat, AnthropicMessagesCompat)
    assert fable.compat.force_adaptive_thinking is True
    # JSON null must load as present-with-None ("off" unsupported), and the
    # explicit xhigh/max entries must enable those levels.
    assert fable.thinking_level_map is not None
    assert "off" in fable.thinking_level_map
    assert fable.thinking_level_map["off"] is None
    assert get_supported_thinking_levels(fable) == ["minimal", "low", "medium", "high", "xhigh", "max"]


def test_includes_xhigh_and_max_for_anthropic_opus_5_on_anthropic_messages_api():
    opus = next(model for model in MODELS["anthropic"] if model.id == "claude-opus-5")

    levels = get_supported_thinking_levels(opus)
    assert "xhigh" in levels
    assert "max" in levels


def test_includes_claude_opus_5_5_with_its_always_on_effort_levels_and_official_pricing():
    model = next(model for model in MODELS["anthropic"] if model.id == "claude-opus-5-5")

    assert (model.cost.input, model.cost.output, model.cost.cache_read, model.cost.cache_write) == (4, 20, 0.2, 5)
    assert model.context_window == 1_000_000
    assert model.max_tokens == 128_000
    assert isinstance(model.compat, AnthropicMessagesCompat)
    assert model.compat.force_adaptive_thinking is True
    assert model.compat.supports_mid_convo_effort is True
    assert model.compat.supports_mid_convo_system_messages is True
    assert model.compat.supports_mid_convo_tool_changes is True
    assert get_supported_thinking_levels(model) == ["low", "medium", "high", "xhigh", "max"]


def test_includes_xhigh_but_not_off_or_max_for_xai_grok_46():
    grok = next(model for model in MODELS["xai"] if model.id == "grok-4.6")

    assert get_supported_thinking_levels(grok) == ["low", "medium", "high", "xhigh"]


@pytest.mark.parametrize(
    "model_id", ["gpt-5.5", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-6-astra", "gpt-6-sol", "gpt-6-luna"]
)
def test_includes_xhigh_for_openai_codex_models(model_id):
    model = next((model for model in MODELS["openai-codex"] if model.id == model_id), None)
    assert model is not None

    assert "xhigh" in get_supported_thinking_levels(model)


@pytest.mark.parametrize(
    ("model_id", "cost"),
    [
        ("gpt-6-sol", ModelCost(input=2, output=10, cache_read=0.2, cache_write=2.5)),
        ("gpt-6-luna", ModelCost(input=0.1, output=0.5, cache_read=0.01, cache_write=0.125)),
    ],
)
def test_includes_official_metadata_for_openai_and_codex(model_id, cost):
    for provider in ("openai", "openai-codex"):
        model = next((model for model in MODELS[provider] if model.id == model_id), None)
        assert model is not None, provider
        assert model.input == ["text", "image"]
        assert model.cost == ModelCost(
            input=cost.input,
            output=cost.output,
            cache_read=cost.cache_read,
            cache_write=cost.cache_write,
            tiers=[
                ModelCostTier(
                    input_tokens_above=272000,
                    input=cost.input * 2,
                    output=cost.output * 1.5,
                    cache_read=cost.cache_read * 2,
                    cache_write=cost.cache_write * 2,
                )
            ],
        )
        assert model.context_window == 272000
        assert model.max_tokens == 128000
        assert isinstance(model.compat, OpenAIResponsesCompat)
        assert model.compat.supports_additional_tools is True
        assert model.compat.supports_mid_convo_system_messages is True
        assert model.compat.supports_openai_grammar_tools is True
        assert model.compat.supports_tool_search is True


def test_includes_low_for_deepseek_v4_flash_on_opencode_go():
    flash = next(model for model in MODELS["opencode-go"] if model.id == "deepseek-v4-flash")

    assert get_supported_thinking_levels(flash) == ["off", "low", "high", "max"]


def test_preserves_low_high_max_metadata_for_deepseek_v4_1_flash_on_openrouter():
    model = next((model for model in MODELS["openrouter"] if model.id == "deepseek/deepseek-v4.1-flash"), None)
    assert model is not None

    assert get_supported_thinking_levels(model) == ["off", "low", "high", "max"]


def test_preserves_low_high_max_metadata_for_deepseek_v4_1_flash_on_opencode_go():
    model = next((model for model in MODELS["opencode-go"] if model.id == "deepseek-v4.1-flash"), None)
    assert model is not None

    assert get_supported_thinking_levels(model) == ["low", "high", "max"]


def test_includes_xhigh_and_max_for_bedrock_claude_opus_5():
    opus = next(model for model in MODELS["amazon-bedrock"] if model.id == "global.anthropic.claude-opus-5")

    levels = get_supported_thinking_levels(opus)
    assert "xhigh" in levels
    assert "max" in levels


def test_openai_models_have_typed_compat():
    openai_models = MODELS["openai"]
    assert openai_models
    assert all(model.api == "openai-responses" for model in openai_models)
    with_compat = [model for model in openai_models if model.compat is not None]
    assert with_compat
    assert all(isinstance(model.compat, OpenAIResponsesCompat) for model in with_compat)


def _responses_model_dict(compat: dict) -> dict:
    return {
        "id": "grok-build-0.1",
        "name": "grok-build-0.1",
        "api": "openai-responses",
        "provider": "opencode",
        "baseUrl": "https://opencode.ai/zen/v1",
        "reasoning": True,
        "input": ["text"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 256000,
        "maxTokens": 64000,
        "compat": compat,
    }


def test_drops_a_compat_field_that_belongs_to_another_api():
    # pi's generator hardcodes `supportsReasoningEffort: false` for this model,
    # which models.dev has since re-typed as an `openai-responses` model. TS
    # carries the stray key and no adapter reads it; the typed dataclass here
    # cannot hold it, so it is dropped, like any key the class does not declare.
    model = parse_model_dict(
        _responses_model_dict({"sessionAffinityFormat": "openai-nosession", "supportsReasoningEffort": False})
    )

    assert isinstance(model.compat, OpenAIResponsesCompat)
    assert model.compat.session_affinity_format == "openai-nosession"


def test_openai_long_context_pricing_tiers_load():
    tiered = [model for model in MODELS["openai"] if model.cost.tiers]
    assert tiered, "expected at least one OpenAI model with long-context pricing tiers"
    tier = tiered[0].cost.tiers[0]
    assert tier.input_tokens_above == 272000
    assert tier.input > tiered[0].cost.input
