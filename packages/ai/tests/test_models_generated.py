"""Tests for the vendored catalog loader (models_generated.py)."""

from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import get_supported_thinking_levels
from pidrei_ai.types import AnthropicMessagesCompat, Model, OpenAIResponsesCompat


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


def test_openai_long_context_pricing_tiers_load():
    tiered = [model for model in MODELS["openai"] if model.cost.tiers]
    assert tiered, "expected at least one OpenAI model with long-context pricing tiers"
    tier = tiered[0].cost.tiers[0]
    assert tier.input_tokens_above == 272000
    assert tier.input > tiered[0].cost.input
