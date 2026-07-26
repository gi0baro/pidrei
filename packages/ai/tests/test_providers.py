"""Tests for env-api-keys and the builtin provider wiring.

Partial mirror of pi's providers.test.ts: its `envApiKeyAuth` and
`createProvider` describes live in test_registry.py, and the bedrock / cloudflare
/ vertex auth-flow cases join with those providers (PLAN.md Phase 5d).
"""

import pytest

from pidrei_ai.env_api_keys import AMBIENT_AUTH_MARKER, find_env_keys, get_env_api_key
from pidrei_ai.providers.all import (
    builtin_models,
    builtin_providers,
    get_builtin_model,
    get_builtin_model_data_generated_at,
    get_builtin_models,
    get_builtin_providers,
)
from pidrei_ai.types import ModelCost


def test_find_env_keys_uses_provider_env_overrides():
    assert find_env_keys("openai") is None or isinstance(find_env_keys("openai"), list)
    assert find_env_keys("openai", {"OPENAI_API_KEY": "k"}) == ["OPENAI_API_KEY"]
    assert find_env_keys("unknown-provider", {"WHATEVER": "x"}) is None


def test_anthropic_oauth_token_takes_precedence():
    env = {"ANTHROPIC_API_KEY": "api-key", "ANTHROPIC_OAUTH_TOKEN": "oauth-token"}
    assert get_env_api_key("anthropic", env) == "oauth-token"
    assert get_env_api_key("anthropic", {"ANTHROPIC_API_KEY": "api-key"}) == "api-key"


def test_bedrock_ambient_credentials_marker():
    env = {"AWS_ACCESS_KEY_ID": "id", "AWS_SECRET_ACCESS_KEY": "secret"}
    assert get_env_api_key("amazon-bedrock", env) == AMBIENT_AUTH_MARKER
    assert get_env_api_key("amazon-bedrock", {"AWS_ACCESS_KEY_ID": "id"}) is None


def test_builtin_catalog_reads():
    assert "anthropic" in get_builtin_providers()
    model = get_builtin_model("anthropic", "claude-haiku-4-5")
    assert model is not None
    assert model.api == "anthropic-messages"
    assert get_builtin_model("anthropic", "nope") is None
    assert get_builtin_models("anthropic")
    assert get_builtin_models("unknown") == []
    assert get_builtin_model_data_generated_at() is not None


def test_builtin_models_collection_wires_anthropic():
    models = builtin_models()
    provider = models.get_provider("anthropic")
    assert provider is not None
    assert models.get_model("anthropic", "claude-haiku-4-5") is not None


def test_builtin_models_registers_every_builtin_provider_with_models():
    models = builtin_models()
    providers = models.get_providers()

    assert len(providers) == len(builtin_providers())
    assert "anthropic" in [provider.id for provider in providers]
    assert models.get_model("anthropic", "claude-haiku-4-5").api == "anthropic-messages"
    assert len(models.get_models()) > 500

    for provider in providers:
        listed = models.get_models(provider.id)
        assert listed, provider.id
        assert all(model.provider == provider.id for model in listed)


def test_stores_native_constrained_sampling_capabilities_in_model_metadata():
    gpt4o = get_builtin_model("openai", "gpt-4o")
    assert gpt4o.compat.supports_strict_mode is True
    assert gpt4o.compat.supports_openai_grammar_tools is None

    gpt54 = get_builtin_model("openai", "gpt-5.4")
    assert gpt54.compat.supports_strict_mode is True
    assert gpt54.compat.supports_openai_grammar_tools is True

    assert get_builtin_model("anthropic", "claude-haiku-4-5").compat.supports_strict_tools is True


@pytest.mark.parametrize("provider", ["moonshotai", "moonshotai-cn"])
def test_uses_official_kimi_k3_pricing_for_moonshot_providers(provider):
    model = get_builtin_model(provider, "kimi-k3")

    assert model is not None
    assert model.cost == ModelCost(input=3, output=15, cache_read=0.3, cache_write=0)


@pytest.mark.parametrize(
    ("model_id", "cost"),
    [
        ("k3", ModelCost(input=3, output=15, cache_read=0.3, cache_write=0)),
        ("kimi-for-coding-highspeed", ModelCost(input=1.9, output=8, cache_read=0.38, cache_write=0)),
    ],
)
def test_uses_api_equivalent_implied_pricing_for_kimi_coding_subscription_models(model_id, cost):
    # models.dev reports zero cost for the subscription-backed catalog; the
    # generator substitutes the equivalent Moonshot API rates.
    model = get_builtin_model("kimi-coding", model_id)

    assert model is not None
    assert model.cost == cost


@pytest.mark.tonio
async def test_anthropic_env_auth_resolution():
    models = builtin_models()
    result = await models.get_auth("anthropic")
    # Depending on the host env this may or may not resolve; force it via overrides.
    from pidrei_ai.auth.resolve import AuthResolutionOverrides

    forced = await models.get_auth("anthropic", AuthResolutionOverrides(api_key="sk-test"))
    assert forced is not None
    assert forced.auth.api_key == "sk-test"
    assert result is None or result.auth.api_key
