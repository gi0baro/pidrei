"""Tests for env-api-keys and the builtin provider wiring.

Partial mirror of pi's providers.test.ts: its `envApiKeyAuth` and
`createProvider` describes live in test_registry.py, and the bedrock / cloudflare
/ vertex auth-flow cases join with those providers (PLAN.md Phase 5d).
"""

import pytest

from pidrei_ai.auth.types import AuthResult, ModelAuth
from pidrei_ai.env_api_keys import AMBIENT_AUTH_MARKER, find_env_keys, get_env_api_key
from pidrei_ai.providers.all import (
    builtin_models,
    builtin_providers,
    get_builtin_model,
    get_builtin_model_data_generated_at,
    get_builtin_models,
    get_builtin_providers,
)
from pidrei_ai.providers.anthropic import anthropic_provider
from pidrei_ai.registry import create_models
from pidrei_ai.types import ModelCost


def test_find_env_keys_uses_provider_env_overrides():
    assert find_env_keys("openai") is None or isinstance(find_env_keys("openai"), list)
    assert find_env_keys("openai", {"OPENAI_API_KEY": "k"}) == ["OPENAI_API_KEY"]
    assert find_env_keys("unknown-provider", {"WHATEVER": "x"}) is None


@pytest.mark.tonio
async def test_reports_anthropic_auth_token_but_preserves_oauth_token_api_key_lookup():
    env = {
        "ANTHROPIC_AUTH_TOKEN": "auth-token",
        "ANTHROPIC_OAUTH_TOKEN": "oauth-token",
        "ANTHROPIC_API_KEY": "api-key",
    }
    assert find_env_keys("anthropic", env) == ["ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_OAUTH_TOKEN", "ANTHROPIC_API_KEY"]
    assert await get_env_api_key("anthropic", env) == "oauth-token"


@pytest.mark.tonio
async def test_does_not_return_anthropic_auth_token_as_an_api_key():
    env = {"ANTHROPIC_AUTH_TOKEN": "auth-token"}
    assert find_env_keys("anthropic", env) == ["ANTHROPIC_AUTH_TOKEN"]
    assert await get_env_api_key("anthropic", env) is None


@pytest.mark.tonio
async def test_preserves_anthropic_oauth_token_as_an_api_key():
    env = {"ANTHROPIC_OAUTH_TOKEN": "oauth-token"}
    assert find_env_keys("anthropic", env) == ["ANTHROPIC_OAUTH_TOKEN"]
    assert await get_env_api_key("anthropic", env) == "oauth-token"


@pytest.mark.tonio
async def test_falls_back_to_anthropic_api_key_for_api_key_lookup():
    assert await get_env_api_key("anthropic", {"ANTHROPIC_API_KEY": "api-key"}) == "api-key"


@pytest.mark.tonio
async def test_bedrock_ambient_credentials_marker():
    env = {"AWS_ACCESS_KEY_ID": "id", "AWS_SECRET_ACCESS_KEY": "secret"}
    assert await get_env_api_key("amazon-bedrock", env) == AMBIENT_AUTH_MARKER
    assert await get_env_api_key("amazon-bedrock", {"AWS_ACCESS_KEY_ID": "id"}) is None


@pytest.mark.tonio
async def test_builtin_catalog_reads():
    assert "anthropic" in get_builtin_providers()
    model = get_builtin_model("anthropic", "claude-haiku-4-5")
    assert model is not None
    assert model.api == "anthropic-messages"
    assert get_builtin_model("anthropic", "nope") is None
    assert get_builtin_models("anthropic")
    assert get_builtin_models("unknown") == []
    assert await get_builtin_model_data_generated_at() is not None


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


class _FakeAuthContext:
    def __init__(self, env: dict[str, str]):
        self._env = env

    async def env(self, name: str) -> str | None:
        return self._env.get(name)

    async def file_exists(self, path: str) -> bool:
        return False


@pytest.mark.tonio
async def test_resolves_anthropic_bearer_auth_from_env_with_auth_token_precedence():
    models = create_models(
        auth_context=_FakeAuthContext(
            {
                "ANTHROPIC_AUTH_TOKEN": "auth-token",
                "ANTHROPIC_OAUTH_TOKEN": "oauth-token",
                "ANTHROPIC_API_KEY": "api-key",
            }
        )
    )
    models.set_provider(anthropic_provider())

    assert await models.get_auth("anthropic") == AuthResult(
        auth=ModelAuth(headers={"Authorization": "Bearer auth-token"}),
        source="ANTHROPIC_AUTH_TOKEN",
    )


@pytest.mark.tonio
async def test_preserves_anthropic_oauth_token_precedence_over_the_api_key():
    models = create_models(
        auth_context=_FakeAuthContext({"ANTHROPIC_API_KEY": "key", "ANTHROPIC_OAUTH_TOKEN": "oauth-token"})
    )
    models.set_provider(anthropic_provider())

    result = await models.get_auth("anthropic")
    assert result is not None
    assert result.auth.api_key == "oauth-token"
    assert result.source == "ANTHROPIC_OAUTH_TOKEN"
