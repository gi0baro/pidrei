"""Tests for env-api-keys and the builtin provider wiring."""

import pytest

from pppi_ai.env_api_keys import AMBIENT_AUTH_MARKER, find_env_keys, get_env_api_key
from pppi_ai.providers.all import (
    builtin_models,
    get_builtin_model,
    get_builtin_model_data_generated_at,
    get_builtin_models,
    get_builtin_providers,
)


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


@pytest.mark.tonio
async def test_anthropic_env_auth_resolution():
    models = builtin_models()
    result = await models.get_auth("anthropic")
    # Depending on the host env this may or may not resolve; force it via overrides.
    from pppi_ai.auth.resolve import AuthResolutionOverrides

    forced = await models.get_auth("anthropic", AuthResolutionOverrides(api_key="sk-test"))
    assert forced is not None
    assert forced.auth.api_key == "sk-test"
    assert result is None or result.auth.api_key
