"""Mirror of pi coding-agent test/model-registry.test.ts.

Provider substitutions (pidrei ships only the anthropic/openai builtins so
far): openrouter/google-based layer tests run against anthropic/openai or a
custom provider; the github-copilot OAuth model-filter test and the zai/
github-copilot display-name assertions are omitted (providers land in Phase
5). The compat-registry mutation test is N/A (no global compat registry).
The malformed-models.json assertions from config-value-migration.test.ts
live here too (its migration halves target unported legacy migrations).
"""

import json
import os

import pytest

from pidrei.core.auth_storage import AuthStorage, FileAuthStorageBackend
from pidrei.core.model_registry import ResolvedRequestAuth, clear_api_key_cache
from pidrei.core.provider_composer import AuthStatus, ExtensionOAuthConfig
from pidrei_ai.auth.types import ApiKeyCredential
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import get_supported_thinking_levels
from tests.model_runtime_helpers import create_model_registry


ANTHROPIC_IDS = [model.id for model in MODELS["anthropic"]]
OPENAI_IDS = [model.id for model in MODELS["openai"]]


@pytest.fixture
def registry_env(tmp_dir, request):
    request.addfinalizer(clear_api_key_cache)
    models_json_path = str(tmp_dir / "models.json")
    # `from_storage`, not `create`: the fixture is sync and cannot await, and
    # tmp_dir is fresh so there is no auth.json to load — same empty state.
    auth_storage = AuthStorage.from_storage(FileAuthStorageBackend(str(tmp_dir / "auth.json")))
    return tmp_dir, models_json_path, auth_storage


def provider_config(base_url, models, api="anthropic-messages"):
    return {
        "baseUrl": base_url,
        "apiKey": "test-key",
        "api": api,
        "models": [
            {
                "id": model["id"],
                "name": model.get("name", model["id"]),
                "reasoning": False,
                "input": ["text"],
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                "contextWindow": 100000,
                "maxTokens": 8000,
            }
            for model in models
        ],
    }


def write_models_json(path, providers):
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"providers": providers}))


def models_for_provider(registry, provider):
    return [model for model in registry.get_all() if model.provider == provider]


def to_sh_path(value):
    return value.replace("\\", "/").replace('"', '\\"')


def override_config(base_url, headers=None):
    config = {"baseUrl": base_url}
    if headers:
        config["headers"] = headers
    return config


class TestBaseUrlOverride:
    @pytest.mark.tonio
    async def test_overriding_base_url_keeps_all_built_in_models(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(models_json_path, {"anthropic": override_config("https://my-proxy.example.com/v1")})

        registry = await create_model_registry(auth_storage, models_json_path)
        anthropic_models = models_for_provider(registry, "anthropic")

        assert len(anthropic_models) > 1
        assert any("claude" in model.id for model in anthropic_models)

    @pytest.mark.tonio
    async def test_overriding_base_url_changes_url_on_all_built_in_models(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(models_json_path, {"anthropic": override_config("https://my-proxy.example.com/v1")})

        registry = await create_model_registry(auth_storage, models_json_path)

        for model in models_for_provider(registry, "anthropic"):
            assert model.base_url == "https://my-proxy.example.com/v1"

    @pytest.mark.tonio
    async def test_overriding_headers_resolves_at_request_time(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(
            models_json_path,
            {"anthropic": override_config("https://my-proxy.example.com/v1", {"X-Custom-Header": "custom-value"})},
        )

        registry = await create_model_registry(auth_storage, models_json_path)

        for model in models_for_provider(registry, "anthropic"):
            auth = await registry.get_api_key_and_headers(model)
            assert auth.ok is True
            assert auth.headers["X-Custom-Header"] == "custom-value"

    @pytest.mark.tonio
    async def test_headers_only_override_resolves_at_request_time(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(models_json_path, {"anthropic": {"headers": {"X-Custom-Header": "custom-value"}}})

        registry = await create_model_registry(auth_storage, models_json_path)
        assert registry.get_error() is None

        for model in models_for_provider(registry, "anthropic"):
            auth = await registry.get_api_key_and_headers(model)
            assert auth.ok is True
            assert auth.headers["X-Custom-Header"] == "custom-value"

    @pytest.mark.tonio
    async def test_unconfigured_compatibility_auth_includes_static_model_headers(self, registry_env):
        from dataclasses import replace

        _tmp, models_json_path, auth_storage = registry_env
        registry = await create_model_registry(auth_storage, models_json_path)
        base = registry.get_all()[0]
        model = replace(base, provider="missing-provider", headers={"X-Static-Model": "static-value"})

        auth = await registry.get_api_key_and_headers(model)

        assert auth == ResolvedRequestAuth(ok=True, headers={"X-Static-Model": "static-value"})

    @pytest.mark.tonio
    async def test_base_url_only_override_does_not_affect_other_providers(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(models_json_path, {"anthropic": override_config("https://my-proxy.example.com/v1")})

        registry = await create_model_registry(auth_storage, models_json_path)
        openai_models = models_for_provider(registry, "openai")

        assert len(openai_models) > 0
        assert openai_models[0].base_url != "https://my-proxy.example.com/v1"

    @pytest.mark.tonio
    async def test_can_mix_base_url_override_and_models_merge(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(
            models_json_path,
            {
                "anthropic": override_config("https://anthropic-proxy.example.com/v1"),
                "openai": provider_config(
                    "https://openai-proxy.example.com/v1", [{"id": "gpt-custom"}], "openai-completions"
                ),
            },
        )

        registry = await create_model_registry(auth_storage, models_json_path)

        anthropic_models = models_for_provider(registry, "anthropic")
        assert len(anthropic_models) > 1
        assert anthropic_models[0].base_url == "https://anthropic-proxy.example.com/v1"

        openai_models = models_for_provider(registry, "openai")
        assert len(openai_models) > 1
        assert any(model.id == "gpt-custom" for model in openai_models)

    @pytest.mark.tonio
    async def test_refresh_picks_up_base_url_override_changes(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(models_json_path, {"anthropic": override_config("https://first-proxy.example.com/v1")})
        registry = await create_model_registry(auth_storage, models_json_path)

        assert models_for_provider(registry, "anthropic")[0].base_url == "https://first-proxy.example.com/v1"

        write_models_json(models_json_path, {"anthropic": override_config("https://second-proxy.example.com/v1")})
        await registry.refresh()

        assert models_for_provider(registry, "anthropic")[0].base_url == "https://second-proxy.example.com/v1"


class TestCustomModelsMergeBehavior:
    @pytest.mark.tonio
    async def test_built_in_provider_custom_models_inherit_api_and_base_url_without_explicit_fields(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(
            models_json_path,
            {
                "anthropic": {
                    "models": [
                        {"id": "fake-provider/fake-model", "name": "Fake model", "reasoning": True, "input": ["text"]}
                    ]
                }
            },
        )

        registry = await create_model_registry(auth_storage, models_json_path)
        assert registry.get_error() is None

        model = registry.find("anthropic", "fake-provider/fake-model")
        assert model is not None
        assert model.api == "anthropic-messages"
        assert model.base_url == "https://api.anthropic.com"

    @pytest.mark.tonio
    async def test_non_built_in_provider_custom_models_still_require_base_url(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(
            models_json_path,
            {
                "my-custom-provider": {
                    "apiKey": "test-key",
                    "models": [{"id": "my-model", "api": "openai-completions", "reasoning": False, "input": ["text"]}],
                }
            },
        )

        registry = await create_model_registry(auth_storage, models_json_path)
        assert "baseUrl" in registry.get_error()

    @pytest.mark.tonio
    async def test_reports_every_provider_composition_error(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(
            models_json_path,
            {
                "broken-one": {"api": "openai-completions", "models": [{"id": "one"}]},
                "broken-two": {"api": "openai-completions", "models": [{"id": "two"}]},
            },
        )

        registry = await create_model_registry(auth_storage, models_json_path)
        error = registry.get_error()

        assert 'Provider "broken-one"' in error
        assert 'Provider "broken-two"' in error

    @pytest.mark.tonio
    async def test_custom_provider_with_same_name_as_built_in_merges_with_built_in_models(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(
            models_json_path,
            {"anthropic": provider_config("https://my-proxy.example.com/v1", [{"id": "claude-custom"}])},
        )

        registry = await create_model_registry(auth_storage, models_json_path)
        anthropic_models = models_for_provider(registry, "anthropic")

        assert len(anthropic_models) > 1
        assert any(model.id == "claude-custom" for model in anthropic_models)
        assert any("claude" in model.id and model.id != "claude-custom" for model in anthropic_models)

    @pytest.mark.tonio
    async def test_custom_model_with_same_id_replaces_built_in_model_by_id(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        builtin_id = ANTHROPIC_IDS[0]
        write_models_json(
            models_json_path,
            {"anthropic": provider_config("https://my-proxy.example.com/v1", [{"id": builtin_id}])},
        )

        registry = await create_model_registry(auth_storage, models_json_path)
        matching = [model for model in models_for_provider(registry, "anthropic") if model.id == builtin_id]

        assert len(matching) == 1
        assert matching[0].base_url == "https://my-proxy.example.com/v1"

    @pytest.mark.tonio
    async def test_custom_provider_with_same_name_as_built_in_does_not_affect_other_built_in_providers(
        self, registry_env
    ):
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(
            models_json_path,
            {"anthropic": provider_config("https://my-proxy.example.com/v1", [{"id": "claude-custom"}])},
        )

        registry = await create_model_registry(auth_storage, models_json_path)

        assert len(models_for_provider(registry, "openai")) > 0

    @pytest.mark.tonio
    async def test_provider_level_base_url_applies_to_both_built_in_and_custom_models(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(
            models_json_path,
            {"anthropic": provider_config("https://merged-proxy.example.com/v1", [{"id": "claude-custom"}])},
        )

        registry = await create_model_registry(auth_storage, models_json_path)

        for model in models_for_provider(registry, "anthropic"):
            assert model.base_url == "https://merged-proxy.example.com/v1"

    @pytest.mark.tonio
    async def test_provider_level_compat_applies_to_custom_models(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(
            models_json_path,
            {
                "demo": {
                    "baseUrl": "https://example.com/v1",
                    "apiKey": "DEMO_KEY",
                    "api": "openai-completions",
                    "compat": {"supportsUsageInStreaming": False, "maxTokensField": "max_tokens"},
                    "models": [
                        {
                            "id": "demo-model",
                            "reasoning": False,
                            "input": ["text"],
                            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                            "contextWindow": 1000,
                            "maxTokens": 100,
                        }
                    ],
                }
            },
        )

        registry = await create_model_registry(auth_storage, models_json_path)
        compat = registry.find("demo", "demo-model").compat

        assert compat.supports_usage_in_streaming is False
        assert compat.max_tokens_field == "max_tokens"

    @pytest.mark.tonio
    async def test_model_level_compat_overrides_provider_level_compat_for_custom_models(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(
            models_json_path,
            {
                "demo": {
                    "baseUrl": "https://example.com/v1",
                    "apiKey": "DEMO_KEY",
                    "api": "openai-completions",
                    "compat": {"supportsUsageInStreaming": False, "maxTokensField": "max_tokens"},
                    "models": [
                        {
                            "id": "demo-model",
                            "reasoning": False,
                            "input": ["text"],
                            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                            "contextWindow": 1000,
                            "maxTokens": 100,
                            "compat": {"supportsUsageInStreaming": True, "maxTokensField": "max_completion_tokens"},
                        }
                    ],
                }
            },
        )

        registry = await create_model_registry(auth_storage, models_json_path)
        compat = registry.find("demo", "demo-model").compat

        assert compat.supports_usage_in_streaming is True
        assert compat.max_tokens_field == "max_completion_tokens"

    @pytest.mark.tonio
    async def test_provider_level_compat_applies_to_built_in_models(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(
            models_json_path,
            {"anthropic": {"compat": {"supportsTemperature": False, "supportsCacheControlOnTools": False}}},
        )

        registry = await create_model_registry(auth_storage, models_json_path)
        models = models_for_provider(registry, "anthropic")

        assert len(models) > 0
        for model in models:
            assert model.compat.supports_temperature is False
            assert model.compat.supports_cache_control_on_tools is False

    @pytest.mark.tonio
    async def test_model_schema_accepts_thinking_level_map_strict_mode_and_cache_control_format(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(
            models_json_path,
            {
                "demo": {
                    "baseUrl": "https://example.com/v1",
                    "apiKey": "DEMO_KEY",
                    "api": "openai-completions",
                    "models": [
                        {
                            "id": "demo-model",
                            "reasoning": True,
                            "input": ["text"],
                            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                            "contextWindow": 1000,
                            "maxTokens": 100,
                            "thinkingLevelMap": {"minimal": None, "high": "max"},
                            "compat": {"supportsStrictMode": False, "cacheControlFormat": "anthropic"},
                        }
                    ],
                }
            },
        )

        registry = await create_model_registry(auth_storage, models_json_path)
        model = registry.find("demo", "demo-model")

        assert registry.get_error() is None
        assert model.thinking_level_map == {"minimal": None, "high": "max"}
        assert model.compat.supports_strict_mode is False
        assert model.compat.cache_control_format == "anthropic"

    @pytest.mark.tonio
    async def test_compat_schema_accepts_chat_template_thinking_configuration(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(
            models_json_path,
            {
                "demo": {
                    "baseUrl": "https://example.com/v1",
                    "apiKey": "DEMO_KEY",
                    "api": "openai-completions",
                    "models": [
                        {
                            "id": "demo-model",
                            "reasoning": True,
                            "input": ["text"],
                            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                            "contextWindow": 1000,
                            "maxTokens": 100,
                            "compat": {
                                "thinkingFormat": "chat-template",
                                "chatTemplateKwargs": {
                                    "preserve_thinking": True,
                                    "thinking": {"$var": "thinking.enabled"},
                                },
                            },
                        }
                    ],
                }
            },
        )

        registry = await create_model_registry(auth_storage, models_json_path)
        compat = registry.find("demo", "demo-model").compat

        assert registry.get_error() is None
        assert compat.thinking_format == "chat-template"
        assert compat.chat_template_kwargs == {"preserve_thinking": True, "thinking": {"$var": "thinking.enabled"}}

    @pytest.mark.tonio
    async def test_compat_schema_accepts_anthropic_eager_tool_input_streaming_flag(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(
            models_json_path,
            {
                "demo": {
                    "baseUrl": "https://example.com",
                    "apiKey": "DEMO_KEY",
                    "api": "anthropic-messages",
                    "compat": {"supportsEagerToolInputStreaming": False},
                    "models": [
                        {
                            "id": "demo-model",
                            "reasoning": True,
                            "input": ["text"],
                            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                            "contextWindow": 1000,
                            "maxTokens": 100,
                        }
                    ],
                }
            },
        )

        registry = await create_model_registry(auth_storage, models_json_path)
        compat = registry.find("demo", "demo-model").compat

        assert registry.get_error() is None
        assert compat.supports_eager_tool_input_streaming is False

    @pytest.mark.tonio
    async def test_compat_schema_accepts_long_cache_retention_flag(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(
            models_json_path,
            {
                "demo": {
                    "baseUrl": "https://example.com",
                    "apiKey": "DEMO_KEY",
                    "api": "anthropic-messages",
                    "compat": {"supportsLongCacheRetention": False},
                    "models": [
                        {
                            "id": "demo-model",
                            "reasoning": True,
                            "input": ["text"],
                            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                            "contextWindow": 1000,
                            "maxTokens": 100,
                        }
                    ],
                }
            },
        )

        registry = await create_model_registry(auth_storage, models_json_path)
        compat = registry.find("demo", "demo-model").compat

        assert registry.get_error() is None
        assert compat.supports_long_cache_retention is False

    @pytest.mark.tonio
    async def test_model_level_base_url_overrides_provider_level_base_url_for_custom_models(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(
            models_json_path,
            {
                "opencode-go": {
                    "baseUrl": "https://opencode.ai/zen/go/v1",
                    "apiKey": "TEST_KEY",
                    "models": [
                        {
                            "id": "minimax-m2.5",
                            "api": "anthropic-messages",
                            "baseUrl": "https://opencode.ai/zen/go",
                            "reasoning": True,
                            "input": ["text"],
                            "cost": {"input": 0.3, "output": 1.2, "cacheRead": 0.03, "cacheWrite": 0},
                            "contextWindow": 204800,
                            "maxTokens": 131072,
                        },
                        {
                            "id": "glm-5",
                            "api": "openai-completions",
                            "reasoning": True,
                            "input": ["text"],
                            "cost": {"input": 1, "output": 3.2, "cacheRead": 0.2, "cacheWrite": 0},
                            "contextWindow": 204800,
                            "maxTokens": 131072,
                        },
                    ],
                }
            },
        )

        registry = await create_model_registry(auth_storage, models_json_path)

        assert registry.find("opencode-go", "minimax-m2.5").base_url == "https://opencode.ai/zen/go"
        assert registry.find("opencode-go", "glm-5").base_url == "https://opencode.ai/zen/go/v1"

    @pytest.mark.tonio
    async def test_model_overrides_still_apply_when_provider_also_defines_models(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        builtin_id = ANTHROPIC_IDS[0]
        write_models_json(
            models_json_path,
            {
                "anthropic": {
                    "baseUrl": "https://my-proxy.example.com/v1",
                    "apiKey": "ANTHROPIC_API_KEY",
                    "api": "anthropic-messages",
                    "models": [
                        {
                            "id": "custom/anthropic-model",
                            "name": "Custom Anthropic Model",
                            "reasoning": False,
                            "input": ["text"],
                            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                            "contextWindow": 128000,
                            "maxTokens": 16384,
                        }
                    ],
                    "modelOverrides": {builtin_id: {"name": "Overridden Built-in Model"}},
                }
            },
        )

        registry = await create_model_registry(auth_storage, models_json_path)
        models = models_for_provider(registry, "anthropic")

        assert any(model.id == "custom/anthropic-model" for model in models)
        assert any(model.id == builtin_id and model.name == "Overridden Built-in Model" for model in models)

    @pytest.mark.tonio
    async def test_refresh_reloads_merged_custom_models_from_disk(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(
            models_json_path,
            {"anthropic": provider_config("https://first-proxy.example.com/v1", [{"id": "claude-custom"}])},
        )
        registry = await create_model_registry(auth_storage, models_json_path)
        assert any(model.id == "claude-custom" for model in models_for_provider(registry, "anthropic"))

        write_models_json(
            models_json_path,
            {"anthropic": provider_config("https://second-proxy.example.com/v1", [{"id": "claude-custom-2"}])},
        )
        await registry.refresh()

        anthropic_models = models_for_provider(registry, "anthropic")
        assert not any(model.id == "claude-custom" for model in anthropic_models)
        assert any(model.id == "claude-custom-2" for model in anthropic_models)
        assert any("claude" in model.id for model in anthropic_models)

    @pytest.mark.tonio
    async def test_removing_custom_models_from_models_json_keeps_built_in_provider_models(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(
            models_json_path,
            {"anthropic": provider_config("https://proxy.example.com/v1", [{"id": "claude-custom"}])},
        )
        registry = await create_model_registry(auth_storage, models_json_path)
        assert any(model.id == "claude-custom" for model in models_for_provider(registry, "anthropic"))

        write_models_json(models_json_path, {})
        await registry.refresh()

        anthropic_models = models_for_provider(registry, "anthropic")
        assert not any(model.id == "claude-custom" for model in anthropic_models)
        assert any("claude" in model.id for model in anthropic_models)


class TestModelOverrides:
    @pytest.mark.tonio
    async def test_model_override_applies_to_a_single_built_in_model(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        first_id, second_id = ANTHROPIC_IDS[0], ANTHROPIC_IDS[1]
        write_models_json(
            models_json_path, {"anthropic": {"modelOverrides": {first_id: {"name": "Custom Sonnet Name"}}}}
        )

        registry = await create_model_registry(auth_storage, models_json_path)
        models = models_for_provider(registry, "anthropic")

        assert next(model for model in models if model.id == first_id).name == "Custom Sonnet Name"
        assert next(model for model in models if model.id == second_id).name != "Custom Sonnet Name"

    @pytest.mark.tonio
    async def test_model_override_with_compat_open_router_routing(self, registry_env):
        """Adapted: openRouterRouting is a completions-compat key, exercised on a
        custom openai-completions provider (no openrouter builtin yet)."""
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(
            models_json_path,
            {
                "demo": {
                    **provider_config("https://example.com/v1", [{"id": "demo-model"}], "openai-completions"),
                    "compat": {"openRouterRouting": {"allow_fallbacks": True}},
                    "modelOverrides": {"demo-model": {"compat": {"openRouterRouting": {"only": ["amazon-bedrock"]}}}},
                }
            },
        )

        registry = await create_model_registry(auth_storage, models_json_path)
        compat = registry.find("demo", "demo-model").compat

        # Deep merge: provider-level routing keys survive under the override.
        assert compat.open_router_routing == {"allow_fallbacks": True, "only": ["amazon-bedrock"]}

    @pytest.mark.tonio
    async def test_multiple_model_overrides_on_same_provider(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        first_id, second_id = ANTHROPIC_IDS[0], ANTHROPIC_IDS[1]
        write_models_json(
            models_json_path,
            {
                "anthropic": {
                    "modelOverrides": {
                        first_id: {"name": "First Override"},
                        second_id: {"name": "Second Override"},
                    }
                }
            },
        )

        registry = await create_model_registry(auth_storage, models_json_path)
        models = models_for_provider(registry, "anthropic")

        assert next(model for model in models if model.id == first_id).name == "First Override"
        assert next(model for model in models if model.id == second_id).name == "Second Override"

    @pytest.mark.tonio
    async def test_model_override_combined_with_base_url_override(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        first_id, second_id = ANTHROPIC_IDS[0], ANTHROPIC_IDS[1]
        write_models_json(
            models_json_path,
            {
                "anthropic": {
                    "baseUrl": "https://my-proxy.example.com/v1",
                    "modelOverrides": {first_id: {"name": "Proxied Model"}},
                }
            },
        )

        registry = await create_model_registry(auth_storage, models_json_path)
        models = models_for_provider(registry, "anthropic")
        first = next(model for model in models if model.id == first_id)
        second = next(model for model in models if model.id == second_id)

        assert first.base_url == "https://my-proxy.example.com/v1"
        assert first.name == "Proxied Model"
        assert second.base_url == "https://my-proxy.example.com/v1"
        assert second.name != "Proxied Model"

    @pytest.mark.tonio
    async def test_model_override_for_non_existent_model_id_is_ignored(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(
            models_json_path,
            {"anthropic": {"modelOverrides": {"nonexistent/model-id": {"name": "This should not appear"}}}},
        )

        registry = await create_model_registry(auth_storage, models_json_path)
        models = models_for_provider(registry, "anthropic")

        assert not any(model.id == "nonexistent/model-id" for model in models)
        assert registry.get_error() is None

    @pytest.mark.tonio
    async def test_model_override_can_change_cost_fields_partially(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        first_id = ANTHROPIC_IDS[0]
        write_models_json(models_json_path, {"anthropic": {"modelOverrides": {first_id: {"cost": {"input": 99}}}}})

        registry = await create_model_registry(auth_storage, models_json_path)
        model = next(entry for entry in models_for_provider(registry, "anthropic") if entry.id == first_id)

        assert model.cost.input == 99
        assert model.cost.output > 0

    @pytest.mark.tonio
    async def test_model_override_can_add_headers_at_request_time(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        first_id = ANTHROPIC_IDS[0]
        write_models_json(
            models_json_path,
            {"anthropic": {"modelOverrides": {first_id: {"headers": {"X-Custom-Model-Header": "value"}}}}},
        )

        registry = await create_model_registry(auth_storage, models_json_path)
        model = next(entry for entry in models_for_provider(registry, "anthropic") if entry.id == first_id)

        auth = await registry.get_api_key_and_headers(model)
        assert auth.ok is True
        assert auth.headers["X-Custom-Model-Header"] == "value"

    @pytest.mark.tonio
    async def test_refresh_picks_up_model_override_changes(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        first_id = ANTHROPIC_IDS[0]
        write_models_json(models_json_path, {"anthropic": {"modelOverrides": {first_id: {"name": "First Name"}}}})

        registry = await create_model_registry(auth_storage, models_json_path)
        assert (
            next(entry for entry in models_for_provider(registry, "anthropic") if entry.id == first_id).name
            == "First Name"
        )

        write_models_json(models_json_path, {"anthropic": {"modelOverrides": {first_id: {"name": "Second Name"}}}})
        await registry.refresh()

        assert (
            next(entry for entry in models_for_provider(registry, "anthropic") if entry.id == first_id).name
            == "Second Name"
        )

    @pytest.mark.tonio
    async def test_removing_model_override_restores_built_in_values(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        first_id = ANTHROPIC_IDS[0]
        write_models_json(models_json_path, {"anthropic": {"modelOverrides": {first_id: {"name": "Custom Name"}}}})

        registry = await create_model_registry(auth_storage, models_json_path)
        assert (
            next(entry for entry in models_for_provider(registry, "anthropic") if entry.id == first_id).name
            == "Custom Name"
        )

        write_models_json(models_json_path, {})
        await registry.refresh()

        assert (
            next(entry for entry in models_for_provider(registry, "anthropic") if entry.id == first_id).name
            != "Custom Name"
        )


def demo_models_entry(id="demo-model", **overrides):
    entry = {
        "id": id,
        "name": "Demo Model",
        "reasoning": False,
        "input": ["text"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 128000,
        "maxTokens": 4096,
    }
    entry.update(overrides)
    return entry


class TestDynamicProviderLifecycle:
    @pytest.mark.tonio
    async def test_get_provider_display_name_resolves_registered_oauth_built_in_and_fallback_names(self, registry_env):
        import time

        _tmp, models_json_path, auth_storage = registry_env
        registry = await create_model_registry(auth_storage, models_json_path)

        assert registry.get_provider_display_name("openai") == "OpenAI"
        assert registry.get_provider_display_name("anthropic") == "Anthropic"
        assert registry.get_provider_display_name("unknown-provider") == "unknown-provider"

        registry.register_provider(
            "named-provider",
            {
                "name": "Named Provider",
                "baseUrl": "https://provider.test/v1",
                "apiKey": "test-key",
                "api": "openai-completions",
                "models": [demo_models_entry()],
            },
        )
        assert registry.get_provider_display_name("named-provider") == "Named Provider"

        async def login(_callbacks):
            return {"access": "access", "refresh": "refresh", "expires": int(time.time() * 1000) + 60_000}

        async def refresh_token(credentials):
            return credentials

        registry.register_provider(
            "oauth-provider",
            {
                "baseUrl": "https://provider.test/v1",
                "api": "openai-completions",
                "oauth": ExtensionOAuthConfig(
                    name="OAuth Provider",
                    login=login,
                    refresh_token=refresh_token,
                    get_api_key=lambda credentials: credentials.access,
                ),
                "models": [demo_models_entry()],
            },
        )
        assert registry.get_provider_display_name("oauth-provider") == "OAuth Provider"

    @pytest.mark.tonio
    async def test_model_overrides_apply_to_dynamically_registered_provider_models(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(
            models_json_path,
            {
                "extension-provider": {
                    "modelOverrides": {
                        "extension-model": {
                            "name": "Overridden Extension Model",
                            "thinkingLevelMap": {
                                "off": None,
                                "minimal": None,
                                "low": None,
                                "medium": None,
                                "xhigh": "max",
                            },
                            "headers": {"x-model-override": "enabled"},
                        }
                    }
                }
            },
        )

        registry = await create_model_registry(auth_storage, models_json_path)
        registry.register_provider(
            "extension-provider",
            {
                "baseUrl": "https://provider.test/v1",
                "apiKey": "test-key",
                "api": "openai-completions",
                "models": [demo_models_entry("extension-model", name="Extension Model", reasoning=True)],
            },
        )

        model = registry.find("extension-provider", "extension-model")
        assert model is not None
        assert model.name == "Overridden Extension Model"
        assert model.thinking_level_map == {"off": None, "minimal": None, "low": None, "medium": None, "xhigh": "max"}
        assert get_supported_thinking_levels(model) == ["high", "xhigh"]
        auth = await registry.get_api_key_and_headers(model)
        assert auth.ok is True
        assert auth.headers["x-model-override"] == "enabled"

    @pytest.mark.tonio
    async def test_stored_api_key_env_propagates_to_request_auth_and_resolves_headers(self, registry_env):
        """Adapted from pi's cloudflare-ai-gateway variant: a custom provider
        exercises the same stored-credential env propagation (the cf builtin
        lands in Phase 5). Unlike cf's builtin auth, the custom provider
        resolves the key into apiKey and keeps all env values."""
        _tmp, models_json_path, auth_storage = registry_env

        async def set_credential(_current):
            return ApiKeyCredential(
                key="$CLOUDFLARE_API_KEY",
                env={
                    "CLOUDFLARE_API_KEY": "stored-cf-token",
                    "CLOUDFLARE_ACCOUNT_ID": "stored-account",
                    "CLOUDFLARE_GATEWAY_ID": "stored-gateway",
                },
            )

        await auth_storage.modify("gateway-provider", set_credential)
        write_models_json(
            models_json_path,
            {
                "gateway-provider": {
                    **provider_config("https://gateway.test/v1", [{"id": "gw-model"}], "openai-completions"),
                    "headers": {"x-account": "$CLOUDFLARE_ACCOUNT_ID"},
                }
            },
        )

        registry = await create_model_registry(auth_storage, models_json_path)
        model = registry.find("gateway-provider", "gw-model")
        assert model is not None

        auth = await registry.get_api_key_and_headers(model)

        assert auth == ResolvedRequestAuth(
            ok=True,
            api_key="stored-cf-token",
            headers={"x-account": "stored-account"},
            env={
                "CLOUDFLARE_API_KEY": "stored-cf-token",
                "CLOUDFLARE_ACCOUNT_ID": "stored-account",
                "CLOUDFLARE_GATEWAY_ID": "stored-gateway",
            },
        )

    @pytest.mark.tonio
    async def test_register_provider_treats_uppercase_api_key_and_headers_as_literals(self, registry_env):
        import contextlib
        import os as os_module

        _tmp, models_json_path, auth_storage = registry_env
        env_keys = ["CUSTOM_NAME", "BEARER", "MODEL_TOKEN"]
        saved = {key: os_module.environ.get(key) for key in env_keys}
        for key in env_keys:
            os_module.environ[key] = f"env-{key}"

        try:
            registry = await create_model_registry(auth_storage, models_json_path)

            registry.register_provider(
                "literal-provider",
                {
                    **provider_config("https://provider.test/v1", [{"id": "demo-model"}], "openai-completions"),
                    "apiKey": "CUSTOM_NAME",
                    "headers": {"Authorization": "BEARER"},
                    "models": [
                        demo_models_entry("demo-model", name="demo-model", headers={"x-model-token": "MODEL_TOKEN"})
                    ],
                },
            )

            assert await registry.get_api_key_for_provider("literal-provider") == "CUSTOM_NAME"
            model = registry.find("literal-provider", "demo-model")
            assert model is not None
            auth = await registry.get_api_key_and_headers(model)
            assert auth.ok is True
            assert auth.api_key == "CUSTOM_NAME"
            assert auth.headers == {"Authorization": "BEARER", "x-model-token": "MODEL_TOKEN"}
        finally:
            for key in env_keys:
                if saved[key] is None:
                    with contextlib.suppress(KeyError):
                        del os_module.environ[key]
                else:
                    os_module.environ[key] = saved[key]

    @pytest.mark.tonio
    async def test_failed_register_provider_does_not_persist_invalid_stream_simple_config(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        registry = await create_model_registry(auth_storage, models_json_path)

        def broken_stream(*_args):
            raise Exception("should not run")

        with pytest.raises(
            Exception, match='Provider broken-provider: "api" is required when registering streamSimple.'
        ):
            registry.register_provider("broken-provider", {"streamSimple": broken_stream})

        await registry.refresh()

    @pytest.mark.tonio
    async def test_failed_register_provider_does_not_remove_existing_provider_models(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        registry = await create_model_registry(auth_storage, models_json_path)

        registry.register_provider(
            "demo-provider",
            {
                "baseUrl": "https://provider.test/v1",
                "apiKey": "test-key",
                "api": "openai-completions",
                "models": [demo_models_entry()],
            },
        )

        assert registry.find("demo-provider", "demo-model") is not None

        with pytest.raises(Exception, match='Provider demo-provider, model broken-model: no "api" specified.'):
            registry.register_provider(
                "demo-provider",
                {
                    "baseUrl": "https://provider.test/v2",
                    "apiKey": "test-key",
                    "models": [demo_models_entry("broken-model", name="Broken Model")],
                },
            )

        assert registry.find("demo-provider", "demo-model") is not None
        await registry.refresh()
        assert registry.find("demo-provider", "demo-model") is not None

    @pytest.mark.tonio
    async def test_unregister_provider_removes_the_runtime_oauth_overlay_without_mutating_global_state(
        self, registry_env
    ):
        import time

        _tmp, models_json_path, auth_storage = registry_env
        registry = await create_model_registry(auth_storage, models_json_path)

        async def login(_callbacks):
            return {
                "access": "custom-access-token",
                "refresh": "custom-refresh-token",
                "expires": int(time.time() * 1000) + 60_000,
            }

        async def refresh_token(credentials):
            return credentials

        registry.register_provider(
            "anthropic",
            {
                "oauth": ExtensionOAuthConfig(
                    name="Custom Anthropic OAuth",
                    login=login,
                    refresh_token=refresh_token,
                    get_api_key=lambda credentials: credentials.access,
                )
            },
        )

        assert registry.get_registered_provider_config("anthropic")["oauth"].name == "Custom Anthropic OAuth"

        registry.unregister_provider("anthropic")

        assert registry.get_registered_provider_config("anthropic") is None


class TestDynamicProviderOverridePersistence:
    @pytest.mark.tonio
    async def test_base_url_only_override_keeps_built_in_provider_models_after_refresh(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        registry = await create_model_registry(auth_storage, models_json_path)

        registry.register_provider("anthropic", {"baseUrl": "https://proxy.test/anthropic"})
        await registry.refresh()

        anthropic_models = models_for_provider(registry, "anthropic")
        assert len(anthropic_models) > 1
        assert all(model.base_url == "https://proxy.test/anthropic" for model in anthropic_models)

    @pytest.mark.tonio
    async def test_models_only_override_replaces_built_in_provider_models_after_refresh(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        registry = await create_model_registry(auth_storage, models_json_path)

        registry.register_provider(
            "anthropic",
            {
                **provider_config("https://custom.test/anthropic", [{"id": "custom-claude"}], "anthropic-messages"),
                "baseUrl": "https://custom.test/anthropic",
            },
        )
        await registry.refresh()

        assert [model.id for model in models_for_provider(registry, "anthropic")] == ["custom-claude"]
        assert registry.find("anthropic", "custom-claude").base_url == "https://custom.test/anthropic"

    @pytest.mark.tonio
    async def test_models_plus_base_url_override_replaces_built_in_provider_models_after_refresh(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        registry = await create_model_registry(auth_storage, models_json_path)

        registry.register_provider(
            "anthropic",
            {
                **provider_config("https://custom.test/anthropic", [{"id": "custom-claude"}], "anthropic-messages"),
                "baseUrl": "https://custom.test/anthropic",
            },
        )
        registry.register_provider("anthropic", {"baseUrl": "https://proxy.test/anthropic"})
        await registry.refresh()

        assert [model.id for model in models_for_provider(registry, "anthropic")] == ["custom-claude"]
        assert registry.find("anthropic", "custom-claude").base_url == "https://proxy.test/anthropic"

    @pytest.mark.tonio
    async def test_models_only_custom_provider_registration_survives_refresh(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        registry = await create_model_registry(auth_storage, models_json_path)

        registry.register_provider(
            "custom-provider",
            provider_config("https://custom.test/v1", [{"id": "custom-a"}, {"id": "custom-b"}], "openai-completions"),
        )
        await registry.refresh()

        assert [model.id for model in models_for_provider(registry, "custom-provider")] == ["custom-a", "custom-b"]

    @pytest.mark.tonio
    async def test_base_url_only_override_keeps_custom_provider_models_after_refresh(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        registry = await create_model_registry(auth_storage, models_json_path)

        registry.register_provider(
            "custom-provider",
            provider_config("https://custom.test/v1", [{"id": "custom-a"}, {"id": "custom-b"}], "openai-completions"),
        )
        registry.register_provider("custom-provider", {"baseUrl": "https://proxy.test/custom"})
        await registry.refresh()

        models = models_for_provider(registry, "custom-provider")
        assert [model.id for model in models] == ["custom-a", "custom-b"]
        assert all(model.base_url == "https://proxy.test/custom" for model in models)

    @pytest.mark.tonio
    async def test_headers_only_override_keeps_custom_provider_models_after_refresh(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        registry = await create_model_registry(auth_storage, models_json_path)

        registry.register_provider(
            "custom-provider",
            provider_config("https://custom.test/v1", [{"id": "custom-a"}, {"id": "custom-b"}], "openai-completions"),
        )
        registry.register_provider("custom-provider", {"headers": {"x-proxy": "enabled"}})
        await registry.refresh()

        models = models_for_provider(registry, "custom-provider")
        assert [model.id for model in models] == ["custom-a", "custom-b"]
        assert all(model.base_url == "https://custom.test/v1" for model in models)
        auth = await registry.get_api_key_and_headers(models[0])
        assert auth.ok is True
        assert auth.headers["x-proxy"] == "enabled"


def provider_with_api_key(api_key):
    return {
        "baseUrl": "https://example.com/v1",
        "apiKey": api_key,
        "api": "anthropic-messages",
        "models": [
            {
                "id": "test-model",
                "name": "Test Model",
                "reasoning": False,
                "input": ["text"],
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                "contextWindow": 100000,
                "maxTokens": 8000,
            }
        ],
    }


class TestApiKeyResolution:
    @pytest.mark.tonio
    async def test_api_key_with_bang_prefix_executes_command_and_uses_stdout(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(
            models_json_path, {"custom-provider": provider_with_api_key("!echo test-api-key-from-command")}
        )
        registry = await create_model_registry(auth_storage, models_json_path)
        assert await registry.get_api_key_for_provider("custom-provider") == "test-api-key-from-command"

    @pytest.mark.tonio
    async def test_api_key_with_bang_prefix_trims_whitespace_from_command_output(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(models_json_path, {"custom-provider": provider_with_api_key("!echo '  spaced-key  '")})
        registry = await create_model_registry(auth_storage, models_json_path)
        assert await registry.get_api_key_for_provider("custom-provider") == "spaced-key"

    @pytest.mark.tonio
    async def test_api_key_with_bang_prefix_handles_multiline_output(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(models_json_path, {"custom-provider": provider_with_api_key("!printf 'line1\\nline2'")})
        registry = await create_model_registry(auth_storage, models_json_path)
        assert await registry.get_api_key_for_provider("custom-provider") == "line1\nline2"

    @pytest.mark.tonio
    async def test_api_key_with_bang_prefix_returns_none_on_command_failure(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(models_json_path, {"custom-provider": provider_with_api_key("!exit 1")})
        registry = await create_model_registry(auth_storage, models_json_path)
        assert await registry.get_api_key_for_provider("custom-provider") is None

    @pytest.mark.tonio
    async def test_api_key_with_bang_prefix_returns_none_on_nonexistent_command(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(models_json_path, {"custom-provider": provider_with_api_key("!nonexistent-command-12345")})
        registry = await create_model_registry(auth_storage, models_json_path)
        assert await registry.get_api_key_for_provider("custom-provider") is None

    @pytest.mark.tonio
    async def test_api_key_with_bang_prefix_returns_none_on_empty_output(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(models_json_path, {"custom-provider": provider_with_api_key("!printf ''")})
        registry = await create_model_registry(auth_storage, models_json_path)
        assert await registry.get_api_key_for_provider("custom-provider") is None

    @pytest.mark.tonio
    async def test_api_key_with_dollar_prefix_resolves_to_env_value(self, registry_env):
        import contextlib

        _tmp, models_json_path, auth_storage = registry_env
        original = os.environ.get("TEST_API_KEY_12345")
        os.environ["TEST_API_KEY_12345"] = "env-api-key-value"
        try:
            write_models_json(models_json_path, {"custom-provider": provider_with_api_key("$TEST_API_KEY_12345")})
            registry = await create_model_registry(auth_storage, models_json_path)
            assert await registry.get_api_key_for_provider("custom-provider") == "env-api-key-value"
        finally:
            if original is None:
                with contextlib.suppress(KeyError):
                    del os.environ["TEST_API_KEY_12345"]
            else:
                os.environ["TEST_API_KEY_12345"] = original

    @pytest.mark.tonio
    async def test_api_key_with_braced_env_syntax_resolves_to_env_value(self, registry_env):
        import contextlib

        _tmp, models_json_path, auth_storage = registry_env
        original = os.environ.get("TEST_BRACED_API_KEY_12345")
        os.environ["TEST_BRACED_API_KEY_12345"] = "braced-env-api-key-value"
        try:
            write_models_json(
                models_json_path, {"custom-provider": provider_with_api_key("${TEST_BRACED_API_KEY_12345}")}
            )
            registry = await create_model_registry(auth_storage, models_json_path)
            assert await registry.get_api_key_for_provider("custom-provider") == "braced-env-api-key-value"
        finally:
            if original is None:
                with contextlib.suppress(KeyError):
                    del os.environ["TEST_BRACED_API_KEY_12345"]
            else:
                os.environ["TEST_BRACED_API_KEY_12345"] = original

    @pytest.mark.tonio
    async def test_api_key_interpolates_braced_env_references_inside_literals(self, registry_env):
        import contextlib

        _tmp, models_json_path, auth_storage = registry_env
        saved_a = os.environ.get("TEST_INTERPOLATED_PART_A_12345")
        saved_b = os.environ.get("TEST_INTERPOLATED_PART_B_12345")
        os.environ["TEST_INTERPOLATED_PART_A_12345"] = "left"
        os.environ["TEST_INTERPOLATED_PART_B_12345"] = "right"
        try:
            write_models_json(
                models_json_path,
                {
                    "custom-provider": provider_with_api_key(
                        "${TEST_INTERPOLATED_PART_A_12345}_${TEST_INTERPOLATED_PART_B_12345}"
                    )
                },
            )
            registry = await create_model_registry(auth_storage, models_json_path)
            assert await registry.get_api_key_for_provider("custom-provider") == "left_right"
        finally:
            for key, saved in (
                ("TEST_INTERPOLATED_PART_A_12345", saved_a),
                ("TEST_INTERPOLATED_PART_B_12345", saved_b),
            ):
                if saved is None:
                    with contextlib.suppress(KeyError):
                        del os.environ[key]
                else:
                    os.environ[key] = saved

    @pytest.mark.tonio
    async def test_api_key_with_double_dollar_prefix_escapes_a_leading_dollar(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(models_json_path, {"custom-provider": provider_with_api_key("$$TEST_API_KEY_12345")})
        registry = await create_model_registry(auth_storage, models_json_path)
        assert await registry.get_api_key_for_provider("custom-provider") == "$TEST_API_KEY_12345"

    @pytest.mark.tonio
    async def test_api_key_with_dollar_bang_escapes_literal_bang_and_still_interpolates(self, registry_env):
        import contextlib

        _tmp, models_json_path, auth_storage = registry_env
        original = os.environ.get("TEST_API_KEY_12345")
        os.environ["TEST_API_KEY_12345"] = "env-api-key-value"
        try:
            write_models_json(
                models_json_path, {"custom-provider": provider_with_api_key("$!literal-$TEST_API_KEY_12345")}
            )
            registry = await create_model_registry(auth_storage, models_json_path)
            assert await registry.get_api_key_for_provider("custom-provider") == "!literal-env-api-key-value"
        finally:
            if original is None:
                with contextlib.suppress(KeyError):
                    del os.environ["TEST_API_KEY_12345"]
            else:
                os.environ["TEST_API_KEY_12345"] = original

    @pytest.mark.tonio
    async def test_plain_api_key_is_used_directly_even_when_it_matches_an_env_var(self, registry_env):
        import contextlib

        _tmp, models_json_path, auth_storage = registry_env
        original = os.environ.get("TEST_API_KEY_12345")
        os.environ["TEST_API_KEY_12345"] = "env-api-key-value"
        try:
            write_models_json(models_json_path, {"custom-provider": provider_with_api_key("TEST_API_KEY_12345")})
            registry = await create_model_registry(auth_storage, models_json_path)
            assert await registry.get_api_key_for_provider("custom-provider") == "TEST_API_KEY_12345"
        finally:
            if original is None:
                with contextlib.suppress(KeyError):
                    del os.environ["TEST_API_KEY_12345"]
            else:
                os.environ["TEST_API_KEY_12345"] = original

    @pytest.mark.tonio
    async def test_api_key_as_literal_value_is_used_directly_when_not_an_env_var(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        os.environ.pop("literal_api_key_value", None)
        write_models_json(models_json_path, {"custom-provider": provider_with_api_key("literal_api_key_value")})
        registry = await create_model_registry(auth_storage, models_json_path)
        assert await registry.get_api_key_for_provider("custom-provider") == "literal_api_key_value"

    @pytest.mark.tonio
    async def test_api_key_command_can_use_shell_features_like_pipes(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(
            models_json_path, {"custom-provider": provider_with_api_key("!echo 'hello world' | tr ' ' '-'")}
        )
        registry = await create_model_registry(auth_storage, models_json_path)
        assert await registry.get_api_key_for_provider("custom-provider") == "hello-world"


class TestRequestTimeResolution:
    @pytest.mark.tonio
    async def test_command_is_executed_on_every_provider_lookup(self, registry_env):
        tmp_dir, models_json_path, auth_storage = registry_env
        counter_file = tmp_dir / "counter"
        counter_file.write_text("0")
        counter_path = to_sh_path(str(counter_file))
        command = f'!sh -c \'count=$(cat "{counter_path}"); echo $((count + 1)) > "{counter_path}"; echo "key-value"\''
        write_models_json(models_json_path, {"custom-provider": provider_with_api_key(command)})

        registry = await create_model_registry(auth_storage, models_json_path)
        await registry.get_api_key_for_provider("custom-provider")
        await registry.get_api_key_for_provider("custom-provider")
        await registry.get_api_key_for_provider("custom-provider")

        assert int(counter_file.read_text().strip()) == 3

    @pytest.mark.tonio
    async def test_commands_are_re_executed_across_registry_instances(self, registry_env):
        tmp_dir, models_json_path, auth_storage = registry_env
        counter_file = tmp_dir / "counter"
        counter_file.write_text("0")
        counter_path = to_sh_path(str(counter_file))
        command = f'!sh -c \'count=$(cat "{counter_path}"); echo $((count + 1)) > "{counter_path}"; echo "key-value"\''
        write_models_json(models_json_path, {"custom-provider": provider_with_api_key(command)})

        registry1 = await create_model_registry(auth_storage, models_json_path)
        await registry1.get_api_key_for_provider("custom-provider")

        registry2 = await create_model_registry(auth_storage, models_json_path)
        await registry2.get_api_key_for_provider("custom-provider")

        assert int(counter_file.read_text().strip()) == 2

    @pytest.mark.tonio
    async def test_different_commands_resolve_independently(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(
            models_json_path,
            {"provider-a": provider_with_api_key("!echo key-a"), "provider-b": provider_with_api_key("!echo key-b")},
        )

        registry = await create_model_registry(auth_storage, models_json_path)

        assert await registry.get_api_key_for_provider("provider-a") == "key-a"
        assert await registry.get_api_key_for_provider("provider-b") == "key-b"

    @pytest.mark.tonio
    async def test_failed_commands_are_retried(self, registry_env):
        tmp_dir, models_json_path, auth_storage = registry_env
        counter_file = tmp_dir / "counter"
        counter_file.write_text("0")
        counter_path = to_sh_path(str(counter_file))
        command = f'!sh -c \'count=$(cat "{counter_path}"); echo $((count + 1)) > "{counter_path}"; exit 1\''
        write_models_json(models_json_path, {"custom-provider": provider_with_api_key(command)})

        registry = await create_model_registry(auth_storage, models_json_path)
        assert await registry.get_api_key_for_provider("custom-provider") is None
        assert await registry.get_api_key_for_provider("custom-provider") is None

        assert int(counter_file.read_text().strip()) == 2

    @pytest.mark.tonio
    async def test_provider_auth_status_reports_api_key_env_vars_from_models_json(self, registry_env):
        import contextlib

        _tmp, models_json_path, auth_storage = registry_env
        env_var_name = "TEST_API_KEY_STATUS_TEST_98765"
        original = os.environ.get(env_var_name)
        try:
            os.environ[env_var_name] = "status-test-key"
            write_models_json(models_json_path, {"custom-provider": provider_with_api_key(f"${env_var_name}")})

            registry = await create_model_registry(auth_storage, models_json_path)

            assert registry.get_provider_auth_status("custom-provider") == AuthStatus(
                configured=True, source="environment", label=env_var_name
            )
        finally:
            if original is None:
                with contextlib.suppress(KeyError):
                    del os.environ[env_var_name]
            else:
                os.environ[env_var_name] = original

    @pytest.mark.tonio
    async def test_provider_auth_status_reports_interpolated_api_key_env_vars(self, registry_env):
        import contextlib

        _tmp, models_json_path, auth_storage = registry_env
        name_a = "TEST_API_KEY_STATUS_PART_A_98765"
        name_b = "TEST_API_KEY_STATUS_PART_B_98765"
        saved_a = os.environ.get(name_a)
        saved_b = os.environ.get(name_b)
        os.environ[name_a] = "left"
        os.environ[name_b] = "right"
        try:
            write_models_json(
                models_json_path, {"custom-provider": provider_with_api_key(f"${{{name_a}}}_${{{name_b}}}")}
            )

            registry = await create_model_registry(auth_storage, models_json_path)

            assert registry.get_provider_auth_status("custom-provider") == AuthStatus(
                configured=True, source="environment", label=f"{name_a}, {name_b}"
            )
        finally:
            for key, saved in ((name_a, saved_a), (name_b, saved_b)):
                if saved is None:
                    with contextlib.suppress(KeyError):
                        del os.environ[key]
                else:
                    os.environ[key] = saved

    @pytest.mark.tonio
    async def test_provider_auth_status_reports_non_env_api_key_values_as_config_key(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(models_json_path, {"custom-provider": provider_with_api_key("literal_api_key_value")})

        registry = await create_model_registry(auth_storage, models_json_path)

        assert registry.get_provider_auth_status("custom-provider") == AuthStatus(
            configured=True, source="models_json_key"
        )

    @pytest.mark.tonio
    async def test_missing_explicit_env_api_key_keeps_provider_unavailable(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        env_var_name = "TEST_API_KEY_MISSING_TEST_98765"
        os.environ.pop(env_var_name, None)

        write_models_json(models_json_path, {"custom-provider": provider_with_api_key(f"${env_var_name}")})

        registry = await create_model_registry(auth_storage, models_json_path)

        assert registry.get_provider_auth_status("custom-provider") == AuthStatus(configured=False)
        assert not any(model.provider == "custom-provider" for model in registry.get_available())

    @pytest.mark.tonio
    async def test_provider_auth_status_reports_command_api_key_values_without_executing_them(self, registry_env):
        tmp_dir, models_json_path, auth_storage = registry_env
        counter_file = tmp_dir / "status-counter"
        counter_file.write_text("0")
        counter_path = to_sh_path(str(counter_file))
        command = f"!sh -c 'echo 1 > \"{counter_path}\"; echo key-value'"
        write_models_json(models_json_path, {"custom-provider": provider_with_api_key(command)})

        registry = await create_model_registry(auth_storage, models_json_path)

        assert registry.get_provider_auth_status("custom-provider") == AuthStatus(
            configured=True, source="models_json_command"
        )
        assert counter_file.read_text() == "0"

    @pytest.mark.tonio
    async def test_environment_variables_are_not_cached(self, registry_env):
        import contextlib

        _tmp, models_json_path, auth_storage = registry_env
        env_var_name = "TEST_API_KEY_CACHE_TEST_98765"
        original = os.environ.get(env_var_name)
        try:
            os.environ[env_var_name] = "first-value"
            write_models_json(models_json_path, {"custom-provider": provider_with_api_key(f"${env_var_name}")})

            registry = await create_model_registry(auth_storage, models_json_path)

            assert await registry.get_api_key_for_provider("custom-provider") == "first-value"
            os.environ[env_var_name] = "second-value"
            assert await registry.get_api_key_for_provider("custom-provider") == "second-value"
        finally:
            if original is None:
                with contextlib.suppress(KeyError):
                    del os.environ[env_var_name]
            else:
                os.environ[env_var_name] = original

    @pytest.mark.tonio
    async def test_get_available_does_not_execute_command_backed_api_key_resolution(self, registry_env):
        tmp_dir, models_json_path, auth_storage = registry_env
        counter_file = tmp_dir / "counter"
        counter_file.write_text("0")
        counter_path = to_sh_path(str(counter_file))
        command = f'!sh -c \'count=$(cat "{counter_path}"); echo $((count + 1)) > "{counter_path}"; echo "key-value"\''
        write_models_json(models_json_path, {"custom-provider": provider_with_api_key(command)})

        registry = await create_model_registry(auth_storage, models_json_path)
        available = registry.get_available()

        assert any(model.provider == "custom-provider" for model in available)
        assert int(counter_file.read_text().strip()) == 0

    @pytest.mark.tonio
    async def test_get_api_key_and_headers_resolves_auth_header_on_every_request(self, registry_env):
        tmp_dir, models_json_path, auth_storage = registry_env
        token_file = tmp_dir / "token"
        token_file.write_text("token-1")
        token_path = to_sh_path(str(token_file))

        write_models_json(
            models_json_path,
            {"custom-provider": {**provider_with_api_key(f"!sh -c 'cat \"{token_path}\"'"), "authHeader": True}},
        )

        registry = await create_model_registry(auth_storage, models_json_path)
        model = registry.find("custom-provider", "test-model")
        assert model is not None

        auth1 = await registry.get_api_key_and_headers(model)
        assert auth1 == ResolvedRequestAuth(ok=True, api_key="token-1", headers={"Authorization": "Bearer token-1"})

        token_file.write_text("token-2")

        auth2 = await registry.get_api_key_and_headers(model)
        assert auth2 == ResolvedRequestAuth(ok=True, api_key="token-2", headers={"Authorization": "Bearer token-2"})

    @pytest.mark.tonio
    async def test_get_api_key_and_headers_resolves_configured_auth_exactly_once(self, registry_env):
        tmp_dir, models_json_path, auth_storage = registry_env
        counter_file = tmp_dir / "auth-counter"
        counter_file.write_text("0")
        counter_path = to_sh_path(str(counter_file))
        command = (
            f'!sh -c \'count=$(cat "{counter_path}"); count=$((count + 1)); '
            f'echo "$count" > "{counter_path}"; echo "token-$count"\''
        )
        write_models_json(models_json_path, {"custom-provider": {**provider_with_api_key(command), "authHeader": True}})

        registry = await create_model_registry(auth_storage, models_json_path)
        auth = await registry.get_api_key_and_headers(registry.find("custom-provider", "test-model"))

        assert auth == ResolvedRequestAuth(ok=True, api_key="token-1", headers={"Authorization": "Bearer token-1"})
        assert counter_file.read_text().strip() == "1"

    @pytest.mark.tonio
    async def test_stored_credentials_bypass_lower_priority_configured_auth_commands(self, registry_env):
        tmp_dir, models_json_path, auth_storage = registry_env
        counter_file = tmp_dir / "fallback-counter"
        counter_file.write_text("0")
        counter_path = to_sh_path(str(counter_file))
        write_models_json(
            models_json_path,
            {"custom-provider": provider_with_api_key(f"!sh -c 'echo 1 > \"{counter_path}\"; echo fallback-key'")},
        )

        async def set_credential(_current):
            return ApiKeyCredential(key="stored-key")

        await auth_storage.modify("custom-provider", set_credential)

        registry = await create_model_registry(auth_storage, models_json_path)
        auth = await registry.get_api_key_and_headers(registry.find("custom-provider", "test-model"))

        assert auth.ok is True
        assert auth.api_key == "stored-key"
        assert counter_file.read_text().strip() == "0"

    @pytest.mark.tonio
    async def test_get_api_key_and_headers_preserves_the_legacy_missing_key_auth_header_error(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(
            models_json_path,
            {
                "custom-provider": {
                    "baseUrl": "https://example.test/v1",
                    "api": "openai-completions",
                    "authHeader": True,
                    "models": [{"id": "test-model"}],
                }
            },
        )

        registry = await create_model_registry(auth_storage, models_json_path)
        auth = await registry.get_api_key_and_headers(registry.find("custom-provider", "test-model"))

        assert auth == ResolvedRequestAuth(ok=False, error='No API key found for "custom-provider"')

    @pytest.mark.tonio
    async def test_get_api_key_and_headers_returns_an_error_for_failed_auth_header_resolution(self, registry_env):
        _tmp, models_json_path, auth_storage = registry_env
        write_models_json(
            models_json_path, {"custom-provider": {**provider_with_api_key("!exit 1"), "authHeader": True}}
        )

        registry = await create_model_registry(auth_storage, models_json_path)
        model = registry.find("custom-provider", "test-model")
        assert model is not None

        auth = await registry.get_api_key_and_headers(model)
        assert auth.ok is False
        assert 'Failed to resolve API key for provider "custom-provider"' in auth.error


class TestModelsJsonErrors:
    """Malformed models.json halves of pi's config-value-migration.test.ts
    (its migration halves target unported legacy startup migrations)."""

    @pytest.mark.tonio
    @pytest.mark.parametrize("content", ['{\n  "providers": {\n', ""])
    async def test_does_not_throw_on_malformed_models_json(self, registry_env, content):
        _tmp, models_json_path, auth_storage = registry_env
        with open(models_json_path, "w", encoding="utf-8") as f:
            f.write(content)

        registry = await create_model_registry(auth_storage, models_json_path)
        load_error = registry.get_error()
        assert "Failed to parse models.json" in load_error
        assert f"File: {models_json_path}" in load_error

    @pytest.mark.tonio
    async def test_leaves_uppercase_models_json_api_key_and_header_values_unchanged(self, registry_env):
        import contextlib

        _tmp, models_json_path, auth_storage = registry_env
        env_keys = ["CUSTOM_API_KEY", "HEADER_API_KEY", "MODEL_API_KEY", "OVERRIDE_API_KEY"]
        saved = {key: os.environ.get(key) for key in env_keys}
        for key in env_keys:
            os.environ[key] = f"env-{key}"

        try:
            write_models_json(
                models_json_path,
                {
                    "custom-provider": {
                        "baseUrl": "https://example.com/v1",
                        "apiKey": "CUSTOM_API_KEY",
                        "api": "openai-completions",
                        "headers": {"x-api-key": "HEADER_API_KEY", "x-literal": "literal"},
                        "models": [{"id": "model-a", "headers": {"x-model-key": "MODEL_API_KEY"}}],
                        "modelOverrides": {"model-b": {"headers": {"x-override-key": "OVERRIDE_API_KEY"}}},
                    }
                },
            )

            registry = await create_model_registry(auth_storage, models_json_path)
            model = registry.find("custom-provider", "model-a")
            assert model is not None
            assert await registry.get_api_key_for_provider("custom-provider") == "CUSTOM_API_KEY"
            auth = await registry.get_api_key_and_headers(model)
            assert auth.ok is True
            assert auth.api_key == "CUSTOM_API_KEY"
            assert auth.headers == {
                "x-api-key": "HEADER_API_KEY",
                "x-literal": "literal",
                "x-model-key": "MODEL_API_KEY",
            }
        finally:
            for key in env_keys:
                if saved[key] is None:
                    with contextlib.suppress(KeyError):
                        del os.environ[key]
                else:
                    os.environ[key] = saved[key]
