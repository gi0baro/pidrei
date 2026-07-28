"""Mirrors of pi coding-agent test/model-runtime-auth-options.test.ts and
test/model-runtime-modify-models-compat.test.ts.

Adaptations: only the anthropic/openai builtins exist (bedrock/vertex/
cloudflare/codex land in Phase 5), and the anthropic OAuth login method is
Phase 5 — the stored-OAuth auth-status check runs against an extension OAuth
provider instead. model-runtime-cloudflare-compat.test.ts is entirely
builtin-specific and is not mirrored yet.
"""

import time

import pytest

from pidrei.core.auth_storage import AuthStorage
from pidrei.core.model_registry import ModelRegistry
from pidrei.core.model_runtime import ModelRuntime, ModelRuntimeAuthOverrides
from pidrei.core.provider_composer import ExtensionOAuthConfig
from pidrei_ai.auth.credential_store import InMemoryCredentialStore
from pidrei_ai.auth.types import (
    ApiKeyAuth,
    ApiKeyCredential,
    AuthCheck,
    AuthResult,
    ModelAuth,
    OAuthCredential,
    ProviderAuth,
)
from pidrei_ai.models_store import InMemoryModelsStore
from pidrei_ai.registry import ModelsRefreshOptions, create_provider
from tests.model_runtime_helpers import make_model


def now_ms() -> int:
    return int(time.time() * 1000)


def auth_options(runtime, type=None):
    options = []
    for provider in runtime.get_providers():
        if (type is None or type == "oauth") and provider.auth.oauth is not None:
            options.append(("oauth", provider, provider.auth.oauth))
        if (type is None or type == "api_key") and provider.auth.api_key is not None:
            options.append(("api_key", provider, provider.auth.api_key))
    return options


def model_entry(id):
    return {
        "id": id,
        "name": id,
        "reasoning": False,
        "input": ["text"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 10000,
        "maxTokens": 1000,
    }


class TestModelRuntimeAuthOptions:
    @pytest.mark.tonio
    async def test_accepts_a_pidrei_ai_credential_store(self):
        credentials = InMemoryCredentialStore()

        async def set_key(_current):
            return ApiKeyCredential(key="stored-key")

        await credentials.modify("anthropic", set_key)
        runtime = await ModelRuntime.create(credentials=credentials, models_path=None)

        resolution = await runtime.get_auth("anthropic")
        assert resolution.auth.api_key == "stored-key"

    @pytest.mark.tonio
    async def test_scopes_provider_availability_reads_and_records_refresh_failures(self):
        base = InMemoryCredentialStore()
        reads = []
        state = {"fail": False}

        class RecordingStore:
            async def read(self, provider_id):
                reads.append(provider_id)
                if state["fail"]:
                    raise Exception(f"read failed for {provider_id}")
                return await base.read(provider_id)

            async def list(self):
                return await base.list()

            async def modify(self, provider_id, fn):
                return await base.modify(provider_id, fn)

            async def delete(self, provider_id):
                await base.delete(provider_id)

        runtime = await ModelRuntime.create(credentials=RecordingStore(), models_path=None)

        reads.clear()
        await runtime.get_available("anthropic")
        assert set(reads) == {"anthropic"}

        state["fail"] = True
        with pytest.raises(Exception, match="Credential store read failed for anthropic"):
            await runtime.get_available("anthropic")
        assert "Availability refresh: Credential store read failed for anthropic" in runtime.get_error()

        state["fail"] = False
        await runtime.get_available()
        assert runtime.get_error() is None

    @pytest.mark.tonio
    async def test_projects_provider_owned_methods_names_and_status(self):
        runtime = await ModelRuntime.create(credentials=AuthStorage.in_memory(), models_path=None)
        options = auth_options(runtime)

        by_provider = {(type_, provider.id): method for type_, provider, method in options}
        assert by_provider[("api_key", "anthropic")].name == "Anthropic API key"
        assert by_provider[("api_key", "openai")].name == "OpenAI API key"
        assert all(type_ == "api_key" for type_, _p, _m in auth_options(runtime, "api_key"))
        assert all(type_ == "oauth" for type_, _p, _m in auth_options(runtime, "oauth"))

    @pytest.mark.tonio
    async def test_attaches_the_providers_active_auth_status_to_every_method_option(self):
        """Adapted: pi checks the anthropic builtin's stored OAuth; pidrei's
        anthropic OAuth method is Phase 5, so an extension OAuth provider
        exercises the same stored-OAuth checkAuth projection."""
        runtime = await ModelRuntime.create(
            credentials=AuthStorage.in_memory(
                {"extension-oauth": OAuthCredential(access="access", refresh="refresh", expires=now_ms() + 60_000)}
            ),
            models_path=None,
        )

        async def login(_callbacks):
            return {"access": "access", "refresh": "refresh", "expires": now_ms() + 60_000}

        async def refresh_token(credentials):
            return credentials

        runtime.register_provider(
            "extension-oauth",
            {
                "baseUrl": "https://example.test/v1",
                "api": "openai-completions",
                "oauth": ExtensionOAuthConfig(
                    name="Extension subscription",
                    login=login,
                    refresh_token=refresh_token,
                    get_api_key=lambda credentials: credentials.access,
                ),
                "models": [model_entry("extension-model")],
            },
        )

        options = [entry for entry in auth_options(runtime) if entry[1].id == "extension-oauth"]
        assert len(options) == 1
        check = await runtime.check_auth("extension-oauth")
        assert check.type == "oauth"

    @pytest.mark.tonio
    async def test_constructs_an_api_key_method_for_an_extension_api_key_provider(self):
        runtime = await ModelRuntime.create(credentials=AuthStorage.in_memory(), models_path=None)
        runtime.register_provider(
            "extension-api-key",
            {
                "name": "Extension API Key",
                "baseUrl": "https://example.test/v1",
                "apiKey": "$EXTENSION_TEST_API_KEY",
                "api": "openai-completions",
                "models": [model_entry("extension-model")],
            },
        )

        options = [entry for entry in auth_options(runtime) if entry[1].id == "extension-api-key"]
        assert len(options) == 1
        type_, provider, method = options[0]
        assert type_ == "api_key"
        assert provider.name == "Extension API Key"
        assert method.name == "API key"
        assert callable(method.login)

    @pytest.mark.tonio
    async def test_resolves_configured_auth_from_request_scoped_environment_overrides(self):
        runtime = await ModelRuntime.create(credentials=AuthStorage.in_memory(), models_path=None)
        runtime.register_provider(
            "request-env-provider",
            {
                "baseUrl": "https://example.test/v1",
                "apiKey": "$REQUEST_SCOPED_API_KEY",
                "headers": {"x-request-value": "$REQUEST_SCOPED_HEADER"},
                "api": "openai-completions",
                "models": [model_entry("request-env-model")],
            },
        )

        auth = await runtime.get_auth(
            "request-env-provider",
            ModelRuntimeAuthOverrides(
                env={"REQUEST_SCOPED_API_KEY": "request-key", "REQUEST_SCOPED_HEADER": "request-header"}
            ),
        )

        assert auth.auth == ModelAuth(api_key="request-key", headers={"x-request-value": "request-header"})

    @pytest.mark.tonio
    async def test_lets_an_explicit_authorization_header_override_auth_header_case_insensitively(self):
        runtime = await ModelRuntime.create(credentials=AuthStorage.in_memory(), models_path=None)
        captured = {}

        def stream_simple(_model, _context, options=None):
            captured["headers"] = dict(options.headers) if options.headers else options.headers
            raise Exception("captured")

        runtime.register_provider(
            "auth-header-provider",
            {
                "baseUrl": "https://example.test/v1",
                "apiKey": "generated-key",
                "authHeader": True,
                "api": "openai-completions",
                "streamSimple": stream_simple,
                "models": [model_entry("auth-header-model")],
            },
        )
        model = runtime.get_model("auth-header-provider", "auth-header-model")
        assert model is not None

        from pidrei_ai.types import Context, SimpleStreamOptions

        await runtime.complete_simple(
            model, Context(messages=[]), SimpleStreamOptions(headers={"authorization": "Explicit token"})
        )

        assert captured["headers"] == {"authorization": "Explicit token"}

    @pytest.mark.tonio
    async def test_transforms_fully_assembled_headers_once_without_forwarding_the_transform(self):
        runtime = await ModelRuntime.create(credentials=AuthStorage.in_memory(), models_path=None)
        captured = {}
        transforms = {"count": 0}

        def stream_simple(_model, _context, options=None):
            assert options.transform_headers is None
            captured["headers"] = dict(options.headers)
            raise Exception("captured")

        runtime.register_provider(
            "header-provider",
            {
                "baseUrl": "https://example.test/v1",
                "apiKey": "generated-key",
                "authHeader": True,
                "headers": {"x-provider": "provider"},
                "api": "openai-completions",
                "streamSimple": stream_simple,
                "models": [{**model_entry("header-model"), "headers": {"x-model": "model"}}],
            },
        )
        model = runtime.get_model("header-provider", "header-model")
        assert model is not None

        async def transform_headers(headers):
            transforms["count"] += 1
            assert headers == {
                "Authorization": "Bearer generated-key",
                "x-provider": "provider",
                "x-model": "model",
                "x-explicit": "explicit",
            }
            return {**headers, "x-transformed": "yes"}

        from pidrei_ai.types import Context, SimpleStreamOptions

        await runtime.complete_simple(
            model,
            Context(messages=[]),
            SimpleStreamOptions(headers={"x-explicit": "explicit"}, transform_headers=transform_headers),
        )

        assert transforms["count"] == 1
        assert captured["headers"] == {
            "Authorization": "Bearer generated-key",
            "x-provider": "provider",
            "x-model": "model",
            "x-explicit": "explicit",
            "x-transformed": "yes",
        }

    @pytest.mark.tonio
    async def test_does_not_fabricate_an_api_key_method_for_an_extension_oauth_only_provider(self):
        runtime = await ModelRuntime.create(credentials=AuthStorage.in_memory(), models_path=None)

        async def login(_callbacks):
            return {"access": "access", "refresh": "refresh", "expires": now_ms() + 60_000}

        async def refresh_token(credentials):
            return credentials

        runtime.register_provider(
            "extension-oauth",
            {
                "name": "Extension OAuth",
                "baseUrl": "https://example.test/v1",
                "api": "openai-completions",
                "oauth": ExtensionOAuthConfig(
                    name="Extension subscription",
                    login=login,
                    refresh_token=refresh_token,
                    get_api_key=lambda credentials: credentials.access,
                ),
                "models": [model_entry("extension-model")],
            },
        )

        options = [entry for entry in auth_options(runtime) if entry[1].id == "extension-oauth"]
        assert len(options) == 1
        type_, provider, method = options[0]
        assert type_ == "oauth"
        assert provider.name == "Extension OAuth"
        assert method.name == "Extension subscription"


class TestExtensionProviderModelLifecycle:
    @pytest.mark.tonio
    async def test_registers_native_pidrei_ai_providers_with_their_auth_implementation(self):
        runtime = await ModelRuntime.create(
            credentials=AuthStorage.in_memory(),
            models_store=InMemoryModelsStore(),
            models_path=None,
            allow_model_network=False,
        )
        native_model = make_model("extension-native", "native", base_url="https://fallback.test/v1")

        async def login(interaction):
            from pidrei_ai.auth.types import AuthPrompt

            return ApiKeyCredential(key=await interaction.prompt(AuthPrompt(type="secret", message="API key")))

        async def check(_ctx, credential):
            if credential is not None and credential.key:
                return AuthCheck(type="api_key", source="stored native key")
            return None

        async def resolve(_ctx, credential):
            if credential is not None and credential.key:
                return AuthResult(
                    auth=ModelAuth(api_key=credential.key, base_url="https://resolved.test/v1"),
                    source="stored native key",
                )
            return None

        provider = create_provider(
            id="extension-native",
            name="Extension Native",
            auth=ProviderAuth(api_key=ApiKeyAuth(name="Native setup", login=login, check=check, resolve=resolve)),
            models=[native_model],
            api={},
        )

        runtime.register_native_provider(provider)
        registry = ModelRegistry(runtime)
        assert registry.get_provider("extension-native") is provider
        assert registry.get_registered_native_provider("extension-native") is provider
        assert "extension-native" in registry.get_registered_provider_ids()
        assert registry.find("extension-native", "native") is not None

        class Interaction:
            cancel = None

            async def prompt(self, _prompt):
                return "secret"

            def notify(self, _event):
                pass

        await runtime.login("extension-native", "api_key", Interaction())
        resolution = await registry.get_provider_auth("extension-native")
        assert resolution.auth.api_key == "secret"
        assert resolution.auth.base_url == "https://resolved.test/v1"

        registry.unregister_provider("extension-native")
        assert registry.get_provider("extension-native") is None

    @pytest.mark.tonio
    async def test_applies_models_json_overrides_above_native_providers(self, tmp_dir):
        import json

        models_path = tmp_dir / "models.json"
        models_path.write_text(
            json.dumps({"providers": {"extension-native": {"modelOverrides": {"native": {"contextWindow": 4242}}}}}),
            encoding="utf-8",
        )
        runtime = await ModelRuntime.create(
            credentials=AuthStorage.in_memory(),
            models_store=InMemoryModelsStore(),
            models_path=str(models_path),
            allow_model_network=False,
        )
        native_model = make_model("extension-native", "native", base_url="https://native.test/v1")

        async def resolve(_ctx, _credential):
            return AuthResult(auth=ModelAuth(api_key="key"), source="native")

        runtime.register_native_provider(
            create_provider(
                id="extension-native",
                name="Extension Native",
                auth=ProviderAuth(api_key=ApiKeyAuth(name="Native key", resolve=resolve)),
                models=[native_model],
                api={},
            )
        )

        assert runtime.get_model("extension-native", "native").context_window == 4242

    @pytest.mark.tonio
    async def test_publishes_refresh_models_results_without_forcing_models_store_persistence(self):
        models_store = InMemoryModelsStore()
        runtime = await ModelRuntime.create(
            credentials=AuthStorage.in_memory(),
            models_store=models_store,
            models_path=None,
            allow_model_network=False,
        )

        async def refresh_models(_context):
            return [
                {
                    **model_entry("live"),
                    "baseUrl": "http://localhost:8080/v1",
                }
            ]

        runtime.register_provider(
            "extension-dynamic",
            {
                "baseUrl": "http://localhost:8080/v1",
                "apiKey": "local",
                "api": "openai-completions",
                "refreshModels": refresh_models,
            },
        )

        await runtime.refresh(ModelsRefreshOptions(allow_network=False))
        assert runtime.get_model("extension-dynamic", "live") is not None
        assert await models_store.read("extension-dynamic") is None

    @pytest.mark.tonio
    async def test_applies_legacy_oauth_modify_models_after_async_credential_initialization(self):
        runtime = await ModelRuntime.create(
            credentials=AuthStorage.in_memory(
                {"extension-oauth": OAuthCredential(access="access", refresh="refresh", expires=now_ms() + 60_000)}
            ),
            models_store=InMemoryModelsStore(),
            models_path=None,
            allow_model_network=False,
        )

        async def login(_callbacks):
            raise Exception("not used")

        async def refresh_token(credential):
            return credential

        def modify_models(models, credential):
            if credential.access == "access":
                return [*models, make_model("extension-oauth", "credential-model", base_url="https://example.test/v1")]
            return models

        runtime.register_provider(
            "extension-oauth",
            {
                "baseUrl": "https://example.test/v1",
                "api": "openai-completions",
                "models": [model_entry("base")],
                "oauth": ExtensionOAuthConfig(
                    name="Extension OAuth",
                    login=login,
                    refresh_token=refresh_token,
                    get_api_key=lambda credential: credential.access,
                    modify_models=modify_models,
                ),
            },
        )

        await runtime.refresh(ModelsRefreshOptions(allow_network=False))
        assert runtime.get_model("extension-oauth", "base") is not None
        assert runtime.get_model("extension-oauth", "credential-model") is not None

        await runtime.logout("extension-oauth")
        assert runtime.get_model("extension-oauth", "credential-model") is None

    @pytest.mark.tonio
    async def test_refresh_is_serialized_against_register_provider_spawned_refreshes(self):
        """Regression (macOS CI, 2026-07-28): `register_provider` fires an
        untracked `refresh()`, and a caller's awaited `refresh()` runs in
        parallel with it. Each refresh rebuilds the composed provider objects,
        so the loser's credential projection landed on instances the winner
        had already replaced — the legacy-OAuth model vanished from the
        snapshot. pi's single thread interleaves the same shape harmlessly;
        `refresh()` is serialized now, and this loop hammers the window."""
        runtime = await ModelRuntime.create(
            credentials=AuthStorage.in_memory(
                {"extension-oauth": OAuthCredential(access="access", refresh="refresh", expires=now_ms() + 60_000)}
            ),
            models_store=InMemoryModelsStore(),
            models_path=None,
            allow_model_network=False,
        )

        async def login(_callbacks):
            raise Exception("not used")

        async def refresh_token(credential):
            return credential

        def modify_models(models, credential):
            if credential.access == "access":
                return [*models, make_model("extension-oauth", "credential-model", base_url="https://example.test/v1")]
            return models

        config = {
            "baseUrl": "https://example.test/v1",
            "api": "openai-completions",
            "models": [model_entry("base")],
            "oauth": ExtensionOAuthConfig(
                name="Extension OAuth",
                login=login,
                refresh_token=refresh_token,
                get_api_key=lambda credential: credential.access,
                modify_models=modify_models,
            ),
        }

        for attempt in range(10):
            runtime.register_provider("extension-oauth", config)  # spawns its own refresh
            await runtime.refresh(ModelsRefreshOptions(allow_network=False))
            assert runtime.get_model("extension-oauth", "credential-model") is not None, f"lost on attempt {attempt}"
