"""Mirror of pi coding-agent src/core/provider-composer.ts.

Extension provider configs (`ProviderConfigInput`) are camelCase dicts, like
pi's registerProvider objects: key presence carries the defined-vs-undefined
merge semantics. The extension OAuth hook is the `ExtensionOAuthConfig`
dataclass.

pi's global compat API registry (`getApiProvider`) is unported; the fallback
dispatch maps known api names onto pidrei-ai adapter modules directly.

A config's optional ``streamSimple`` handler
(``(model, context, options) -> AssistantMessageEventStream``) carries the same
contract as the built-in providers: it must invoke ``options.on_payload`` before
sending the provider request and use any replacement payload it returns, and it
must invoke ``options.on_response`` after receiving the response and before
consuming its body. ``api`` is required alongside it.
"""

from dataclasses import dataclass, replace
from typing import Any

from pidrei_ai.api.lazy import lazy_stream
from pidrei_ai.auth.types import (
    ApiKeyAuth,
    ApiKeyCredential,
    AuthCheck,
    AuthContext,
    AuthEvent,
    AuthInteraction,
    AuthPrompt,
    AuthResult,
    ModelAuth,
    OAuthAuth,
    OAuthCredential,
    ProviderAuth,
)
from pidrei_ai.registry import ModelsPublication, Provider, RefreshModelsContext
from pidrei_ai.types import Model, ModelCost

from .model_config import ModelConfig
from .model_wire import cost_from_dict, cost_tiers_from_list, merge_compat, parse_compat
from .resolve_config_value import (
    clear_config_value_cache,
    get_config_value_env_var_names,
    is_command_config_value,
    is_config_value_configured,
    resolve_config_value_or_throw,
    resolve_headers_or_throw,
)


type ProviderConfigInput = dict[str, Any]


@dataclass(slots=True, kw_only=True)
class ExtensionOAuthConfig:
    """Extension-supplied legacy OAuth hooks (pi's ExtensionOAuthConfig)."""

    name: str
    login: Any
    refresh_token: Any
    get_api_key: Any
    # Whether access through this auth method is backed by a provider subscription.
    is_subscription: bool | None = None
    # Deprecated: retained for extension source compatibility; ignored by canonical auth flows.
    uses_callback_server: bool = False
    modify_models: Any = None


@dataclass(slots=True)
class OAuthLoginCallbacks:
    """Callback surface handed to ExtensionOAuthConfig.login."""

    on_auth: Any
    on_device_code: Any
    on_prompt: Any
    on_progress: Any
    on_manual_code_input: Any
    on_select: Any
    cancel: Any = None


@dataclass(slots=True)
class AuthStatus:
    configured: bool
    source: str | None = None
    label: str | None = None


@dataclass(slots=True)
class CompatibilityRequestConfig:
    auth_header: bool
    headers: dict[str, str | None] | None = None


clear_api_key_cache = clear_config_value_cache


def _nn(*values: Any) -> Any:
    """JS `??` chain: first non-None value."""
    for value in values:
        if value is not None:
            return value
    return None


def _to_oauth_credential(value: Any) -> OAuthCredential:
    if isinstance(value, OAuthCredential):
        return value
    extra = {key: item for key, item in value.items() if key not in ("type", "refresh", "access", "expires")}
    return OAuthCredential(refresh=value["refresh"], access=value["access"], expires=value["expires"], extra=extra)


def _get_api_provider(api: str) -> Any | None:
    if api == "anthropic-messages":
        # lazy: api adapters load on demand (see api/*_lazy.py)
        from pidrei_ai.api import anthropic_messages

        return anthropic_messages
    if api == "openai-completions":
        # lazy: api adapters load on demand (see api/*_lazy.py)
        from pidrei_ai.api import openai_completions

        return openai_completions
    if api == "openai-responses":
        # lazy: api adapters load on demand (see api/*_lazy.py)
        from pidrei_ai.api import openai_responses

        return openai_responses
    return None


def _model_cost(value: Any) -> ModelCost:
    if isinstance(value, ModelCost):
        return value
    return cost_from_dict(value)


def _model_compat(api: str, value: Any) -> Any:
    if value is None or not isinstance(value, dict):
        return value
    return parse_compat(api, value)


def apply_model_override(model: Model, override: dict[str, Any]) -> Model:
    override_cost = override.get("cost")
    if override_cost:
        tiers_raw = override_cost.get("tiers")
        cost = ModelCost(
            input=_nn(override_cost.get("input"), model.cost.input),
            output=_nn(override_cost.get("output"), model.cost.output),
            cache_read=_nn(override_cost.get("cacheRead"), model.cost.cache_read),
            cache_write=_nn(override_cost.get("cacheWrite"), model.cost.cache_write),
            tiers=cost_tiers_from_list(tiers_raw) if tiers_raw is not None else model.cost.tiers,
        )
    else:
        cost = model.cost
    override_map = override.get("thinkingLevelMap")
    return replace(
        model,
        name=_nn(override.get("name"), model.name),
        reasoning=_nn(override.get("reasoning"), model.reasoning),
        thinking_level_map={**(model.thinking_level_map or {}), **override_map}
        if override_map
        else model.thinking_level_map,
        input=list(_nn(override.get("input"), model.input)),
        cost=cost,
        context_window=_nn(override.get("contextWindow"), model.context_window),
        max_tokens=_nn(override.get("maxTokens"), model.max_tokens),
        sampling_params={**(model.sampling_params or {}), **override["samplingParams"]}
        if override.get("samplingParams") is not None
        else model.sampling_params,
        compat=merge_compat(model.api, model.compat, override.get("compat")),
    )


def _model_from_json(
    provider_id: str,
    definition: dict[str, Any],
    provider_config: dict[str, Any],
    defaults: Model | None,
) -> Model:
    api = _nn(definition.get("api"), provider_config.get("api"), defaults.api if defaults else None)
    if not api:
        raise Exception(
            f'Provider {provider_id}, model {definition["id"]}: no "api" specified. Set at provider or model level.'
        )
    base_url = _nn(definition.get("baseUrl"), provider_config.get("baseUrl"), defaults.base_url if defaults else None)
    if not base_url:
        raise Exception(f'Provider {provider_id}: "baseUrl" is required when defining custom models.')
    if definition.get("contextWindow") is not None and definition["contextWindow"] <= 0:
        raise Exception(f"Provider {provider_id}, model {definition['id']}: invalid contextWindow")
    if definition.get("maxTokens") is not None and definition["maxTokens"] <= 0:
        raise Exception(f"Provider {provider_id}, model {definition['id']}: invalid maxTokens")
    return Model(
        id=definition["id"],
        name=_nn(definition.get("name"), definition["id"]),
        api=api,
        provider=provider_id,
        base_url=base_url,
        reasoning=_nn(definition.get("reasoning"), False),
        thinking_level_map=definition.get("thinkingLevelMap"),
        input=list(_nn(definition.get("input"), ["text"])),
        cost=_model_cost(definition["cost"]) if definition.get("cost") else ModelCost(0, 0, 0, 0),
        context_window=_nn(definition.get("contextWindow"), 128000),
        max_tokens=_nn(definition.get("maxTokens"), 16384),
        sampling_params=definition.get("samplingParams"),
        headers=None,
        compat=merge_compat(api, parse_compat(api, provider_config.get("compat")), definition.get("compat")),
    )


def apply_models_json(
    provider_id: str,
    base_models: list[Model],
    config: dict[str, Any] | None,
) -> list[Model]:
    if not config:
        return list(base_models)
    has_overrides = bool(config.get("modelOverrides"))
    if (
        not config.get("models")
        and not config.get("baseUrl")
        and not config.get("headers")
        and not config.get("compat")
        and not has_overrides
        and not config.get("apiKey")
        and "authHeader" not in config
    ):
        raise Exception(
            f'Provider {provider_id}: must specify "baseUrl", "headers", "compat", "modelOverrides", or "models".'
        )

    models = [
        replace(
            model,
            base_url=_nn(config.get("baseUrl"), model.base_url),
            compat=merge_compat(model.api, model.compat, config.get("compat")),
        )
        for model in base_models
    ]
    for definition in config.get("models") or []:
        existing_index = next((i for i, model in enumerate(models) if model.id == definition["id"]), -1)
        defaults = models[existing_index] if existing_index >= 0 else (models[0] if models else None)
        model = _model_from_json(provider_id, definition, config, defaults)
        if existing_index >= 0:
            models[existing_index] = model
        else:
            models.append(model)
    return models


def apply_extension(
    provider_id: str,
    models: list[Model],
    config: ProviderConfigInput | None,
) -> list[Model]:
    if not config:
        return list(models)
    if not config.get("models"):
        if config.get("baseUrl"):
            return [replace(model, base_url=config["baseUrl"]) for model in models]
        return list(models)
    result: list[Model] = []
    for definition in config["models"]:
        defaults = next((model for model in models if model.id == definition["id"]), None) or (
            models[0] if models else None
        )
        api = _nn(definition.get("api"), config.get("api"), defaults.api if defaults else None)
        if not api:
            raise Exception(
                f'Provider {provider_id}, model {definition["id"]}: no "api" specified. Set at provider or model level.'
            )
        base_url = _nn(definition.get("baseUrl"), config.get("baseUrl"), defaults.base_url if defaults else None)
        if not base_url:
            raise Exception(f'Provider {provider_id}: "baseUrl" is required when defining custom models.')
        result.append(
            Model(
                id=definition["id"],
                name=_nn(definition.get("name"), definition["id"]),
                api=api,
                provider=provider_id,
                base_url=base_url,
                reasoning=_nn(definition.get("reasoning"), False),
                thinking_level_map=definition.get("thinkingLevelMap"),
                input=list(_nn(definition.get("input"), ["text"])),
                cost=_model_cost(definition["cost"]) if definition.get("cost") else ModelCost(0, 0, 0, 0),
                context_window=_nn(definition.get("contextWindow"), 128000),
                max_tokens=_nn(definition.get("maxTokens"), 16384),
                headers=None,
                compat=_model_compat(api, definition.get("compat")),
            )
        )
    return result


def adapt_oauth(config: ExtensionOAuthConfig) -> OAuthAuth:
    async def login(interaction: AuthInteraction) -> OAuthCredential:
        def notify_event(type: str, info: dict[str, Any]) -> None:
            interaction.notify(AuthEvent(type=type, **info))

        async def on_prompt(prompt: dict[str, Any]) -> str:
            return await interaction.prompt(AuthPrompt(type="text", **prompt))

        async def on_manual_code_input() -> str:
            return await interaction.prompt(AuthPrompt(type="manual_code", message="Paste the authorization code"))

        async def on_select(prompt: dict[str, Any]) -> str:
            return await interaction.prompt(AuthPrompt(type="select", **prompt))

        callbacks = OAuthLoginCallbacks(
            on_auth=lambda info: notify_event("auth_url", info),
            on_device_code=lambda info: notify_event("device_code", info),
            on_prompt=on_prompt,
            on_progress=lambda message: interaction.notify(AuthEvent(type="progress", message=message)),
            on_manual_code_input=on_manual_code_input,
            on_select=on_select,
            cancel=interaction.cancel,
        )
        return _to_oauth_credential(await config.login(callbacks))

    async def refresh(credential: OAuthCredential, cancel: Any) -> OAuthCredential:
        return _to_oauth_credential(await config.refresh_token(credential, cancel))

    async def to_auth(credential: OAuthCredential) -> ModelAuth:
        return ModelAuth(api_key=config.get_api_key(credential))

    return OAuthAuth(
        name=config.name, is_subscription=config.is_subscription, login=login, refresh=refresh, to_auth=to_auth
    )


def _with_configured_auth(auth: ModelAuth, headers: dict[str, str] | None, auth_header: bool) -> ModelAuth:
    merged: dict[str, str | None] | None = (
        {**(auth.headers or {}), **(headers or {})} if auth.headers or headers else None
    )
    if auth_header:
        if not auth.api_key:
            raise Exception("authHeader requires a resolved API key")
        merged = {**(merged or {}), "Authorization": f"Bearer {auth.api_key}"}
    return replace(auth, headers=merged)


def _configured_api_key(config: dict[str, Any] | None, extension: ProviderConfigInput | None) -> str | None:
    return _nn(extension.get("apiKey") if extension else None, config.get("apiKey") if config else None)


def _configured_headers(config: dict[str, Any] | None, extension: ProviderConfigInput | None) -> dict[str, str] | None:
    config_headers = config.get("headers") if config else None
    extension_headers = extension.get("headers") if extension else None
    if not config_headers and not extension_headers:
        return None
    return {**(config_headers or {}), **(extension_headers or {})}


async def _config_context_env(
    values: list[str],
    ctx: AuthContext,
    explicit: dict[str, str] | None = None,
) -> dict[str, str] | None:
    env = dict(explicit or {})
    names: list[str] = []
    for value in values:
        for name in get_config_value_env_var_names(value):
            if name not in names:
                names.append(name)
    for name in names:
        if env.get(name) is not None:
            continue
        value = await ctx.env(name)
        if value is not None:
            env[name] = value
    return env if env else None


def compose_api_key_auth(
    provider_id: str,
    base: Provider | None,
    config: dict[str, Any] | None,
    extension: ProviderConfigInput | None,
) -> ApiKeyAuth | None:
    inherited = base.auth.api_key if base is not None else None
    raw_key = _configured_api_key(config, extension)
    oauth = _nn(extension.get("oauth") if extension else None, base.auth.oauth if base is not None else None)
    # OAuth-only providers get no fabricated API-key login method.
    if inherited is None and raw_key is None and oauth is not None:
        return None
    raw_headers = _configured_headers(config, extension)
    auth_header = _nn(
        extension.get("authHeader") if extension else None,
        config.get("authHeader") if config else None,
        False,
    )

    async def default_login(interaction: AuthInteraction) -> ApiKeyCredential:
        return ApiKeyCredential(key=await interaction.prompt(AuthPrompt(type="secret", message="Enter API key")))

    async def check(ctx: AuthContext, credential: ApiKeyCredential | None, cancel: Any) -> AuthCheck | None:
        if credential is not None:
            if inherited is not None and inherited.check is not None:
                return await inherited.check(ctx, credential, cancel)
            if credential.key:
                return AuthCheck(type="api_key", source="stored credential")
            resolved = await inherited.resolve(ctx, credential, cancel) if inherited is not None else None
            return AuthCheck(type="api_key", source=resolved.source) if resolved is not None else None
        if raw_key is not None:
            if is_command_config_value(raw_key):
                return AuthCheck(type="api_key", source="configured API key")
            for name in get_config_value_env_var_names(raw_key):
                if await ctx.env(name) is None:
                    return None
            return AuthCheck(type="api_key", source="configured API key")
        if inherited is not None and inherited.check is not None:
            return await inherited.check(ctx, None, cancel)
        resolved = await inherited.resolve(ctx, None, cancel) if inherited is not None else None
        return AuthCheck(type="api_key", source=resolved.source) if resolved is not None else None

    async def resolve(ctx: AuthContext, credential: ApiKeyCredential | None, cancel: Any) -> AuthResult | None:
        if credential is not None:
            if inherited is not None:
                result = await inherited.resolve(ctx, credential, cancel)
            elif credential.key:
                result = AuthResult(
                    auth=ModelAuth(api_key=credential.key), env=credential.env, source="stored credential"
                )
            else:
                result = None
        elif raw_key is not None:
            env = await _config_context_env([raw_key], ctx)
            key = await resolve_config_value_or_throw(raw_key, f'API key for provider "{provider_id}"', env)
            if inherited is not None:
                result = await inherited.resolve(ctx, ApiKeyCredential(key=key), cancel)
            else:
                result = AuthResult(auth=ModelAuth(api_key=key), source="configured API key")
        else:
            result = await inherited.resolve(ctx, None, cancel) if inherited is not None else None
        if result is None:
            return None
        explicit_env = {
            **(dict(credential.env) if credential is not None and credential.env else {}),
            **(dict(result.env) if result.env else {}),
        }
        header_env = await _config_context_env(list((raw_headers or {}).values()), ctx, explicit_env)
        headers = await resolve_headers_or_throw(raw_headers, f'provider "{provider_id}"', header_env)
        return replace(result, auth=_with_configured_auth(result.auth, headers, auth_header))

    return ApiKeyAuth(
        name=inherited.name if inherited is not None else "API key",
        login=_nn(inherited.login if inherited is not None else None, default_login),
        check=check,
        resolve=resolve,
    )


def compose_oauth_auth(
    provider_id: str,
    base: Provider | None,
    config: dict[str, Any] | None,
    extension: ProviderConfigInput | None,
) -> OAuthAuth | None:
    extension_oauth = extension.get("oauth") if extension else None
    oauth = adapt_oauth(extension_oauth) if extension_oauth else (base.auth.oauth if base is not None else None)
    if oauth is None:
        return None
    raw_headers = _configured_headers(config, extension)
    auth_header = _nn(
        extension.get("authHeader") if extension else None,
        config.get("authHeader") if config else None,
        False,
    )
    inner_to_auth = oauth.to_auth

    async def to_auth(credential: OAuthCredential) -> ModelAuth:
        auth = await inner_to_auth(credential)
        env = credential.extra.get("env")
        headers = await resolve_headers_or_throw(
            raw_headers,
            f'provider "{provider_id}"',
            env if isinstance(env, dict) else None,
        )
        return _with_configured_auth(auth, headers, auth_header)

    return replace(oauth, to_auth=to_auth)


def _raw_model_headers(
    model: Model,
    config: dict[str, Any] | None,
    extension: ProviderConfigInput | None,
) -> dict[str, str] | None:
    definition = next(
        (entry for entry in (config.get("models") if config else None) or [] if entry["id"] == model.id), None
    )
    extension_model = next(
        (entry for entry in (extension.get("models") if extension else None) or [] if entry["id"] == model.id), None
    )
    override = (config.get("modelOverrides") if config else None) or {}
    headers = {
        **((override.get(model.id) or {}).get("headers") or {}),
        **((definition or {}).get("headers") or {}),
        **((extension_model or {}).get("headers") or {}),
    }
    return headers if headers else None


def validate_extension_provider(
    provider_id: str,
    base: Provider | None,
    models_config: dict[str, Any] | None,
    extension: ProviderConfigInput,
) -> None:
    if extension.get("streamSimple") and not extension.get("api"):
        raise Exception(f'Provider {provider_id}: "api" is required when registering streamSimple.')
    apply_extension(
        provider_id,
        apply_models_json(provider_id, base.get_models() if base is not None else [], models_config),
        extension,
    )


class ComposedProvider:
    """The composed provider object pi builds as an object literal."""

    def __init__(
        self,
        *,
        provider_id: str,
        base: Provider | None,
        config: dict[str, Any] | None,
        extension: ProviderConfigInput | None,
        auth: ProviderAuth,
    ):
        self._base = base
        self._config = config
        self._extension = extension
        self._extension_oauth_credential: OAuthCredential | None = None
        self._refreshed_extension_models: list[dict[str, Any]] | None = None

        self.id = provider_id
        extension_oauth = extension.get("oauth") if extension else None
        self.name = _nn(
            extension.get("name") if extension else None,
            config.get("name") if config else None,
            base.name if base is not None else None,
            extension_oauth.name if extension_oauth else None,
            provider_id,
        )
        self.base_url = _nn(
            extension.get("baseUrl") if extension else None,
            config.get("baseUrl") if config else None,
            base.base_url if base is not None else None,
        )
        self.headers = base.headers if base is not None else None
        self.auth = auth
        self.filter_models = base.filter_models if base is not None else None

    def _current_extension(self) -> ProviderConfigInput | None:
        if self._extension and self._refreshed_extension_models is not None:
            return {**self._extension, "models": self._refreshed_extension_models}
        return self._extension

    def get_models(self) -> list[Model]:
        # models.json modelOverrides are the topmost user-config layer: they apply once,
        # after custom-model upserts, extension model replacement, and legacy OAuth projection.
        models = apply_extension(
            self.id,
            apply_models_json(self.id, self._base.get_models() if self._base is not None else [], self._config),
            self._current_extension(),
        )
        extension_oauth = self._extension.get("oauth") if self._extension else None
        if self._extension_oauth_credential is not None and extension_oauth and extension_oauth.modify_models:
            models = extension_oauth.modify_models(models, self._extension_oauth_credential)
        overrides = (self._config.get("modelOverrides") if self._config else None) or {}
        return [
            apply_model_override(model, overrides[model.id]) if model.id in overrides else model for model in models
        ]

    @property
    def has_dynamic_models(self) -> bool:
        extension_oauth = self._extension.get("oauth") if self._extension else None
        return bool(
            (self._base is not None and self._base.has_dynamic_models)
            or (self._extension and self._extension.get("refreshModels"))
            or (extension_oauth and extension_oauth.modify_models)
        )

    async def refresh_models(self, context: RefreshModelsContext) -> None:
        if self._base is not None and self._base.has_dynamic_models:
            await self._base.refresh_models(context)
        extension_refresh = self._extension.get("refreshModels") if self._extension else None
        refreshed: list[dict[str, Any]] | None = None
        if extension_refresh is not None:
            refreshed = await extension_refresh(context)
        if context.cancel.cancelled:
            return
        oauth_credential = context.credential if isinstance(context.credential, OAuthCredential) else None

        def _apply() -> None:
            if refreshed is not None:
                # Validate before publishing the new synchronous list.
                apply_extension(
                    self.id,
                    apply_models_json(self.id, self._base.get_models() if self._base is not None else [], self._config),
                    {**(self._extension or {}), "models": refreshed},
                )
                self._refreshed_extension_models = refreshed
            self._extension_oauth_credential = oauth_credential

        await context.publish(ModelsPublication(update=_apply))

    def _supports_base_api(self, model: Model) -> bool:
        if self._base is None:
            return False
        return any(entry.api == model.api for entry in self._base.get_models())

    def _stream_with(self, model: Model, context: Any, options: Any, simple: bool) -> Any:
        async def setup() -> Any:
            extension = self._extension
            if extension and extension.get("streamSimple") and model.api == extension.get("api"):
                return extension["streamSimple"](model, context, options)
            if self._base is not None and self._supports_base_api(model):
                return (
                    self._base.stream_simple(model, context, options)
                    if simple
                    else self._base.stream(model, context, options)
                )
            api = _get_api_provider(model.api)
            if api is None:
                raise Exception(f"No API provider registered for api: {model.api}")
            return api.stream_simple(model, context, options) if simple else api.stream(model, context, options)

        return lazy_stream(model, setup)

    def stream(self, model: Model, context: Any, options: Any = None) -> Any:
        return self._stream_with(model, context, options, False)

    def stream_simple(self, model: Model, context: Any, options: Any = None) -> Any:
        return self._stream_with(model, context, options, True)

    # Native deferred methods pass straight through to the base provider
    # (pi: `provider.fetchDeferred = base?.fetchDeferred` conditional wiring).

    @property
    def supports_fetch_deferred(self) -> bool:
        return self._base is not None and self._base.supports_fetch_deferred

    @property
    def supports_cancel_deferred(self) -> bool:
        return self._base is not None and self._base.supports_cancel_deferred

    def fetch_deferred(self, model: Model, handle: Any, options: Any = None) -> Any:
        return self._base.fetch_deferred(model, handle, options)

    async def cancel_deferred(self, model: Model, handle: Any, options: Any = None) -> None:
        await self._base.cancel_deferred(model, handle, options)


def compose_model_provider(
    provider_id: str,
    base: Provider | None,
    model_config: ModelConfig,
    extension: ProviderConfigInput | None,
) -> ComposedProvider:
    """Compose built-in, models.json, and extension layers without reading credentials."""
    config = model_config.get_provider(provider_id)
    api_key = compose_api_key_auth(provider_id, base, config, extension)
    oauth = compose_oauth_auth(provider_id, base, config, extension)
    if api_key is None and oauth is None:
        raise Exception(f"Provider {provider_id}: no authentication method configured.")
    provider = ComposedProvider(
        provider_id=provider_id,
        base=base,
        config=config,
        extension=extension,
        auth=ProviderAuth(api_key=api_key, oauth=oauth),
    )
    # Validate eagerly so registration/reload reports structural errors immediately.
    provider.get_models()
    return provider


async def resolve_configured_model_headers(
    model: Model,
    config: dict[str, Any] | None,
    extension: ProviderConfigInput | None,
    env: dict[str, str] | None = None,
) -> dict[str, str] | None:
    return await resolve_headers_or_throw(
        _raw_model_headers(model, config, extension), f'model "{model.provider}/{model.id}"', env
    )


async def resolve_compatibility_request_config(
    model: Model,
    config: dict[str, Any] | None,
    extension: ProviderConfigInput | None,
) -> CompatibilityRequestConfig:
    raw = {**(_configured_headers(config, extension) or {}), **(_raw_model_headers(model, config, extension) or {})}
    configured = await resolve_headers_or_throw(raw if raw else None, f'model "{model.provider}/{model.id}"')
    headers = {**(model.headers or {}), **(configured or {})} if model.headers or configured else None
    return CompatibilityRequestConfig(
        headers=headers,
        auth_header=_nn(
            extension.get("authHeader") if extension else None,
            config.get("authHeader") if config else None,
            False,
        ),
    )


def configured_request_auth_status(
    config: dict[str, Any] | None,
    extension: ProviderConfigInput | None,
) -> AuthStatus | None:
    value = _configured_api_key(config, extension)
    if value is None:
        return None
    if is_command_config_value(value):
        return AuthStatus(configured=True, source="models_json_command")
    names = get_config_value_env_var_names(value)
    if names:
        if is_config_value_configured(value):
            return AuthStatus(configured=True, source="environment", label=", ".join(names))
        return AuthStatus(configured=False)
    source = "fallback" if extension is not None and extension.get("apiKey") is not None else "models_json_key"
    return AuthStatus(configured=True, source=source)
