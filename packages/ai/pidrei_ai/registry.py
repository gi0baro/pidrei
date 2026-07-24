"""Port of pi's models registry (packages/ai/src/models.ts).

`Models` is the runtime collection of providers plus auth application and
stream convenience; providers own stream behavior, `Models` resolves auth and
delegates each request to the provider that owns the model.
"""

import inspect
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any

import tonio.colored as tonio

from pidrei_ai.api.lazy import lazy_stream
from pidrei_ai.auth.context import default_provider_auth_context
from pidrei_ai.auth.credential_store import InMemoryCredentialStore
from pidrei_ai.auth.resolve import AuthResolutionOverrides, ModelsError, resolve_provider_auth
from pidrei_ai.auth.types import (
    ApiKeyCredential,
    AuthCheck,
    AuthContext,
    AuthInteraction,
    AuthResult,
    AuthType,
    Credential,
    CredentialStore,
    ProviderAuth,
)
from pidrei_ai.models_store import InMemoryModelsStore, ModelsStore, ModelsStoreEntry, ProviderModelsStore
from pidrei_ai.types import (
    Context,
    Model,
    ModelThinkingLevel,
    ProviderHeaders,
    SimpleStreamOptions,
    StreamOptions,
    Usage,
    UsageCost,
)
from pidrei_ai.utils.cancel import CancelToken
from pidrei_ai.utils.event_stream import AssistantMessageEventStream
from pidrei_ai.utils.headers import merge_headers


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass(slots=True)
class RefreshModelsContext:
    # Persistent model storage scoped to this provider ID.
    store: ProviderModelsStore
    # False during offline/cache-only initialization.
    allow_network: bool
    # Effective configured credential. OAuth credentials are refreshed before network access.
    credential: Credential | None = None
    # Bypass provider freshness checks and fetch immediately when network access is allowed.
    force: bool = False
    cancel: CancelToken | None = None


@dataclass(slots=True)
class ModelsRefreshOptions:
    allow_network: bool = True
    force: bool = False
    cancel: CancelToken | None = None


@dataclass(slots=True)
class ModelsRefreshResult:
    aborted: bool
    errors: dict[str, Exception]


class _ScopedModelsStore:
    __slots__ = ("_provider_id", "_store")

    def __init__(self, store: ModelsStore, provider_id: str):
        self._store = store
        self._provider_id = provider_id

    async def read(self) -> ModelsStoreEntry | None:
        return await self._store.read(self._provider_id)

    async def write(self, entry: ModelsStoreEntry) -> None:
        await self._store.write(self._provider_id, entry)

    async def delete(self) -> None:
        await self._store.delete(self._provider_id)


class Provider:
    """The concrete runtime unit built by `create_provider`: owns id/name/base
    metadata, auth methods, model listing, and stream behavior.
    """

    def __init__(
        self,
        *,
        id: str,
        name: str | None = None,
        base_url: str | None = None,
        headers: ProviderHeaders | None = None,
        auth: ProviderAuth,
        models: list[Model],
        fetch_models: Callable[[RefreshModelsContext], Awaitable[list[Model]]] | None = None,
        filter_models: Callable[[list[Model], Credential | None], list[Model]] | None = None,
        api: Any,
    ):
        self.id = id
        self.name = name if name is not None else id
        self.base_url = base_url
        self.headers = headers
        self.auth = auth
        self.filter_models = filter_models
        self._baseline_models = models
        self._dynamic_models: list[Model] = []
        self._fetch_models = fetch_models
        self._inflight_guard = threading.Lock()
        self._inflight: tuple[Any, list[BaseException | None]] | None = None

        single = getattr(api, "stream", None)
        self._single = api if callable(single) else None
        self._by_api: dict[str, Any] | None = None if self._single is not None else dict(api)

    @property
    def has_dynamic_models(self) -> bool:
        return self._fetch_models is not None

    def get_models(self) -> list[Model]:
        """Baseline catalog with the dynamic overlay merged in by model id."""
        merged = list(self._baseline_models)
        for model in self._dynamic_models:
            for index, entry in enumerate(merged):
                if entry.id == model.id:
                    merged[index] = model
                    break
            else:
                merged.append(model)
        return merged

    async def refresh_models(self, context: RefreshModelsContext) -> None:
        """Restore the stored catalog and optionally fetch a newer list.

        Concurrent callers share one in-flight refresh (pi dedupes through a
        shared promise; errors propagate to every caller).
        """
        if self._fetch_models is None:
            return

        with self._inflight_guard:
            inflight = self._inflight
            if inflight is None:
                done = tonio.Event()
                box: list[BaseException | None] = [None]
                self._inflight = (done, box)
        if inflight is not None:
            done, box = inflight
            await done.wait()
            if box[0] is not None:
                raise box[0]
            return

        try:
            stored = await context.store.read()
            if stored is not None:
                self._dynamic_models = [model for model in stored.models if model.provider == self.id]
            if not context.allow_network or (context.cancel is not None and context.cancel.cancelled):
                return
            refreshed = await self._fetch_models(context)
            if context.cancel is not None and context.cancel.cancelled:
                return
            self._dynamic_models = list(refreshed)
            await context.store.write(ModelsStoreEntry(models=list(refreshed), checked_at=_now_ms()))
        except BaseException as error:
            box[0] = error
            raise
        finally:
            with self._inflight_guard:
                self._inflight = None
            done.set()

    def _api_for(self, model: Model) -> Any | None:
        if self._single is not None:
            return self._single
        return self._by_api.get(model.api) if self._by_api is not None else None

    def _dispatch(self, model: Model, run: Callable[[Any], AssistantMessageEventStream]) -> AssistantMessageEventStream:
        streams = self._api_for(model)
        if streams is None:

            async def _fail() -> Any:
                raise ModelsError("stream", f'Provider {self.id} has no API implementation for "{model.api}"')

            return lazy_stream(model, _fail)
        return run(streams)

    def stream(self, model: Model, context: Context, options: StreamOptions | None = None):
        return self._dispatch(model, lambda streams: streams.stream(model, context, options))

    def stream_simple(self, model: Model, context: Context, options: SimpleStreamOptions | None = None):
        return self._dispatch(model, lambda streams: streams.stream_simple(model, context, options))


def create_provider(
    *,
    id: str,
    name: str | None = None,
    base_url: str | None = None,
    headers: ProviderHeaders | None = None,
    auth: ProviderAuth,
    models: list[Model],
    fetch_models: Callable[[RefreshModelsContext], Awaitable[list[Model]]] | None = None,
    filter_models: Callable[[list[Model], Credential | None], list[Model]] | None = None,
    api: Any,
) -> Provider:
    """Build a provider from parts. Built-in provider factories and models.json
    custom providers both go through this. A single `api` streams all models;
    an `api` dict dispatches on `model.api`, and a model whose api has no entry
    produces a stream error.
    """
    return Provider(
        id=id,
        name=name,
        base_url=base_url,
        headers=headers,
        auth=auth,
        models=models,
        fetch_models=fetch_models,
        filter_models=filter_models,
        api=api,
    )


class Models:
    """Port of pi's `ModelsImpl` (mutable: `set_provider`/`delete_provider`)."""

    def __init__(
        self,
        *,
        credentials: CredentialStore | None = None,
        models_store: ModelsStore | None = None,
        auth_context: AuthContext | None = None,
    ):
        self._providers: dict[str, Provider] = {}
        self._providers_guard = threading.Lock()
        self._credentials = credentials if credentials is not None else InMemoryCredentialStore()
        self._models_store = models_store if models_store is not None else InMemoryModelsStore()
        self._auth_context = auth_context if auth_context is not None else default_provider_auth_context()

    # -- provider collection ---------------------------------------------------

    def set_provider(self, provider: Provider) -> None:
        with self._providers_guard:
            self._providers[provider.id] = provider

    def delete_provider(self, id: str) -> None:
        with self._providers_guard:
            self._providers.pop(id, None)

    def clear_providers(self) -> None:
        with self._providers_guard:
            self._providers.clear()

    def get_providers(self) -> list[Provider]:
        with self._providers_guard:
            return list(self._providers.values())

    def get_provider(self, id: str) -> Provider | None:
        with self._providers_guard:
            return self._providers.get(id)

    def get_models(self, provider: str | None = None) -> list[Model]:
        """Sync read of last-known models. Best-effort: a provider whose
        `get_models()` raises yields no models."""
        if provider is not None:
            entry = self.get_provider(provider)
            if entry is None:
                return []
            try:
                return entry.get_models()
            except Exception:
                return []

        models: list[Model] = []
        for entry in self.get_providers():
            try:
                models.extend(entry.get_models())
            except Exception:
                pass  # Best-effort: ill-behaved providers yield no models.
        return models

    def get_model(self, provider: str, id: str) -> Model | None:
        for model in self.get_models(provider):
            if model.id == id:
                return model
        return None

    # -- refresh ---------------------------------------------------------------

    async def refresh(self, options: ModelsRefreshOptions | None = None) -> ModelsRefreshResult:
        """Refresh every configured dynamic provider concurrently. Provider
        errors and cancellation are returned without raising."""
        options = options or ModelsRefreshOptions()
        allow_network = options.allow_network
        cancel = options.cancel
        refreshable = [provider for provider in self.get_providers() if provider.has_dynamic_models]

        async def refresh_one(provider: Provider) -> tuple[str, Exception | None]:
            if cancel is not None and cancel.cancelled:
                return provider.id, None
            store = _ScopedModelsStore(self._models_store, provider.id)
            stored: Credential | None = None
            try:
                stored = await self._read_credential(provider.id)
                credential = await self._resolve_refresh_credential(provider, stored, allow_network, cancel)
                if credential is None:
                    return provider.id, None
                await provider.refresh_models(
                    RefreshModelsContext(
                        credential=credential,
                        store=store,
                        allow_network=allow_network,
                        force=options.force,
                        cancel=cancel,
                    )
                )
                return provider.id, None
            except Exception as error:
                recorded: Exception | None = None
                if cancel is None or not cancel.cancelled:
                    recorded = (
                        error
                        if isinstance(error, Exception)
                        else ModelsError("model_source", f"Model refresh failed for {provider.id}", cause=error)
                    )
                try:
                    await provider.refresh_models(
                        RefreshModelsContext(credential=stored, store=store, allow_network=False, cancel=cancel)
                    )
                except Exception:
                    pass  # Preserve the original error; cache restoration is best-effort here.
                return provider.id, recorded

        errors: dict[str, Exception] = {}
        if refreshable:
            results = await tonio.spawn(*[refresh_one(provider) for provider in refreshable])
            if len(refreshable) == 1:
                results = [results]
            for provider_id, error in results:
                if error is not None:
                    errors[provider_id] = error

        return ModelsRefreshResult(aborted=cancel.cancelled if cancel is not None else False, errors=errors)

    async def _resolve_refresh_credential(
        self,
        provider: Provider,
        stored: Credential | None,
        allow_network: bool,
        cancel: CancelToken | None,
    ) -> Credential | None:
        if stored is not None and stored.type == "oauth":
            oauth = provider.auth.oauth
            if oauth is None:
                return None
            if not allow_network or _now_ms() < stored.expires:
                return stored
            if cancel is not None and cancel.cancelled:
                return None

            async def _refresh(current: Credential | None) -> Credential | None:
                if current is None or current.type != "oauth" or _now_ms() < current.expires:
                    return None
                return await oauth.refresh(current, cancel)

            post = await self._credentials.modify(provider.id, _refresh)
            return post if post is not None and post.type == "oauth" else None

        api_key = provider.auth.api_key
        if api_key is None:
            return None
        credential = stored if stored is not None and stored.type == "api_key" else None
        result = await api_key.resolve(self._auth_context, credential)
        if result is None:
            return None
        return ApiKeyCredential(key=result.auth.api_key, env=result.env)

    # -- auth ------------------------------------------------------------------

    async def _read_credential(self, provider_id: str) -> Credential | None:
        try:
            return await self._credentials.read(provider_id)
        except Exception as error:
            raise ModelsError("auth", f"Credential store read failed for {provider_id}", cause=error)

    async def _check_provider_auth(self, provider: Provider, credential: Credential | None) -> AuthCheck | None:
        if credential is not None and credential.type == "oauth":
            return AuthCheck(source="OAuth", type="oauth") if provider.auth.oauth is not None else None
        api_key = provider.auth.api_key
        if api_key is None:
            return None
        if api_key.check is not None:
            try:
                return await api_key.check(
                    self._auth_context,
                    credential if credential is not None and credential.type == "api_key" else None,
                )
            except Exception as error:
                raise ModelsError("auth", f"API key auth check failed for provider {provider.id}", cause=error)

        resolution = await resolve_provider_auth(provider, self._credentials, self._auth_context)
        return AuthCheck(source=resolution.source, type="api_key") if resolution is not None else None

    async def check_auth(self, provider_id: str) -> AuthCheck | None:
        """Check whether a provider has complete auth configuration without refreshing OAuth."""
        provider = self.get_provider(provider_id)
        if provider is None:
            return None
        return await self._check_provider_auth(provider, await self._read_credential(provider_id))

    async def get_available(self, provider_id: str | None = None) -> list[Model]:
        """Return models whose providers have complete auth configuration."""
        providers = (
            [entry for entry in [self.get_provider(provider_id)] if entry is not None]
            if provider_id
            else self.get_providers()
        )

        async def check_one(provider: Provider) -> tuple[Provider, Credential | None, AuthCheck | None]:
            credential = await self._read_credential(provider.id)
            return provider, credential, await self._check_provider_auth(provider, credential)

        if not providers:
            return []
        checks = await tonio.spawn(*[check_one(provider) for provider in providers])
        if len(providers) == 1:
            checks = [checks]

        available: list[Model] = []
        for provider, credential, auth in checks:
            if auth is None:
                continue
            models = provider.get_models()
            available.extend(provider.filter_models(models, credential) if provider.filter_models else models)
        return available

    async def get_auth(
        self,
        provider_or_model: str | Model,
        overrides: AuthResolutionOverrides | None = None,
    ) -> AuthResult | None:
        """Resolve provider-scoped auth by provider id, or provider auth plus
        static model headers when passed a model."""
        provider_id = provider_or_model if isinstance(provider_or_model, str) else provider_or_model.provider
        provider = self.get_provider(provider_id)
        if provider is None:
            return None
        result = await resolve_provider_auth(provider, self._credentials, self._auth_context, overrides)
        if result is None or isinstance(provider_or_model, str) or not provider_or_model.headers:
            return result
        return replace(
            result,
            auth=replace(result.auth, headers=merge_headers(result.auth.headers, provider_or_model.headers)),
        )

    async def login(self, provider_id: str, type: AuthType, interaction: AuthInteraction) -> Credential:
        """Run a provider-owned login flow and persist its returned credential."""
        provider = self.get_provider(provider_id)
        if provider is None:
            raise ModelsError("provider", f"Unknown provider: {provider_id}")
        method = provider.auth.oauth if type == "oauth" else provider.auth.api_key
        login = getattr(method, "login", None) if method is not None else None
        if login is None:
            raise ModelsError("auth", f"{provider.name} does not support {type} login")
        credential = await login(interaction)

        async def _persist(_current: Credential | None) -> Credential | None:
            return credential

        try:
            await self._credentials.modify(provider_id, _persist)
        except Exception as error:
            raise ModelsError("auth", f"Credential store modify failed for {provider_id}", cause=error)
        return credential

    async def logout(self, provider_id: str) -> None:
        try:
            await self._credentials.delete(provider_id)
        except Exception as error:
            raise ModelsError("auth", f"Credential store delete failed for {provider_id}", cause=error)

    # -- streaming -------------------------------------------------------------

    def _require_provider(self, model: Model) -> Provider:
        provider = self.get_provider(model.provider)
        if provider is None:
            raise ModelsError("provider", f"Unknown provider: {model.provider}")
        return provider

    async def _apply_auth(
        self,
        model: Model,
        options: StreamOptions | None,
    ) -> tuple[Model, StreamOptions]:
        self._require_provider(model)
        resolution = await self.get_auth(
            model,
            AuthResolutionOverrides(
                api_key=options.api_key if options is not None else None,
                env=options.env if options is not None else None,
            ),
        )
        if resolution is None:
            raise ModelsError("auth", f"Provider is not configured: {model.provider}")
        auth = resolution.auth

        # Explicit request options win per-field; the Models-only transform runs last.
        api_key = options.api_key if options is not None and options.api_key is not None else auth.api_key
        headers = merge_headers(auth.headers, options.headers if options is not None else None)
        if options is not None and options.transform_headers is not None:
            transformed = options.transform_headers(headers if headers is not None else {})
            headers = await transformed if inspect.isawaitable(transformed) else transformed
        options_env = options.env if options is not None else None
        env = {**(resolution.env or {}), **(options_env or {})} if resolution.env or options_env else None

        request_model = replace(model, base_url=auth.base_url) if auth.base_url else model
        request_options = replace(
            options if options is not None else StreamOptions(),
            api_key=api_key,
            headers=headers,
            env=env,
            transform_headers=None,
        )
        return request_model, request_options

    def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        async def _setup():
            provider = self._require_provider(model)
            request_model, request_options = await self._apply_auth(
                model, options if options is not None else StreamOptions()
            )
            return provider.stream(request_model, context, request_options)

        return lazy_stream(model, _setup)

    async def complete(self, model: Model, context: Context, options: StreamOptions | None = None):
        return await self.stream(model, context, options).result()

    def stream_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        async def _setup():
            provider = self._require_provider(model)
            request_model, request_options = await self._apply_auth(
                model, options if options is not None else SimpleStreamOptions()
            )
            return provider.stream_simple(request_model, context, request_options)

        return lazy_stream(model, _setup)

    async def complete_simple(self, model: Model, context: Context, options: SimpleStreamOptions | None = None):
        return await self.stream_simple(model, context, options).result()


def create_models(
    *,
    credentials: CredentialStore | None = None,
    models_store: ModelsStore | None = None,
    auth_context: AuthContext | None = None,
) -> Models:
    return Models(credentials=credentials, models_store=models_store, auth_context=auth_context)


# -- model helpers -------------------------------------------------------------


def has_api(model: Model, api: str) -> bool:
    """Runtime narrowing check for dynamically looked-up models."""
    return model.api == api


def calculate_cost(model: Model, usage: Usage) -> UsageCost:
    """Compute request cost into `usage.cost` (mutates and returns it)."""
    input_tokens = usage.input + usage.cache_read + usage.cache_write
    rates_input = model.cost.input
    rates_output = model.cost.output
    rates_cache_read = model.cost.cache_read
    rates_cache_write = model.cost.cache_write
    matched_threshold = -1
    for tier in model.cost.tiers or []:
        if input_tokens > tier.input_tokens_above and tier.input_tokens_above > matched_threshold:
            rates_input = tier.input
            rates_output = tier.output
            rates_cache_read = tier.cache_read
            rates_cache_write = tier.cache_write
            matched_threshold = tier.input_tokens_above

    # Anthropic charges 2x base input for 1h cache writes.
    long_write = usage.cache_write_1h or 0
    short_write = usage.cache_write - long_write
    usage.cost.input = (rates_input / 1_000_000) * usage.input
    usage.cost.output = (rates_output / 1_000_000) * usage.output
    usage.cost.cache_read = (rates_cache_read / 1_000_000) * usage.cache_read
    usage.cost.cache_write = (rates_cache_write * short_write + rates_input * 2 * long_write) / 1_000_000
    usage.cost.total = usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write
    return usage.cost


_EXTENDED_THINKING_LEVELS: list[ModelThinkingLevel] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"]


def get_supported_thinking_levels(model: Model) -> list[ModelThinkingLevel]:
    if not model.reasoning:
        return ["off"]

    mapping = model.thinking_level_map

    def supported(level: ModelThinkingLevel) -> bool:
        # A null mapping marks the level unsupported; a *missing* key uses
        # provider defaults — except xhigh/max, which require an explicit entry.
        present = mapping is not None and level in mapping
        if present and mapping[level] is None:  # type: ignore[index]
            return False
        if level in ("xhigh", "max"):
            return present
        return True

    return [level for level in _EXTENDED_THINKING_LEVELS if supported(level)]


def clamp_thinking_level(model: Model, level: ModelThinkingLevel) -> ModelThinkingLevel:
    available_levels = get_supported_thinking_levels(model)
    if level in available_levels:
        return level

    if level not in _EXTENDED_THINKING_LEVELS:
        return available_levels[0] if available_levels else "off"
    requested_index = _EXTENDED_THINKING_LEVELS.index(level)

    for candidate in _EXTENDED_THINKING_LEVELS[requested_index:]:
        if candidate in available_levels:
            return candidate
    for candidate in reversed(_EXTENDED_THINKING_LEVELS[:requested_index]):
        if candidate in available_levels:
            return candidate
    return available_levels[0] if available_levels else "off"


def models_are_equal(a: Model | None, b: Model | None) -> bool:
    """Check if two models are equal by comparing both their id and provider."""
    if a is None or b is None:
        return False
    return a.id == b.id and a.provider == b.provider
