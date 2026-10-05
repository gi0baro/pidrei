"""Port of pi's models registry (packages/ai/src/models.ts).

`Models` is the runtime collection of providers plus auth application and
request convenience; providers own request behavior (streaming, image
generation, classification), `Models` resolves auth and delegates each request
to the provider that owns the model.

Read accessors come in three flavors: the unqualified ones (`get_models`,
`get_model`, `get_available`) return chat models, the `*_of_type` accessors
return one model type, and `get_all_models`/`get_all_available` return every
type.

pi's optional provider members (`getAllModels?`, `filterAllModels?`,
`generateImages?`, `classify?`) are attributes here that are None when
absent, the way `filter_models` already is.
"""

import copy
import threading
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from types import EllipsisType
from typing import Any

import tonio.colored as tonio
from tonio.colored import sync

from pidrei_ai.api.lazy import _cancel_of, call_stream_into, lazy_stream
from pidrei_ai.auth.context import default_provider_auth_context
from pidrei_ai.auth.credential_store import InMemoryCredentialStore
from pidrei_ai.auth.resolve import (
    AuthResolutionOverrides,
    ModelsError,
    refresh_stored_oauth_credential,
    resolve_provider_auth,
)
from pidrei_ai.auth.types import (
    ApiKeyCredential,
    AuthCheck,
    AuthContext,
    AuthEvent,
    AuthInteraction,
    AuthOperationOptions,
    AuthPrompt,
    AuthResult,
    AuthType,
    Credential,
    CredentialStore,
    LoginOptions,
    ProviderAuth,
)
from pidrei_ai.builders import UsageBuilder, UsageCostBuilder
from pidrei_ai.models_store import InMemoryModelsStore, ModelsStore, ModelsStoreEntry, ModelsStoreOperationOptions
from pidrei_ai.types import (
    AnyModel,
    AssistantImages,
    AssistantMessage,
    ClassifierContext,
    ClassifierModel,
    ClassifierOptions,
    ClassifierResult,
    Context,
    DeferredCancelOptions,
    DeferredFetchOptions,
    DeferredHandle,
    ImageModel,
    ImagesContext,
    ImagesOptions,
    Model,
    ModelThinkingLevel,
    ModelType,
    ProviderHeaders,
    ProviderRequestOptions,
    SimpleStreamOptions,
    StreamOptions,
    TranscriptContext,
)
from pidrei_ai.utils.abort import operation_cancel, race_with_cancel
from pidrei_ai.utils.event_stream import AssistantMessageEventStream
from pidrei_ai.utils.headers import merge_headers
from pidrei_ai.utils.model_operations import (
    assert_chat_model,
    assert_classifier_model,
    assert_image_model,
    classifier_error_result,
    get_model_type,
    image_error_result,
    is_model_type,
)
from pidrei_ai.utils.transcript import normalize_context
from pidrei_utils import clock
from pidrei_utils.cancel import CancelToken, combine_cancel_tokens


_KNOWN_MODEL_TYPES: frozenset[ModelType] = frozenset(("chat", "image", "classifier"))


def _has_known_model_type(model: AnyModel) -> bool:
    """Models from stores and remote sources may have types that only newer versions know."""
    return get_model_type(model) in _KNOWN_MODEL_TYPES


def _with_known_model_types(entry: ModelsStoreEntry) -> ModelsStoreEntry:
    """Drops stored models whose type this version does not know."""
    return replace(entry, models=[model for model in entry.models if _has_known_model_type(model)])


def _provider_all_models(provider: Any) -> list[AnyModel]:
    """pi: `provider.getAllModels?.() ?? provider.getModels()`."""
    get_all_models = provider.get_all_models
    return get_all_models() if get_all_models is not None else provider.get_models()


@dataclass(slots=True)
class ModelsPublication:
    # Provider-selected persisted catalog. Leave as the `...` sentinel to keep
    # storage unchanged; None deletes it (pi's `persist?: entry | null`).
    persist: ModelsStoreEntry | None | EllipsisType = ...
    # Optional synchronous update of provider-private in-memory catalog state.
    update: Callable[[], None] | None = None


@dataclass(slots=True)
class RefreshModelsContext:
    # Generation-checked publication. Persistence policy remains provider-owned;
    # the update runs synchronously only after the selected persistence mutation.
    publish: Callable[[ModelsPublication], Awaitable[bool]]
    # False during offline/cache-only initialization.
    allow_network: bool
    # Always present, including when the public refresh caller omits its optional cancel.
    cancel: CancelToken
    # Effective configured credential. OAuth credentials are refreshed before network access.
    credential: Credential | None = None
    # Immutable provider-scoped catalog snapshot captured before this refresh phase.
    stored: ModelsStoreEntry | None = None
    # Bypass provider freshness checks and fetch immediately when network access is allowed.
    force: bool | None = None


@dataclass(slots=True)
class ModelsRefreshOptions:
    allow_network: bool = True
    # Restrict refresh to these provider IDs. Unknown and static providers are ignored.
    providers: list[str] | None = None
    # Bypass provider freshness checks and fetch immediately when network access is allowed.
    force: bool = False
    cancel: CancelToken | None = None


@dataclass(slots=True)
class ModelsRefreshResult:
    aborted: bool
    errors: dict[str, Exception]


class Provider:
    """The concrete runtime unit built by `create_provider`: owns id/name/base
    metadata, auth methods, model listing, and the operations its models
    support (streaming, image generation, classification).
    """

    def __init__(
        self,
        *,
        id: str,
        name: str | None = None,
        base_url: str | None = None,
        headers: ProviderHeaders | None = None,
        auth: ProviderAuth,
        models: list[AnyModel],
        fetch_models: Callable[[RefreshModelsContext], Awaitable[list[AnyModel]]] | None = None,
        filter_models: Callable[[list[Model], Credential | None], list[Model]] | None = None,
        filter_all_models: Callable[[list[AnyModel], Credential | None], list[AnyModel]] | None = None,
        api: Any = None,
        images: Mapping[str, Any] | None = None,
        classifiers: Mapping[str, Any] | None = None,
    ):
        single = getattr(api, "stream", None)
        self._single = api if callable(single) else None
        self._by_api: dict[str, Any] | None = None if self._single is not None or api is None else dict(api)
        self._images = {key: value for key, value in (images or {}).items() if value is not None}
        self._classifiers = {key: value for key, value in (classifiers or {}).items() if value is not None}
        if not self._stream_entries() and not self._images and not self._classifiers:
            raise Exception(f'Provider {id}: at least one of "api", "images", or "classifiers" is required.')

        self.id = id
        self.name = name if name is not None else id
        self.base_url = base_url
        self.headers = headers
        self.auth = auth
        self.filter_models = filter_models
        self.filter_all_models = filter_all_models
        # Present when the provider supports dedicated image models / structured
        # classifier models. Never raise.
        self.generate_images = self._generate_images if self._images else None
        self.classify = self._classify if self._classifiers else None
        self._baseline_models = models
        self._dynamic_models: list[AnyModel] = []
        self._fetch_models = fetch_models

    @property
    def has_dynamic_models(self) -> bool:
        return self._fetch_models is not None

    def get_all_models(self) -> list[AnyModel]:
        return self._current_models()

    def get_models(self) -> list[Model]:
        """The chat models of the catalog (read independently of `get_all_models`)."""
        return [model for model in self._current_models() if is_model_type(model, "chat")]

    def _current_models(self) -> list[AnyModel]:
        """Baseline catalog with the dynamic overlay merged in by model type and id."""
        merged = list(self._baseline_models)
        for model in self._dynamic_models:
            for index, entry in enumerate(merged):
                if get_model_type(entry) == get_model_type(model) and entry.id == model.id:
                    merged[index] = model
                    break
            else:
                merged.append(model)
        return merged

    async def refresh_models(self, context: RefreshModelsContext) -> None:
        """Restore `context.stored` and optionally fetch a newer list using the
        effective credential, publishing persistence and synchronous state
        changes through the generation-checked `context.publish()`."""
        if self._fetch_models is None:
            return

        if context.stored is not None:
            restored = [model for model in context.stored.models if model.provider == self.id]

            def _apply_restored(restored: list[AnyModel] = restored) -> None:
                self._dynamic_models = restored

            if not await context.publish(ModelsPublication(update=_apply_restored)):
                return
        if not context.allow_network or context.cancel.cancelled:
            return
        fetched = await self._fetch_models(context)
        if context.cancel.cancelled:
            return
        refreshed = [model for model in fetched if _has_known_model_type(model)]

        def _apply_refreshed() -> None:
            self._dynamic_models = list(refreshed)

        await context.publish(
            ModelsPublication(
                persist=ModelsStoreEntry(models=list(refreshed), checked_at=clock.now_ms()),
                update=_apply_refreshed,
            )
        )

    def _api_for(self, model: Model) -> Any | None:
        if self._single is not None:
            return self._single
        return self._by_api.get(model.api) if self._by_api is not None else None

    def _dispatch(
        self,
        model: Model,
        method: str,
        args: tuple[Any, ...],
        options: Any,
        into: AssistantMessageEventStream | None,
    ) -> AssistantMessageEventStream:
        streams = self._api_for(model)
        if streams is None:

            async def _fail(_stream: AssistantMessageEventStream) -> Any:
                raise ModelsError("stream", f'Provider {self.id} has no API implementation for "{model.api}"')

            return lazy_stream(model, _fail, _cancel_of(options), into=into)
        if into is None:
            return getattr(streams, method)(*args, options)
        return call_stream_into(getattr(streams, method), *args, options, into=into)

    def stream(
        self,
        model: Model,
        context: TranscriptContext,
        options: StreamOptions | None = None,
        *,
        into: AssistantMessageEventStream | None = None,
    ):
        """Stream a normalized transcript. `Models` normalizes the caller's `Context` before dispatching here."""
        return self._dispatch(model, "stream", (model, context), options, into)

    def stream_simple(
        self,
        model: Model,
        context: TranscriptContext,
        options: SimpleStreamOptions | None = None,
        *,
        into: AssistantMessageEventStream | None = None,
    ):
        return self._dispatch(model, "stream_simple", (model, context), options, into)

    # -- deferred responses ----------------------------------------------------
    # pi attaches `fetchDeferred`/`cancelDeferred` to the provider object only
    # when some streams entry declares them; here presence maps to the
    # `supports_*` flags and the methods themselves fail per-api like pi's
    # conditional wrappers do.

    def _stream_entries(self) -> list[Any]:
        if self._single is not None:
            return [self._single]
        return [entry for entry in (self._by_api or {}).values() if entry is not None]

    @property
    def supports_fetch_deferred(self) -> bool:
        return any(getattr(entry, "fetch_deferred", None) is not None for entry in self._stream_entries())

    @property
    def supports_cancel_deferred(self) -> bool:
        return any(getattr(entry, "cancel_deferred", None) is not None for entry in self._stream_entries())

    def fetch_deferred(
        self, model: Model, handle: DeferredHandle, options: DeferredFetchOptions | None = None
    ) -> AssistantMessageEventStream:
        implementation = self._api_for(model)
        fetch = getattr(implementation, "fetch_deferred", None) if implementation is not None else None
        if fetch is None:

            async def _fail(_stream: AssistantMessageEventStream) -> Any:
                raise ModelsError(
                    "provider", f'Provider {self.id} does not support deferred responses for "{model.api}"'
                )

            return lazy_stream(model, _fail, _cancel_of(options))
        return fetch(model, handle, options)

    async def cancel_deferred(
        self, model: Model, handle: DeferredHandle, options: DeferredCancelOptions | None = None
    ) -> None:
        implementation = self._api_for(model)
        cancel = getattr(implementation, "cancel_deferred", None) if implementation is not None else None
        if cancel is None:
            raise ModelsError("provider", f'Provider {self.id} cannot cancel deferred responses for "{model.api}"')
        await cancel(model, handle, options)

    # -- one-shot operations ---------------------------------------------------
    # Dispatch on `model.api` like the stream map; a model whose api has no
    # entry yields an error result.

    async def _generate_images(
        self, model: ImageModel, context: ImagesContext, options: ImagesOptions | None = None
    ) -> AssistantImages:
        implementation = self._images.get(model.api)
        if implementation is None:
            return image_error_result(
                model,
                ModelsError("provider", f'Provider {self.id} has no image generation implementation for "{model.api}"'),
            )
        return await implementation.generate_images(model, context, options)

    async def _classify(
        self, model: ClassifierModel, context: ClassifierContext, options: ClassifierOptions | None = None
    ) -> ClassifierResult:
        implementation = self._classifiers.get(model.api)
        if implementation is None:
            return classifier_error_result(
                model,
                ModelsError("provider", f'Provider {self.id} has no classifier implementation for "{model.api}"'),
            )
        return await implementation.classify(model, context, options)


def create_provider(
    *,
    id: str,
    name: str | None = None,
    base_url: str | None = None,
    headers: ProviderHeaders | None = None,
    auth: ProviderAuth,
    models: list[AnyModel],
    fetch_models: Callable[[RefreshModelsContext], Awaitable[list[AnyModel]]] | None = None,
    filter_models: Callable[[list[Model], Credential | None], list[Model]] | None = None,
    filter_all_models: Callable[[list[AnyModel], Credential | None], list[AnyModel]] | None = None,
    api: Any = None,
    images: Mapping[str, Any] | None = None,
    classifiers: Mapping[str, Any] | None = None,
) -> Provider:
    """Build a provider from parts. Built-in provider factories and models.json
    custom providers both go through this. A single `api` streams all chat
    models; an `api` dict dispatches on `model.api`, and a model whose api has
    no entry produces a stream error. The `images`/`classifiers` maps dispatch
    on `model.api` the same way. At least one concrete implementation across
    `api`/`images`/`classifiers` is required; empty maps are rejected.

    `models` and `fetch_models` carry models of every type; models without
    `type` are chat models, and fetched models of unknown types are dropped.
    `filter_models` is credential-specific chat availability, and
    `filter_all_models` the same across every model type.
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
        filter_all_models=filter_all_models,
        api=api,
        images=images,
        classifiers=classifiers,
    )


class _NormalizedAuthInteraction:
    """pi: `{ ...interaction, signal }` — the same interaction surface with a
    guaranteed cancel token (`ProviderAuthInteraction`)."""

    __slots__ = ("_base", "cancel")

    def __init__(self, base: AuthInteraction, cancel: CancelToken):
        self._base = base
        self.cancel = cancel

    def prompt(self, prompt: AuthPrompt) -> Awaitable[str]:
        return self._base.prompt(prompt)

    def notify(self, event: AuthEvent) -> None:
        self._base.notify(event)


class Models:
    """Port of pi's `ModelsImpl` (mutable: `set_provider`/`delete_provider`)."""

    def __init__(
        self,
        *,
        credentials: CredentialStore | None = None,
        models_store: ModelsStore | None = None,
        auth_context: AuthContext | None = None,
    ):
        # Immutable snapshot swapped on write (never mutated in place):
        # readers pin one attribute read and take no lock — get_provider sits
        # on every stream request. The guard serializes writers only.
        self._providers: dict[str, Provider] = {}
        self._providers_guard = threading.Lock()
        self._credentials = credentials if credentials is not None else InMemoryCredentialStore()
        self._models_store = models_store if models_store is not None else InMemoryModelsStore()
        self._auth_context = auth_context if auth_context is not None else default_provider_auth_context()
        self._refresh_state_guard = threading.Lock()
        self._refresh_generations: dict[str, int] = {}
        self._refresh_controllers: dict[str, CancelToken] = {}
        self._publication_guard = threading.Lock()
        # Per-provider publication serialization (pi chains promises; here one
        # FIFO lock per provider, created on first use).
        self._publication_locks: dict[str, sync.Lock] = {}

    # -- provider collection ---------------------------------------------------

    def set_provider(self, provider: Provider) -> None:
        self._supersede_provider_refresh(provider.id)
        with self._providers_guard:
            self._providers = {**self._providers, provider.id: provider}

    def delete_provider(self, id: str) -> None:
        self._supersede_provider_refresh(id)
        with self._providers_guard:
            providers = dict(self._providers)
            providers.pop(id, None)
            self._providers = providers

    def clear_providers(self) -> None:
        provider_ids = set(self._providers.keys())
        with self._refresh_state_guard:
            provider_ids |= set(self._refresh_controllers.keys())
        for provider_id in provider_ids:
            self._supersede_provider_refresh(provider_id)
        with self._providers_guard:
            self._providers = {}

    def get_providers(self) -> list[Provider]:
        return list(self._providers.values())

    def get_provider(self, id: str) -> Provider | None:
        return self._providers.get(id)

    def get_models(self, provider: str | None = None) -> list[Model]:
        """Sync read of last-known chat models. Best-effort: a provider whose
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
        """Sync runtime chat model lookup against last-known lists."""
        for model in self.get_models(provider):
            if model.id == id:
                return model
        return None

    def get_all_models(self, provider: str | None = None) -> list[AnyModel]:
        """Sync read of last-known models of every type from one provider or
        all providers. Best-effort, like `get_models()`."""
        if provider is not None:
            entry = self.get_provider(provider)
            if entry is None:
                return []
            try:
                return _provider_all_models(entry)
            except Exception:
                return []

        models: list[AnyModel] = []
        for entry in self.get_providers():
            try:
                models.extend(_provider_all_models(entry))
            except Exception:
                pass  # Best-effort: ill-behaved providers yield no models.
        return models

    def get_models_of_type(self, type: ModelType, provider: str | None = None) -> list[AnyModel]:
        """Sync read of last-known models of one type from one provider or all providers."""
        return [model for model in self.get_all_models(provider) if is_model_type(model, type)]

    def get_model_of_type(self, type: ModelType, provider: str, id: str) -> AnyModel | None:
        """Sync runtime lookup of a model of one type against last-known lists."""
        for model in self.get_models_of_type(type, provider):
            if model.id == id:
                return model
        return None

    # -- refresh ---------------------------------------------------------------

    def _supersede_provider_refresh(self, provider_id: str) -> int:
        with self._refresh_state_guard:
            generation = self._refresh_generations.get(provider_id, 0) + 1
            self._refresh_generations[provider_id] = generation
            previous = self._refresh_controllers.pop(provider_id, None)
        if previous is not None:
            previous.cancel()
        return generation

    def _begin_provider_refresh(self, provider_id: str) -> tuple[int, CancelToken]:
        generation = self._supersede_provider_refresh(provider_id)
        controller = CancelToken()
        with self._refresh_state_guard:
            self._refresh_controllers[provider_id] = controller
        return generation, controller

    async def _publish_provider_models(
        self,
        provider_id: str,
        generation: int,
        cancel: CancelToken,
        publication: ModelsPublication,
    ) -> bool:
        with self._publication_guard:
            lock = self._publication_locks.get(provider_id)
            if lock is None:
                lock = self._publication_locks[provider_id] = sync.Lock()

        async def _task() -> bool:
            async with lock:
                if cancel.cancelled or self._refresh_generations.get(provider_id) != generation:
                    return False

                if publication.persist is None:
                    await self._models_store.delete(provider_id, ModelsStoreOperationOptions(cancel=cancel))
                elif publication.persist is not ...:
                    await self._models_store.write(
                        provider_id, copy.deepcopy(publication.persist), ModelsStoreOperationOptions(cancel=cancel)
                    )

                if cancel.cancelled or self._refresh_generations.get(provider_id) != generation:
                    return False
                if publication.update is not None:
                    publication.update()
                return True

        return await race_with_cancel(_task(), cancel)

    async def _run_provider_refresh_phase(
        self,
        provider: Provider,
        credential: Credential | None,
        allow_network: bool,
        force: bool | None,
        generation: int,
        cancel: CancelToken,
    ) -> None:
        stored = await self._models_store.read(provider.id, ModelsStoreOperationOptions(cancel=cancel))

        async def publish(publication: ModelsPublication) -> bool:
            return await self._publish_provider_models(provider.id, generation, cancel, publication)

        await provider.refresh_models(
            RefreshModelsContext(
                credential=credential,
                stored=_with_known_model_types(copy.deepcopy(stored)) if stored is not None else None,
                publish=publish,
                allow_network=allow_network,
                force=force if allow_network else None,
                cancel=cancel,
            )
        )

    async def refresh(self, options: ModelsRefreshOptions | None = None) -> ModelsRefreshResult:
        """Refresh selected configured dynamic providers concurrently (all when
        `providers` is omitted). Provider errors and cancellation are returned
        without raising; static, unknown, and unconfigured providers are skipped."""
        options = options or ModelsRefreshOptions()
        allow_network = options.allow_network
        caller_cancel = operation_cancel(options.cancel)
        errors: dict[str, Exception] = {}
        if caller_cancel.cancelled:
            return ModelsRefreshResult(aborted=True, errors=errors)
        selected = set(options.providers) if options.providers is not None else None
        refreshable = [
            provider
            for provider in self.get_providers()
            if provider.has_dynamic_models and (selected is None or provider.id in selected)
        ]

        async def refresh_one(provider: Provider) -> None:
            generation, controller = self._begin_provider_refresh(provider.id)
            combined = combine_cancel_tokens(caller_cancel, controller)
            cancel = combined.token
            assert cancel is not None

            async def operation() -> None:
                stored_credential: Credential | None = None
                credential_error: BaseException | None = None
                try:
                    stored_credential = await self._read_credential(provider.id, cancel)
                except Exception as error:
                    credential_error = error

                # Restore cached provider state before auth resolution or network access.
                await self._run_provider_refresh_phase(provider, stored_credential, False, None, generation, cancel)
                if credential_error is not None:
                    raise credential_error
                if not allow_network or cancel.cancelled:
                    return

                credential = await self._resolve_refresh_credential(provider, stored_credential, cancel)
                if credential is None:
                    return
                await self._run_provider_refresh_phase(provider, credential, True, options.force, generation, cancel)

            try:
                await race_with_cancel(operation(), cancel)
            except Exception as error:
                if not cancel.cancelled:
                    errors[provider.id] = (
                        error
                        if isinstance(error, Exception)
                        else ModelsError("model_source", f"Model refresh failed for {provider.id}", cause=error)
                    )
            finally:
                with self._refresh_state_guard:
                    if self._refresh_controllers.get(provider.id) is controller:
                        del self._refresh_controllers[provider.id]
                combined.cleanup()

        if refreshable:

            async def refresh_all() -> None:
                await tonio.map(refresh_one, refreshable)

            try:
                await race_with_cancel(refresh_all(), caller_cancel)
            except Exception:
                if not caller_cancel.cancelled:
                    raise

        return ModelsRefreshResult(aborted=caller_cancel.cancelled, errors=dict(errors))

    async def _resolve_refresh_credential(
        self,
        provider: Provider,
        stored: Credential | None,
        cancel: CancelToken,
    ) -> Credential | None:
        if stored is not None and stored.type == "oauth":
            oauth = provider.auth.oauth
            if oauth is None:
                return None
            if clock.now_ms() < stored.expires:
                return stored
            if cancel.cancelled:
                return None
            # A refresh that has started survives cancellation or a superseding model refresh, so a
            # rotated refresh token is always persisted. A newer refresh then sees the fresh credential.
            return await refresh_stored_oauth_credential(
                self._credentials,
                provider.id,
                oauth,
                lambda current: clock.now_ms() >= current.expires,
                cancel,
            )

        api_key = provider.auth.api_key
        if api_key is None:
            return None
        credential = stored if stored is not None and stored.type == "api_key" else None
        result = await api_key.resolve(self._auth_context, credential, cancel)
        if result is None:
            return None
        return ApiKeyCredential(key=result.auth.api_key, env=result.env)

    # -- auth ------------------------------------------------------------------

    async def _read_credential(self, provider_id: str, cancel: CancelToken) -> Credential | None:
        try:
            return await self._credentials.read(provider_id, AuthOperationOptions(cancel=cancel))
        except Exception as error:
            raise ModelsError("auth", f"Credential store read failed for {provider_id}", cause=error)

    async def _check_provider_auth(
        self, provider: Provider, credential: Credential | None, cancel: CancelToken
    ) -> AuthCheck | None:
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
                    cancel,
                )
            except Exception as error:
                raise ModelsError("auth", f"API key auth check failed for provider {provider.id}", cause=error)

        resolution = await resolve_provider_auth(
            provider, self._credentials, self._auth_context, AuthResolutionOverrides(cancel=cancel)
        )
        return AuthCheck(source=resolution.source, type="api_key") if resolution is not None else None

    async def check_auth(self, provider_id: str, options: AuthOperationOptions | None = None) -> AuthCheck | None:
        """Check whether a provider has complete auth configuration without refreshing OAuth."""
        cancel = operation_cancel(options.cancel if options is not None else None)

        async def _check() -> AuthCheck | None:
            cancel.raise_if_cancelled()
            provider = self.get_provider(provider_id)
            if provider is None:
                return None
            return await self._check_provider_auth(provider, await self._read_credential(provider_id, cancel), cancel)

        return await race_with_cancel(_check(), cancel)

    async def _get_authenticated_providers(
        self, provider_id: str | None, cancel: CancelToken
    ) -> list[tuple[Provider, Credential | None]]:
        cancel.raise_if_cancelled()
        providers = (
            [entry for entry in [self.get_provider(provider_id)] if entry is not None]
            if provider_id
            else self.get_providers()
        )

        async def check_one(provider: Provider) -> tuple[Provider, Credential | None, AuthCheck | None]:
            credential = await self._read_credential(provider.id, cancel)
            return provider, credential, await self._check_provider_auth(provider, credential, cancel)

        if not providers:
            return []
        checks = await tonio.map(check_one, providers)
        return [(provider, credential) for provider, credential, auth in checks if auth is not None]

    async def get_available(
        self, provider_id: str | None = None, options: AuthOperationOptions | None = None
    ) -> list[Model]:
        """Return chat models whose providers have complete auth configuration."""
        cancel = operation_cancel(options.cancel if options is not None else None)

        async def _available() -> list[Model]:
            available: list[Model] = []
            for provider, credential in await self._get_authenticated_providers(provider_id, cancel):
                models = provider.get_models()
                available.extend(provider.filter_models(models, credential) if provider.filter_models else models)
            return available

        return await race_with_cancel(_available(), cancel)

    async def get_available_of_type(
        self, type: ModelType, provider_id: str | None = None, options: AuthOperationOptions | None = None
    ) -> list[AnyModel]:
        """Return models of one type whose providers have complete auth configuration."""
        return [model for model in await self.get_all_available(provider_id, options) if is_model_type(model, type)]

    async def get_all_available(
        self, provider_id: str | None = None, options: AuthOperationOptions | None = None
    ) -> list[AnyModel]:
        """Return models of every type whose providers have complete auth configuration."""
        cancel = operation_cancel(options.cancel if options is not None else None)

        async def _available() -> list[AnyModel]:
            available: list[AnyModel] = []
            for provider, credential in await self._get_authenticated_providers(provider_id, cancel):
                models = _provider_all_models(provider)
                if provider.filter_all_models is not None:
                    available.extend(provider.filter_all_models(models, credential))
                elif provider.filter_models is None:
                    available.extend(models)
                else:
                    available_chat_ids = {
                        model.id for model in provider.filter_models(provider.get_models(), credential)
                    }
                    available.extend(
                        model for model in models if not is_model_type(model, "chat") or model.id in available_chat_ids
                    )
            return available

        return await race_with_cancel(_available(), cancel)

    async def get_auth(
        self,
        provider_or_model: str | AnyModel,
        overrides: AuthResolutionOverrides | None = None,
    ) -> AuthResult | None:
        """Resolve provider-scoped auth by provider id, or provider auth plus
        static model headers when passed a model."""
        provider_id = provider_or_model if isinstance(provider_or_model, str) else provider_or_model.provider
        provider = self.get_provider(provider_id)
        if provider is None:
            return None
        return await self.get_auth_for_provider(provider, provider_or_model, overrides)

    async def get_auth_for_provider(
        self,
        provider: Provider,
        provider_or_model: str | AnyModel,
        overrides: AuthResolutionOverrides | None = None,
    ) -> AuthResult | None:
        """pidrei-only epoch variant of `get_auth`: resolve auth against an
        already-pinned provider object, so a caller that resolved the provider
        for a request cannot get auth from a newer composition than the
        provider it will stream with (config epochs, spec/concurrency.md)."""
        cancel = operation_cancel(overrides.cancel if overrides is not None else None)
        merged_overrides = (
            replace(overrides, cancel=cancel) if overrides is not None else AuthResolutionOverrides(cancel=cancel)
        )
        result = await resolve_provider_auth(provider, self._credentials, self._auth_context, merged_overrides)
        if result is None or isinstance(provider_or_model, str) or not provider_or_model.headers:
            return result
        return replace(
            result,
            auth=replace(result.auth, headers=merge_headers(result.auth.headers, provider_or_model.headers)),
        )

    async def login(
        self, provider_id: str, type: AuthType, interaction: AuthInteraction, options: LoginOptions | None = None
    ) -> Credential:
        """Run a provider-owned login flow and persist its returned credential.

        A cancellation raised before the store mutation begins rejects with the
        abort reason; once the mutation's callback has started, the write is
        awaited to completion so the stored credential stays locally consistent.

        `options` reaches OAuth logins only: pi passes it to both methods, but
        `ApiKeyAuth.login`'s type takes the interaction alone.
        """
        cancel = operation_cancel(interaction.cancel)
        cancel.raise_if_cancelled()
        provider = self.get_provider(provider_id)
        if provider is None:
            raise ModelsError("provider", f"Unknown provider: {provider_id}")
        method = provider.auth.oauth if type == "oauth" else provider.auth.api_key
        login = getattr(method, "login", None) if method is not None else None
        if login is None:
            raise ModelsError("auth", f"{provider.name} does not support {type} login")
        normalized = _NormalizedAuthInteraction(interaction, cancel)
        login_operation = login(normalized, options) if type == "oauth" else login(normalized)
        credential = await race_with_cancel(login_operation, cancel)

        # The persist is detached on purpose (pi lets a started write finish
        # even when the caller aborts); the wait below resumes on started,
        # done or abort, whichever comes first.
        mutation_started = tonio.Event()
        mutation_done = tonio.Event()
        mutation_box = tonio.Result()

        async def _persist(_current: Credential | None) -> Credential | None:
            mutation_started.set()
            return credential

        async def _mutation() -> None:
            try:
                mutation_box.store(
                    (
                        "value",
                        await self._credentials.modify(provider_id, _persist, AuthOperationOptions(cancel=cancel)),
                    )
                )
            except Exception as error:
                mutation_box.store(("error", error))
            finally:
                mutation_done.set()

        tonio.spawn.without_tracking(_mutation())

        try:
            await tonio.Waiter.any(mutation_started, mutation_done, cancel.event)
            if cancel.cancelled and not mutation_started.is_set() and not mutation_done.is_set():
                raise cancel.reason  # type: ignore[misc]
            await mutation_done.wait()
            kind, payload = mutation_box.fetch()
            if kind == "error":
                raise payload
        except Exception as error:
            cancel.raise_if_cancelled()
            raise ModelsError("auth", f"Credential store modify failed for {provider_id}", cause=error)
        return credential

    async def logout(self, provider_id: str, options: AuthOperationOptions | None = None) -> None:
        cancel = operation_cancel(options.cancel if options is not None else None)
        cancel.raise_if_cancelled()
        try:
            await self._credentials.delete(provider_id, AuthOperationOptions(cancel=cancel))
        except Exception as error:
            cancel.raise_if_cancelled()
            raise ModelsError("auth", f"Credential store delete failed for {provider_id}", cause=error)

    # -- streaming -------------------------------------------------------------

    def _require_provider(self, model: AnyModel) -> Provider:
        provider = self.get_provider(model.provider)
        if provider is None:
            raise ModelsError("provider", f"Unknown provider: {model.provider}")
        return provider

    def _require_chat_provider(self, model: Model) -> Provider:
        assert_chat_model(model)
        return self._require_provider(model)

    async def _apply_auth[TModel: Model | ImageModel | ClassifierModel, TOptions: ProviderRequestOptions](
        self,
        provider: Provider,
        model: TModel,
        options: TOptions,
    ) -> tuple[TModel, TOptions]:
        # Epoch discipline: auth resolves against the provider the caller
        # pinned for this request, never a fresh (possibly newer) lookup.
        resolution = await self.get_auth_for_provider(
            provider,
            model,
            AuthResolutionOverrides(
                api_key=options.api_key if options is not None else None,
                env=options.env if options is not None else None,
                cancel=options.cancel if options is not None else None,
            ),
        )
        if resolution is None:
            raise ModelsError("auth", f"Provider is not configured: {model.provider}")
        auth = resolution.auth

        # Explicit request options win per-field; the Models-only transform runs last.
        api_key = options.api_key if options is not None and options.api_key is not None else auth.api_key
        headers = merge_headers(auth.headers, options.headers if options is not None else None)
        if options is not None and options.transform_headers is not None:
            headers = await options.transform_headers(headers if headers is not None else {})
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
        context: Context | TranscriptContext,
        options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        transcript = normalize_context(context)

        async def _setup(stream: AssistantMessageEventStream):
            provider = self._require_chat_provider(model)
            request_model, request_options = await self._apply_auth(
                provider, model, options if options is not None else StreamOptions()
            )
            return call_stream_into(provider.stream, request_model, transcript, request_options, into=stream)

        return lazy_stream(model, _setup, _cancel_of(options))

    async def complete(self, model: Model, context: Context, options: StreamOptions | None = None):
        return await self.stream(model, context, options).result()

    def stream_simple(
        self,
        model: Model,
        context: Context | TranscriptContext,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        transcript = normalize_context(context)

        async def _setup(stream: AssistantMessageEventStream):
            provider = self._require_chat_provider(model)
            request_model, request_options = await self._apply_auth(
                provider, model, options if options is not None else SimpleStreamOptions()
            )
            return call_stream_into(provider.stream_simple, request_model, transcript, request_options, into=stream)

        return lazy_stream(model, _setup, _cancel_of(options))

    async def complete_simple(self, model: Model, context: Context, options: SimpleStreamOptions | None = None):
        return await self.stream_simple(model, context, options).result()

    def stream_deferred(
        self,
        model: Model,
        handle: DeferredHandle,
        options: DeferredFetchOptions | None = None,
    ) -> AssistantMessageEventStream:
        async def _setup(_stream: AssistantMessageEventStream):
            provider = self._require_chat_provider(model)
            if not provider.supports_fetch_deferred:
                raise ModelsError("provider", f"Provider {model.provider} does not support deferred responses")
            request_model, request_options = await self._apply_auth(
                provider, model, options if options is not None else DeferredFetchOptions()
            )
            return provider.fetch_deferred(request_model, handle, request_options)

        return lazy_stream(model, _setup, _cancel_of(options))

    async def fetch_deferred(
        self,
        model: Model,
        handle: DeferredHandle,
        options: DeferredFetchOptions | None = None,
    ) -> AssistantMessage:
        return await self.stream_deferred(model, handle, options).result()

    async def cancel_deferred(
        self,
        model: Model,
        handle: DeferredHandle,
        options: DeferredCancelOptions | None = None,
    ) -> None:
        provider = self._require_chat_provider(model)
        if not provider.supports_cancel_deferred:
            raise ModelsError("provider", f"Provider {model.provider} does not support deferred responses")
        request_model, request_options = await self._apply_auth(
            provider, model, options if options is not None else ProviderRequestOptions()
        )
        await provider.cancel_deferred(request_model, handle, request_options)

    # -- one-shot operations ---------------------------------------------------

    async def generate_images(
        self, model: ImageModel, context: ImagesContext, options: ImagesOptions | None = None
    ) -> AssistantImages:
        """Generate images through the owning provider with auth resolved like
        `stream()`. Never raises: unknown providers, unconfigured auth, and
        providers without `generate_images` return an error `AssistantImages`."""
        try:
            assert_image_model(model)
            provider = self._require_provider(model)
            if provider.generate_images is None:
                raise ModelsError("provider", f"Provider {model.provider} does not support image generation")
            request_model, request_options = await self._apply_auth(
                provider, model, options if options is not None else ImagesOptions()
            )
            return await provider.generate_images(request_model, context, request_options)
        except Exception as error:
            return image_error_result(model, error, _cancelled(options))

    async def classify(
        self, model: ClassifierModel, context: ClassifierContext, options: ClassifierOptions | None = None
    ) -> ClassifierResult:
        """Classify structured state through the owning provider. Never raises."""
        try:
            assert_classifier_model(model)
            provider = self._require_provider(model)
            if provider.classify is None:
                raise ModelsError("provider", f"Provider {model.provider} does not support classification")
            request_model, request_options = await self._apply_auth(
                provider, model, options if options is not None else ClassifierOptions()
            )
            return await provider.classify(request_model, context, request_options)
        except Exception as error:
            return classifier_error_result(model, error, _cancelled(options))


def _cancelled(options: ProviderRequestOptions | None) -> bool:
    """pi: `options?.signal?.aborted`."""
    return options is not None and options.cancel is not None and options.cancel.cancelled


def create_models(
    *,
    credentials: CredentialStore | None = None,
    models_store: ModelsStore | None = None,
    auth_context: AuthContext | None = None,
) -> Models:
    return Models(credentials=credentials, models_store=models_store, auth_context=auth_context)


# -- model helpers -------------------------------------------------------------


def has_api(model: AnyModel, api: str) -> bool:
    """Runtime narrowing check for dynamically looked-up models. Non-chat
    models never match, even when their api id equals `api`."""
    return is_model_type(model, "chat") and model.api == api


def calculate_cost(model: AnyModel, usage: UsageBuilder) -> UsageCostBuilder:
    """Compute request cost into `usage.cost` (mutates and returns it).

    Producer-side only: `usage` is a builder — the frozen `Usage` on a
    published message is a value and never passes through here."""
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


def models_are_equal(a: AnyModel | None, b: AnyModel | None) -> bool:
    """Check if two models are equal by comparing their type, id, and provider."""
    if a is None or b is None:
        return False
    return get_model_type(a) == get_model_type(b) and a.id == b.id and a.provider == b.provider
