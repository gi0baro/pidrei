"""Port of pi's models registry (packages/ai/src/models.ts).

`Models` is the runtime collection of providers plus auth application and
stream convenience; providers own stream behavior, `Models` resolves auth and
delegates each request to the provider that owns the model.
"""

import copy
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from types import EllipsisType
from typing import Any

import tonio.colored as tonio

from pidrei_ai.api.lazy import _cancel_of, call_stream_into, lazy_stream
from pidrei_ai.auth.context import default_provider_auth_context
from pidrei_ai.auth.credential_store import InMemoryCredentialStore
from pidrei_ai.auth.resolve import AuthResolutionOverrides, ModelsError, resolve_provider_auth
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
    ProviderAuth,
)
from pidrei_ai.models_store import InMemoryModelsStore, ModelsStore, ModelsStoreEntry, ModelsStoreOperationOptions
from pidrei_ai.types import (
    AssistantMessage,
    Context,
    DeferredCancelOptions,
    DeferredFetchOptions,
    DeferredHandle,
    Model,
    ModelThinkingLevel,
    ProviderHeaders,
    ProviderRequestOptions,
    SimpleStreamOptions,
    StreamOptions,
    Usage,
    UsageCost,
)
from pidrei_ai.utils.abort import operation_cancel, race_with_cancel
from pidrei_ai.utils.cancel import CancelToken, combine_cancel_tokens
from pidrei_ai.utils.event_stream import AssistantMessageEventStream
from pidrei_ai.utils.headers import merge_headers


def _now_ms() -> int:
    return int(time.time() * 1000)


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
        """Restore `context.stored` and optionally fetch a newer list using the
        effective credential, publishing persistence and synchronous state
        changes through the generation-checked `context.publish()`."""
        if self._fetch_models is None:
            return

        if context.stored is not None:
            restored = [model for model in context.stored.models if model.provider == self.id]

            def _apply_restored(restored: list[Model] = restored) -> None:
                self._dynamic_models = restored

            if not await context.publish(ModelsPublication(update=_apply_restored)):
                return
        if not context.allow_network or context.cancel.cancelled:
            return
        refreshed = await self._fetch_models(context)
        if context.cancel.cancelled:
            return

        def _apply_refreshed() -> None:
            self._dynamic_models = list(refreshed)

        await context.publish(
            ModelsPublication(
                persist=ModelsStoreEntry(models=list(refreshed), checked_at=_now_ms()),
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
        context: Context,
        options: StreamOptions | None = None,
        *,
        into: AssistantMessageEventStream | None = None,
    ):
        return self._dispatch(model, "stream", (model, context), options, into)

    def stream_simple(
        self,
        model: Model,
        context: Context,
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


class _NormalizedAuthInteraction:
    """pi: `{ ...interaction, signal }` — the same interaction surface with a
    guaranteed cancel token (`ProviderAuthInteraction`)."""

    __slots__ = ("_base", "cancel")

    def __init__(self, base: AuthInteraction, cancel: CancelToken):
        self._base = base
        self.cancel = cancel

    async def prompt(self, prompt: AuthPrompt) -> str:
        return await self._base.prompt(prompt)

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
        self._providers: dict[str, Provider] = {}
        self._providers_guard = threading.Lock()
        self._credentials = credentials if credentials is not None else InMemoryCredentialStore()
        self._models_store = models_store if models_store is not None else InMemoryModelsStore()
        self._auth_context = auth_context if auth_context is not None else default_provider_auth_context()
        self._refresh_state_guard = threading.Lock()
        self._refresh_generations: dict[str, int] = {}
        self._refresh_controllers: dict[str, CancelToken] = {}
        self._publication_guard = threading.Lock()
        # Per-provider publication chains (pi chains promises; here an Event
        # marks each link settled).
        self._publication_chains: dict[str, tuple[Any, ...]] = {}

    # -- provider collection ---------------------------------------------------

    def set_provider(self, provider: Provider) -> None:
        self._supersede_provider_refresh(provider.id)
        with self._providers_guard:
            self._providers[provider.id] = provider

    def delete_provider(self, id: str) -> None:
        self._supersede_provider_refresh(id)
        with self._providers_guard:
            self._providers.pop(id, None)

    def clear_providers(self) -> None:
        with self._providers_guard:
            provider_ids = set(self._providers.keys())
        with self._refresh_state_guard:
            provider_ids |= set(self._refresh_controllers.keys())
        for provider_id in provider_ids:
            self._supersede_provider_refresh(provider_id)
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
            previous = self._publication_chains.get(provider_id)
            done = tonio.Event()
            entry = (done,)
            self._publication_chains[provider_id] = entry

        async def _task() -> bool:
            if previous is not None:
                await previous[0].wait()
            try:
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
            finally:
                done.set()
                with self._publication_guard:
                    if self._publication_chains.get(provider_id) is entry:
                        del self._publication_chains[provider_id]

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
                stored=copy.deepcopy(stored) if stored is not None else None,
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
            if _now_ms() < stored.expires:
                return stored
            if cancel.cancelled:
                return None

            async def _refresh(current: Credential | None) -> Credential | None:
                if current is None or current.type != "oauth" or _now_ms() < current.expires:
                    return None
                return await oauth.refresh(current, cancel)

            post = await self._credentials.modify(provider.id, _refresh, AuthOperationOptions(cancel=cancel))
            return post if post is not None and post.type == "oauth" else None

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

    async def get_available(
        self, provider_id: str | None = None, options: AuthOperationOptions | None = None
    ) -> list[Model]:
        """Return models whose providers have complete auth configuration."""
        cancel = operation_cancel(options.cancel if options is not None else None)

        async def _available() -> list[Model]:
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

            available: list[Model] = []
            for provider, credential, auth in checks:
                if auth is None:
                    continue
                models = provider.get_models()
                available.extend(provider.filter_models(models, credential) if provider.filter_models else models)
            return available

        return await race_with_cancel(_available(), cancel)

    async def get_auth(
        self,
        provider_or_model: str | Model,
        overrides: AuthResolutionOverrides | None = None,
    ) -> AuthResult | None:
        """Resolve provider-scoped auth by provider id, or provider auth plus
        static model headers when passed a model."""
        cancel = operation_cancel(overrides.cancel if overrides is not None else None)
        provider_id = provider_or_model if isinstance(provider_or_model, str) else provider_or_model.provider
        provider = self.get_provider(provider_id)
        if provider is None:
            return None
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

    async def login(self, provider_id: str, type: AuthType, interaction: AuthInteraction) -> Credential:
        """Run a provider-owned login flow and persist its returned credential.

        A cancellation raised before the store mutation begins rejects with the
        abort reason; once the mutation's callback has started, the write is
        awaited to completion so the stored credential stays locally consistent.
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
        credential = await race_with_cancel(login(_NormalizedAuthInteraction(interaction, cancel)), cancel)

        mutation_started = tonio.Event()
        mutation_done = tonio.Event()
        mutation_box: list[tuple[str, Any]] = []

        async def _persist(_current: Credential | None) -> Credential | None:
            mutation_started.set()
            return credential

        async def _mutation() -> None:
            try:
                mutation_box.append(
                    (
                        "value",
                        await self._credentials.modify(provider_id, _persist, AuthOperationOptions(cancel=cancel)),
                    )
                )
            except BaseException as error:
                mutation_box.append(("error", error))
            finally:
                mutation_done.set()

        tonio.spawn.without_tracking(_mutation())

        async def _wait_started() -> str:
            await mutation_started.wait()
            return "started"

        async def _wait_done() -> str:
            await mutation_done.wait()
            return "done"

        async def _wait_aborted() -> str:
            await cancel.wait()
            return "aborted"

        try:
            winner = await tonio.select(_wait_started(), _wait_done(), _wait_aborted())
            if winner == "aborted" and not mutation_started.is_set() and not mutation_done.is_set():
                raise cancel.reason  # type: ignore[misc]
            await mutation_done.wait()
            kind, payload = mutation_box[0]
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
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        async def _setup(stream: AssistantMessageEventStream):
            provider = self._require_provider(model)
            request_model, request_options = await self._apply_auth(
                model, options if options is not None else StreamOptions()
            )
            return call_stream_into(provider.stream, request_model, context, request_options, into=stream)

        return lazy_stream(model, _setup, _cancel_of(options))

    async def complete(self, model: Model, context: Context, options: StreamOptions | None = None):
        return await self.stream(model, context, options).result()

    def stream_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        async def _setup(stream: AssistantMessageEventStream):
            provider = self._require_provider(model)
            request_model, request_options = await self._apply_auth(
                model, options if options is not None else SimpleStreamOptions()
            )
            return call_stream_into(provider.stream_simple, request_model, context, request_options, into=stream)

        return lazy_stream(model, _setup, _cancel_of(options))

    async def complete_simple(self, model: Model, context: Context, options: SimpleStreamOptions | None = None):
        return await self.stream_simple(model, context, options).result()

    async def fetch_deferred(
        self,
        model: Model,
        handle: DeferredHandle,
        options: DeferredFetchOptions | None = None,
    ) -> AssistantMessage:
        async def _setup(_stream: AssistantMessageEventStream):
            provider = self._require_provider(model)
            if not provider.supports_fetch_deferred:
                raise ModelsError("provider", f"Provider {model.provider} does not support deferred responses")
            request_model, request_options = await self._apply_auth(
                model, options if options is not None else DeferredFetchOptions()
            )
            return provider.fetch_deferred(request_model, handle, request_options)

        return await lazy_stream(model, _setup, _cancel_of(options)).result()

    async def cancel_deferred(
        self,
        model: Model,
        handle: DeferredHandle,
        options: DeferredCancelOptions | None = None,
    ) -> None:
        provider = self._require_provider(model)
        if not provider.supports_cancel_deferred:
            raise ModelsError("provider", f"Provider {model.provider} does not support deferred responses")
        request_model, request_options = await self._apply_auth(
            model, options if options is not None else ProviderRequestOptions()
        )
        await provider.cancel_deferred(request_model, handle, request_options)


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
