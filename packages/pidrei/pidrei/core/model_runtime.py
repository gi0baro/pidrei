"""Mirror of pi coding-agent src/core/model-runtime.ts.

Configured pidrei-ai Models collection used by the coding agent and SDK
consumers.

Deviations from pi (documented):
- radius gateway providers are not ported at all. pi lets models.json declare
  `oauth: "radius"` to swap in a radius builtin; the provider was a documented
  drop, and Phase 6 removed the last remnants of the pathway, so the key is now
  an unrecognised entry that models.json ignores and such configs compose as
  plain api-key providers.
- `PI_OFFLINE` → `PIDREI_OFFLINE`.
"""

import os
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any

import tonio.colored as tonio
from tonio.colored import sync

from pidrei_ai.api.lazy import _cancel_of, call_stream_into, lazy_stream
from pidrei_ai.auth.resolve import AuthResolutionOverrides, ModelsError
from pidrei_ai.auth.types import (
    ApiKeyCredential,
    AuthCheck,
    AuthInteraction,
    AuthOperationOptions,
    AuthResult,
    AuthType,
    Credential,
    CredentialInfo,
    CredentialStore,
)
from pidrei_ai.models_store import ModelsStore
from pidrei_ai.providers.all import builtin_providers, get_builtin_model_data_generated_at
from pidrei_ai.registry import ModelsRefreshOptions, ModelsRefreshResult, Provider, create_models
from pidrei_ai.types import Context, DeferredHandle, Model, SimpleStreamOptions, StreamOptions
from pidrei_ai.utils.cancel import CancelToken, combine_cancel_tokens
from pidrei_ai.utils.headers import merge_headers

from ..config import get_agent_dir
from ..utils.abort import operation_cancel
from .auth_storage import AuthStorage
from .model_config import ModelConfig
from .models_store import FileModelsStore, InMemoryCodingAgentModelsStore
from .provider_composer import (
    AuthStatus,
    CompatibilityRequestConfig,
    ProviderConfigInput,
    compose_model_provider,
    configured_request_auth_status,
    resolve_compatibility_request_config,
    resolve_configured_model_headers,
    validate_extension_provider,
)
from .remote_catalog import with_remote_catalog
from .runtime_credentials import RuntimeCredentials


@dataclass(slots=True)
class _Snapshot:
    all: list[Model] = field(default_factory=list)
    available: list[Model] = field(default_factory=list)
    configured_providers: set[str] = field(default_factory=set)
    stored_providers: set[str] = field(default_factory=set)
    auth: dict[str, AuthCheck | None] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class _CompositionEpoch:
    """One immutable snapshot of the provider-composition inputs, published
    by `_publish_composition` under `_composition_guard` and pinned by
    readers with a single attribute read (PROPER_MT_DESIGN.md step 3, the
    §5 pattern `_Snapshot` established for availability). The working dicts
    stay writer-private under the guard; the epoch's dicts are copies that
    are never mutated after publication, so an operation that pins the epoch
    can never see config from one composition pass and extension providers
    or errors from another — and never trips over a concurrently-mutated
    dict while iterating."""

    config: ModelConfig
    extension_providers: dict[str, ProviderConfigInput]
    native_extension_providers: dict[str, Provider]
    composition_errors: dict[str, str]


def _unwrap_spawn_error(error: Exception) -> Exception:
    """tonio surfaces child failures as ExceptionGroup('SpawnExceptionGroup');
    pi's contract exposes the underlying error — unwrap single-child groups."""
    while (
        isinstance(error, ExceptionGroup) and len(error.exceptions) == 1 and isinstance(error.exceptions[0], Exception)
    ):
        error = error.exceptions[0]
    return error


@dataclass(slots=True, kw_only=True)
class ModelRuntimeAuthOverrides:
    api_key: str | None = None
    env: dict[str, str] | None = None
    #: Require this much remaining OAuth-token validity; defaults to five minutes.
    min_oauth_validity_ms: int | None = None
    cancel: CancelToken | None = None


type CredentialSynchronizationOperation = str  # "login" | "logout" | "setRuntimeApiKey" | "removeRuntimeApiKey"


class _CancelBoundInteraction:
    """pi: `{ ...interaction, signal }` — the interaction surface with the
    operation's cancel token bound."""

    __slots__ = ("_base", "cancel")

    def __init__(self, base: AuthInteraction, cancel: CancelToken):
        self._base = base
        self.cancel = cancel

    async def prompt(self, prompt: Any) -> str:
        return await self._base.prompt(prompt)

    def notify(self, event: Any) -> None:
        self._base.notify(event)


class CredentialSynchronizationError(Exception):
    """Credentials changed successfully, but the local model/auth snapshot could
    not be synchronized."""

    def __init__(
        self,
        provider_id: str,
        operation: CredentialSynchronizationOperation,
        credential: Credential | None,
        *,
        cause: BaseException | None = None,
    ):
        super().__init__(f"Credential {operation} committed for {provider_id}, but local synchronization failed")
        self.provider_id = provider_id
        self.operation = operation
        self.credential = credential
        if cause is not None:
            self.__cause__ = cause


class ModelRuntime:
    def __init__(
        self,
        credentials: RuntimeCredentials,
        config: ModelConfig,
        models_path: str | None,
        models_store: ModelsStore,
        providers: list[Provider],
        model_network_enabled: bool,
    ):
        self._credentials = credentials
        self._config = config
        self._models_path = models_path
        self._model_network_enabled = model_network_enabled
        self._default_builtins: dict[str, Provider] = {provider.id: provider for provider in providers}
        self._builtins: dict[str, Provider] = dict(self._default_builtins)
        self._native_extension_providers: dict[str, Provider] = {}
        self._extension_providers: dict[str, ProviderConfigInput] = {}
        self._composition_errors: dict[str, str] = {}
        self._snapshot = _Snapshot()
        self._availability_refresh_seq = 0
        self._availability_error_seq = 0
        # One credential-driven availability pass per provider at a time (see
        # `_refresh_provider_availability`).
        self._provider_availability_locks: dict[str, sync.Lock] = {}
        # Providers with a credential-driven availability pass in flight, by
        # count (see `_refresh_provider_availability`). pi supersedes those from
        # a newer full pass by bumping their seq; that is safe on one loop,
        # where the full pass can only have started after the credential write
        # it is racing. Here the two run on different threads, so the full pass
        # can be carrying pre-credential data and still be the "newer" one by
        # seq — it would then drop the login's publish and republish the state
        # from before the login. The full pass leaves these providers alone
        # instead, and the pass that owns them publishes its own merge.
        self._provider_availability_inflight: dict[str, int] = {}
        # Guards every mutation of the availability state above (the seq
        # counters, the inflight map, the error slot) and every
        # read-modify-write of `_snapshot`. pi's seq guards are check-then-act
        # sections with no await between check and publish, atomic on one
        # loop; here passes run on different threads, so an unguarded pass can
        # pass its seq check, lose the CPU while a credential-driven pass
        # bumps the seq and publishes, then clobber that publish with
        # pre-credential data. Held only across sync sections, never an await.
        self._availability_guard = threading.Lock()
        self._availability_error: str | None = None
        # Per-provider serialized credential operations (pi chains promises;
        # here one FIFO lock per provider, created on first use).
        self._credential_operations: dict[str, sync.Lock] = {}
        self._credential_operations_guard = threading.Lock()
        # pi composes providers on one event loop, so a mutation and a rebuild
        # can never interleave. Here a detached refresh rebuilds on another
        # thread while a caller registers or unregisters on this one, and the
        # window inside _recompose_provider (compose, *then* publish) is wide
        # enough to lose an unregister. Every mutation of the composition
        # inputs and every publish of a composed provider holds this.
        self._composition_guard = threading.RLock()
        # Serializes whole `refresh()` runs. pi's `void this.refresh(...)` from
        # registerProvider overlaps the caller's awaited refresh only at await
        # points on one thread; here two refreshes genuinely run in parallel,
        # and one's `_rebuild_providers()` replaces the composed provider
        # objects the other is mid-way through populating (the legacy-OAuth
        # credential landed on stale objects; seen as a macOS-CI failure).
        self._refresh_serial = sync.Lock()
        # Serialization alone is not enough: a spawned trigger that acquires
        # the lock *after* an awaited refresh returned would rebuild the
        # providers again and readers of the live registry (`get_model`) see
        # the de-projected window. pi never hits this because its microtask
        # FIFO finishes the fire-and-forget refresh before the awaited one
        # resolves. Triggers therefore only *request* a refresh; any run
        # clears the flag at its start, satisfying every request made before
        # it — so no run happens after an awaited refresh unless state
        # actually changed since that refresh began.
        self._refresh_requested = False
        # Published composition epoch; see _CompositionEpoch. Seeded here so
        # readers never observe a missing epoch, republished by every
        # composition mutation.
        self._composition = _CompositionEpoch(
            config=config,
            extension_providers={},
            native_extension_providers={},
            composition_errors={},
        )
        self._models = create_models(credentials=credentials, models_store=models_store)
        self._rebuild_providers()

    @staticmethod
    async def create(
        *,
        credentials: CredentialStore | None = None,
        auth_path: str | None = None,
        models_path: object = ...,  # str path; None disables models.json entirely; default ~/.pidrei/agent/models.json
        models_store: ModelsStore | None = None,
        models_store_path: str | None = None,
        allow_model_network: bool = False,
        model_refresh_timeout_ms: float | None = None,
        catalog_base_url: str | None = None,
        cancel: CancelToken | None = None,
        refresh_on_create: bool = True,
    ) -> ModelRuntime:
        runtime_credentials = RuntimeCredentials(
            credentials if credentials is not None else await AuthStorage.create(auth_path)
        )
        resolved_models_path: str | None
        if models_path is None:
            resolved_models_path = None
        elif models_path is ...:
            resolved_models_path = os.path.join(get_agent_dir(), "models.json")
        else:
            resolved_models_path = models_path
        config = await ModelConfig.load(resolved_models_path)
        if models_store is None:
            if resolved_models_path:
                store_path = (
                    models_store_path
                    if models_store_path is not None
                    else os.path.join(os.path.dirname(resolved_models_path), "models-store.json")
                )
                models_store = FileModelsStore(store_path)
            else:
                models_store = InMemoryCodingAgentModelsStore()
        builtin_model_data_generated_at = await get_builtin_model_data_generated_at()
        providers = [
            with_remote_catalog(provider, catalog_base_url, builtin_model_data_generated_at)
            for provider in builtin_providers()
        ]
        runtime = ModelRuntime(
            runtime_credentials,
            config,
            resolved_models_path,
            models_store,
            providers,
            os.environ.get("PIDREI_OFFLINE") is None,
        )
        runtime._rebuild_providers()
        refresh_from_network = runtime._model_network_enabled and allow_model_network is True
        controller = CancelToken() if refresh_from_network and model_refresh_timeout_ms is not None else None
        settled = tonio.Event()
        if controller is not None:
            timeout_s = model_refresh_timeout_ms / 1000

            async def watchdog() -> None:
                await settled.wait(timeout_s)
                if not settled.is_set():
                    controller.cancel()

            tonio.spawn.without_tracking(watchdog())
        combined = combine_cancel_tokens(cancel, controller)
        try:
            if refresh_on_create:
                await runtime.refresh(ModelsRefreshOptions(allow_network=refresh_from_network, cancel=combined.token))
        finally:
            combined.cleanup()
            settled.set()
        return runtime

    # -- provider composition --------------------------------------------------

    def _provider_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for source in (
            self._builtins.keys(),
            self._native_extension_providers.keys(),
            self._config.get_provider_ids(),
            self._extension_providers.keys(),
        ):
            for provider_id in source:
                seen[provider_id] = None
        return list(seen.keys())

    def _publish_composition(self) -> None:
        """Publish an immutable epoch of the composition inputs. Callers hold
        `_composition_guard`; the copies taken here are never mutated after
        the rebind, so readers pin `self._composition` lock-free."""
        self._composition = _CompositionEpoch(
            config=self._config,
            extension_providers=dict(self._extension_providers),
            native_extension_providers=dict(self._native_extension_providers),
            composition_errors=dict(self._composition_errors),
        )

    def _recompose_provider(self, provider_id: str) -> None:
        with self._composition_guard:
            try:
                base = self._native_extension_providers.get(provider_id) or self._builtins.get(provider_id)
                extension = self._extension_providers.get(provider_id)
                if base is None and self._config.get_provider(provider_id) is None and extension is None:
                    self._models.delete_provider(provider_id)
                    self._composition_errors.pop(provider_id, None)
                    return
                if base is not None and self._config.get_provider(provider_id) is None and extension is None:
                    # No overlays: use the builtin untouched so its auth/login/stream behavior is exact.
                    self._models.set_provider(base)
                    self._composition_errors.pop(provider_id, None)
                    return
                try:
                    self._models.set_provider(compose_model_provider(provider_id, base, self._config, extension))
                    self._composition_errors.pop(provider_id, None)
                except Exception as error:
                    self._composition_errors[provider_id] = str(error)
                    if base is not None:
                        self._models.set_provider(base)
                    else:
                        self._models.delete_provider(provider_id)
            finally:
                self._publish_composition()

    def _rebuild_providers(self) -> None:
        # pi clears the collection and recomposes; its event loop makes that
        # atomic. Here mutations run truly in parallel with readers (detached
        # refresh tasks), so rebuild in place: recompose every desired
        # provider, then drop only the removed ones — no empty window.
        with self._composition_guard:
            self._composition_errors.clear()
            desired = set(self._provider_ids())
            for provider_id in desired:
                self._recompose_provider(provider_id)
            for provider in self._models.get_providers():
                if provider.id not in desired:
                    self._models.delete_provider(provider.id)
            # Covers the empty-desired edge (no _recompose_provider call
            # published the cleared error map).
            self._publish_composition()
            self._update_model_snapshot()

    def _update_model_snapshot(self) -> None:
        all_models = list(self._models.get_models())
        with self._availability_guard:
            self._snapshot = replace(
                self._snapshot,
                all=all_models,
                available=[model for model in all_models if model.provider in self._snapshot.configured_providers],
            )

    # -- availability ----------------------------------------------------------

    async def _run_availability_refresh(self, seq: int, error_seq: int, cancel: CancelToken) -> None:
        providers = self._models.get_providers()
        options = AuthOperationOptions(cancel=cancel)

        async def check_one(provider: Provider) -> tuple[str, AuthCheck | None]:
            return provider.id, await self._models.check_auth(provider.id, options)

        async def check_all() -> list[tuple[str, AuthCheck | None]]:
            if not providers:
                return []
            return await tonio.map(check_one, providers)

        available, checks, credentials = await tonio.spawn(
            self._models.get_available(None, options), check_all(), self._credentials.list(options)
        )
        all_models = list(self._models.get_models())
        with self._availability_guard:
            # A newer rebuild was requested while this one was in flight; drop
            # this result so a slow, superseded refresh cannot clobber the
            # snapshot with stale data.
            if seq != self._availability_refresh_seq:
                return
            auth = dict(checks)
            configured_providers = {provider_id for provider_id, check in checks if check is not None}
            stored_providers = {entry.provider_id for entry in credentials}
            available_models = list(available)
            inflight = frozenset(self._provider_availability_inflight)
            if inflight:
                # These providers belong to a credential-driven pass that has
                # not published yet; carry their current state through
                # untouched rather than overwrite it with this pass's reading
                # (see the note on `_provider_availability_inflight`).
                previous = self._snapshot
                for provider_id in inflight:
                    auth.pop(provider_id, None)
                    configured_providers.discard(provider_id)
                    stored_providers.discard(provider_id)
                    if provider_id in previous.auth:
                        auth[provider_id] = previous.auth[provider_id]
                    if provider_id in previous.configured_providers:
                        configured_providers.add(provider_id)
                    if provider_id in previous.stored_providers:
                        stored_providers.add(provider_id)
                available_by_id = {
                    (model.provider, model.id): model
                    for model in [
                        *[model for model in available_models if model.provider not in inflight],
                        *[model for model in previous.available if model.provider in inflight],
                    ]
                }
                available_models = [
                    available_by_id[(model.provider, model.id)]
                    for model in all_models
                    if (model.provider, model.id) in available_by_id
                ]
            self._snapshot = _Snapshot(
                all=all_models,
                available=available_models,
                configured_providers=configured_providers,
                stored_providers=stored_providers,
                auth=auth,
            )
            if error_seq == self._availability_error_seq:
                self._availability_error = None

    async def _queue_availability_refresh(self, cancel: CancelToken | None = None) -> None:
        with self._availability_guard:
            self._availability_refresh_seq += 1
            seq = self._availability_refresh_seq
            # pi bumps every provider seq here so this pass supersedes any
            # provider-scoped one still in flight. That invalidation is what
            # `_run_availability_refresh` now expresses through
            # `_provider_availability_inflight` instead — bumping the seq would
            # make the in-flight pass drop its own, newer publish.
            self._availability_error_seq += 1
            error_seq = self._availability_error_seq
        effective_cancel = cancel if cancel is not None else CancelToken()
        try:
            await self._run_availability_refresh(seq, error_seq, effective_cancel)
        except Exception as error:
            unwrapped = _unwrap_spawn_error(error)
            with self._availability_guard:
                # Only the latest requested rebuild owns the error state.
                if error_seq == self._availability_error_seq and not effective_cancel.cancelled:
                    self._availability_error = str(unwrapped)
            raise unwrapped from error

    async def _refresh_provider_availability(self, provider_id: str, cancel: CancelToken) -> None:
        """One credential-driven availability pass for ``provider_id``.

        pi drops the older of two overlapping passes by provider seq. That is
        safe on one loop: the pass that bumps last is always the one that
        started last, and a caller's own pass is never the dropped one unless
        something newer has already run. Here the passes run on different
        threads, so a login's pass could be superseded by the tail of a
        refresh its own recompose had just cancelled — and `login()` returned
        with the pre-credential snapshot still live, the correct publish
        landing a moment later, unawaited. Passes are serialized per provider
        instead: every caller returns after a publish that saw its credential
        state, and a later pass simply publishes later.
        """
        with self._availability_guard:
            lock = self._provider_availability_locks.get(provider_id)
            if lock is None:
                lock = self._provider_availability_locks[provider_id] = sync.Lock()
        async with lock:
            await self._run_provider_availability_refresh(provider_id, cancel)

    async def _run_provider_availability_refresh(self, provider_id: str, cancel: CancelToken) -> None:
        with self._availability_guard:
            # Invalidate any full availability pass that started before this credential change.
            self._availability_refresh_seq += 1
            self._availability_error_seq += 1
            error_seq = self._availability_error_seq
            self._provider_availability_inflight[provider_id] = (
                self._provider_availability_inflight.get(provider_id, 0) + 1
            )
        options = AuthOperationOptions(cancel=cancel)
        try:
            available, auth, credential = await tonio.spawn(
                self._models.get_available(provider_id, options),
                self._models.check_auth(provider_id, options),
                self._credentials.read(provider_id, options),
            )
            cancel.raise_if_cancelled()
            all_models = list(self._models.get_models())
            with self._availability_guard:
                configured_providers = set(self._snapshot.configured_providers)
                stored_providers = set(self._snapshot.stored_providers)
                auth_by_provider = dict(self._snapshot.auth)
                if auth is not None:
                    configured_providers.add(provider_id)
                    auth_by_provider[provider_id] = auth
                else:
                    configured_providers.discard(provider_id)
                    auth_by_provider.pop(provider_id, None)
                if credential is not None:
                    stored_providers.add(provider_id)
                else:
                    stored_providers.discard(provider_id)
                available_by_id = {
                    (model.provider, model.id): model
                    for model in [
                        *[model for model in self._snapshot.available if model.provider != provider_id],
                        *available,
                    ]
                }
                self._snapshot = _Snapshot(
                    all=all_models,
                    available=[
                        available_by_id[(model.provider, model.id)]
                        for model in all_models
                        if (model.provider, model.id) in available_by_id
                    ],
                    configured_providers=configured_providers,
                    stored_providers=stored_providers,
                    auth=auth_by_provider,
                )
                if error_seq == self._availability_error_seq:
                    self._availability_error = None
        except Exception as error:
            unwrapped = _unwrap_spawn_error(error)
            with self._availability_guard:
                if error_seq == self._availability_error_seq and not cancel.cancelled:
                    self._availability_error = str(unwrapped)
            raise unwrapped from error
        finally:
            with self._availability_guard:
                remaining = self._provider_availability_inflight.get(provider_id, 0) - 1
                if remaining > 0:
                    self._provider_availability_inflight[provider_id] = remaining
                else:
                    self._provider_availability_inflight.pop(provider_id, None)

    # -- reads -----------------------------------------------------------------

    def get_providers(self) -> list[Provider]:
        return self._models.get_providers()

    def get_provider(self, provider_id: str) -> Provider | None:
        return self._models.get_provider(provider_id)

    def get_models(self, provider_id: str | None = None) -> list[Model]:
        return self._models.get_models(provider_id)

    def get_model(self, provider_id: str, model_id: str) -> Model | None:
        return self._models.get_model(provider_id, model_id)

    async def check_auth(self, provider_id: str, options: AuthOperationOptions | None = None) -> AuthCheck | None:
        return await self._models.check_auth(provider_id, options)

    async def get_available(
        self, provider_id: str | None = None, options: AuthOperationOptions | None = None
    ) -> list[Model]:
        if provider_id:
            with self._availability_guard:
                self._availability_error_seq += 1
                error_seq = self._availability_error_seq
            try:
                available = await self._models.get_available(provider_id, options)
            except Exception as error:
                unwrapped = _unwrap_spawn_error(error)
                cancel = options.cancel if options is not None else None
                with self._availability_guard:
                    if error_seq == self._availability_error_seq and (cancel is None or not cancel.cancelled):
                        self._availability_error = str(unwrapped)
                raise unwrapped from error
            with self._availability_guard:
                if error_seq == self._availability_error_seq:
                    self._availability_error = None
            return available
        await self._queue_availability_refresh(options.cancel if options is not None else None)
        return self._snapshot.available

    def get_available_snapshot(self) -> list[Model]:
        return self._snapshot.available

    def get_error(self) -> str | None:
        # Pinned epoch: config and composition errors come from one
        # composition pass, and the iteration cannot race a writer's
        # clear()/pop() (the epoch's dicts are never mutated).
        composition = self._composition
        errors: list[str] = []
        config_error = composition.config.get_error()
        if config_error:
            errors.append(config_error)
        for provider_id, error in composition.composition_errors.items():
            errors.append(f'Provider "{provider_id}": {error}')
        availability_error = self._availability_error
        if availability_error:
            errors.append(f"Availability refresh: {availability_error}")
        return "\n\n".join(errors) if errors else None

    def get_registered_provider_config(self, provider_id: str) -> ProviderConfigInput | None:
        return self._composition.extension_providers.get(provider_id)

    def get_registered_provider_ids(self) -> list[str]:
        composition = self._composition
        seen: dict[str, None] = {}
        for provider_id in (*composition.extension_providers, *composition.native_extension_providers):
            seen[provider_id] = None
        return list(seen.keys())

    def get_registered_native_provider(self, provider_id: str) -> Provider | None:
        return self._composition.native_extension_providers.get(provider_id)

    async def get_compatibility_request_config(self, model: Model) -> CompatibilityRequestConfig:
        """Compatibility fallback for ModelRegistry when provider auth is unconfigured."""
        composition = self._composition
        return await resolve_compatibility_request_config(
            model,
            composition.config.get_provider(model.provider),
            composition.extension_providers.get(model.provider),
        )

    def is_using_oauth(self, provider_id: str) -> bool:
        check = self._snapshot.auth.get(provider_id)
        return check is not None and check.type == "oauth"

    def is_using_subscription(self, provider_id: str) -> bool:
        if not self.is_using_oauth(provider_id):
            return False
        provider = self._models.get_provider(provider_id)
        return provider is not None and provider.auth.oauth is not None and provider.auth.oauth.is_subscription is True

    def has_configured_auth(self, provider_id: str) -> bool:
        return provider_id in self._snapshot.configured_providers

    # -- auth ------------------------------------------------------------------

    async def get_auth(
        self,
        provider_or_model: str | Model,
        overrides: ModelRuntimeAuthOverrides | None = None,
    ) -> AuthResult | None:
        overrides = overrides if overrides is not None else ModelRuntimeAuthOverrides()
        if isinstance(provider_or_model, str):
            return await self._models.get_auth(
                provider_or_model,
                AuthResolutionOverrides(
                    api_key=overrides.api_key,
                    env=overrides.env,
                    min_oauth_validity_ms=overrides.min_oauth_validity_ms,
                    cancel=overrides.cancel,
                ),
            )
        provider = self._models.get_provider(provider_or_model.provider)
        if provider is None:
            return None
        return await self._get_model_auth(provider, provider_or_model, overrides)

    async def _get_model_auth(
        self,
        provider: Provider,
        model: Model,
        overrides: ModelRuntimeAuthOverrides,
    ) -> AuthResult | None:
        """`get_auth`'s Model path against a pinned provider and one pinned
        composition epoch: the auth resolution, the configured headers, and
        the provider the caller will stream with cannot come from different
        composition passes."""
        composition = self._composition
        resolution = await self._models.get_auth_for_provider(
            provider,
            model,
            AuthResolutionOverrides(
                api_key=overrides.api_key,
                env=overrides.env,
                min_oauth_validity_ms=overrides.min_oauth_validity_ms,
                cancel=overrides.cancel,
            ),
        )
        if resolution is None:
            return None
        configured_headers = await resolve_configured_model_headers(
            model,
            composition.config.get_provider(model.provider),
            composition.extension_providers.get(model.provider),
            {**(dict(resolution.env) if resolution.env else {}), **(overrides.env or {})},
        )
        return replace(
            resolution,
            auth=replace(resolution.auth, headers=merge_headers(resolution.auth.headers, configured_headers)),
        )

    async def _enqueue_credential_operation(
        self,
        provider_id: str,
        cancel: CancelToken,
        task: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Serialize credential operations per provider. A cancellation before
        the queued task begins rejects with the abort reason; once the task has
        started, its result (or failure) is awaited to completion."""
        with self._credential_operations_guard:
            lock = self._credential_operations.get(provider_id)
            if lock is None:
                # pi's per-provider promise chain; the lock is FIFO.
                lock = self._credential_operations[provider_id] = sync.Lock()

        started = tonio.Event()
        done = tonio.Event()
        # Fires on done *or* abort, whichever comes first, without a task per branch.
        progress = tonio.Event()
        outcome: list[tuple[str, Any]] = []

        async def _operation() -> None:
            try:
                async with lock:
                    cancel.raise_if_cancelled()
                    started.set()
                    outcome.append(("value", await task()))
            except BaseException as error:
                outcome.append(("error", error))
            finally:
                done.set()
                progress.set()

        tonio.spawn.without_tracking(_operation())
        unsubscribe = cancel.on_cancel(lambda _reason: progress.set())
        await progress.wait()
        unsubscribe()
        if cancel.cancelled and not started.is_set() and not done.is_set():
            raise cancel.reason  # type: ignore[misc]
        await done.wait()
        kind, payload = outcome[0]
        if kind == "error":
            raise payload
        return payload

    async def _synchronize_credential_state(
        self,
        provider_id: str,
        operation: CredentialSynchronizationOperation,
        credential: Credential | None,
        cancel: CancelToken,
    ) -> None:
        try:
            cancel.raise_if_cancelled()
            self._recompose_provider(provider_id)
            # Read from the epoch that recompose just published, not the
            # working dict a concurrent pass may be rewriting.
            composition_error = self._composition.composition_errors.get(provider_id)
            if composition_error:
                raise Exception(composition_error)
            result = await self._models.refresh(
                ModelsRefreshOptions(allow_network=False, providers=[provider_id], cancel=cancel)
            )
            if result.aborted:
                cancel.raise_if_cancelled()
            refresh_error = result.errors.get(provider_id)
            if refresh_error is not None:
                raise refresh_error
            self._update_model_snapshot()
            await self._refresh_provider_availability(provider_id, cancel)
        except BaseException as cause:
            raise CredentialSynchronizationError(provider_id, operation, credential, cause=cause)

    async def set_runtime_api_key(
        self,
        provider_id: str,
        api_key: str,
        options: AuthOperationOptions | None = None,
    ) -> None:
        cancel = operation_cancel(options.cancel if options is not None else None)

        async def task() -> None:
            self._credentials.set_runtime_api_key(provider_id, api_key)
            await self._synchronize_credential_state(
                provider_id, "setRuntimeApiKey", ApiKeyCredential(key=api_key), cancel
            )

        await self._enqueue_credential_operation(provider_id, cancel, task)

    async def remove_runtime_api_key(self, provider_id: str, options: AuthOperationOptions | None = None) -> None:
        cancel = operation_cancel(options.cancel if options is not None else None)

        async def task() -> None:
            self._credentials.remove_runtime_api_key(provider_id)
            await self._synchronize_credential_state(provider_id, "removeRuntimeApiKey", None, cancel)

        await self._enqueue_credential_operation(provider_id, cancel, task)

    async def list_credentials(self, options: AuthOperationOptions | None = None) -> list[CredentialInfo]:
        return await self._credentials.list(options)

    def get_provider_auth_status(self, provider_id: str) -> AuthStatus:
        # Pin both epochs once: the answer cannot mix an availability
        # snapshot from one pass with config/extensions from another.
        snapshot = self._snapshot
        composition = self._composition
        if self._credentials.has_runtime_api_key(provider_id):
            return AuthStatus(configured=True, source="runtime")
        if provider_id in snapshot.stored_providers:
            return AuthStatus(configured=True, source="stored")
        configured = configured_request_auth_status(
            composition.config.get_provider(provider_id),
            composition.extension_providers.get(provider_id),
        )
        if configured is not None:
            return configured
        check = snapshot.auth.get(provider_id)
        if check is not None:
            return AuthStatus(configured=True, source="environment", label=check.source)
        return AuthStatus(configured=False)

    # -- streaming -------------------------------------------------------------

    async def _prepare_request(
        self,
        model: Model,
        options: StreamOptions | None,
    ) -> tuple[Provider, Model, StreamOptions]:
        provider = self._models.get_provider(model.provider)
        if provider is None:
            raise ModelsError("provider", f"Unknown provider: {model.provider}")
        # Epoch discipline: auth resolves against the provider pinned above —
        # the one this request will stream with — not a fresh lookup.
        resolution = await self._get_model_auth(
            provider,
            model,
            ModelRuntimeAuthOverrides(
                api_key=options.api_key if options is not None else None,
                env=options.env if options is not None else None,
                cancel=options.cancel if options is not None else None,
            ),
        )
        if resolution is None:
            raise ModelsError("auth", f"Provider is not configured: {model.provider}")

        provider_options = options if options is not None else StreamOptions()
        transform_headers = provider_options.transform_headers
        headers = merge_headers(resolution.auth.headers, provider_options.headers)
        if transform_headers is not None:
            headers = await transform_headers(headers if headers is not None else {})
        env = (
            {**(dict(resolution.env) if resolution.env else {}), **(provider_options.env or {})}
            if resolution.env or provider_options.env
            else None
        )
        request_model = replace(model, base_url=resolution.auth.base_url) if resolution.auth.base_url else model
        request_options = replace(
            provider_options,
            api_key=provider_options.api_key if provider_options.api_key is not None else resolution.auth.api_key,
            headers=headers,
            env=env,
            transform_headers=None,
        )
        return provider, request_model, request_options

    def stream(self, model: Model, context: Context, options: StreamOptions | None = None):
        async def setup(stream):
            provider, request_model, request_options = await self._prepare_request(model, options)
            return call_stream_into(provider.stream, request_model, context, request_options, into=stream)

        return lazy_stream(model, setup, _cancel_of(options))

    async def complete(self, model: Model, context: Context, options: StreamOptions | None = None):
        return await self.stream(model, context, options).result()

    def stream_simple(self, model: Model, context: Context, options: SimpleStreamOptions | None = None):
        async def setup(stream):
            provider, request_model, request_options = await self._prepare_request(model, options)
            return call_stream_into(provider.stream_simple, request_model, context, request_options, into=stream)

        return lazy_stream(model, setup, _cancel_of(options))

    async def complete_simple(self, model: Model, context: Context, options: SimpleStreamOptions | None = None):
        return await self.stream_simple(model, context, options).result()

    async def fetch_deferred(self, model: Model, handle: DeferredHandle, options: StreamOptions | None = None):
        async def setup(_stream):
            provider, request_model, request_options = await self._prepare_request(model, options)
            if not getattr(provider, "supports_fetch_deferred", False):
                raise ModelsError("provider", f"Provider {model.provider} does not support deferred responses")
            return provider.fetch_deferred(request_model, handle, request_options)

        return await lazy_stream(model, setup, _cancel_of(options)).result()

    async def cancel_deferred(self, model: Model, handle: DeferredHandle, options: StreamOptions | None = None) -> None:
        provider, request_model, request_options = await self._prepare_request(model, options)
        if not getattr(provider, "supports_cancel_deferred", False):
            raise ModelsError("provider", f"Provider {model.provider} does not support deferred responses")
        await provider.cancel_deferred(request_model, handle, request_options)

    # -- lifecycle -------------------------------------------------------------

    async def login(self, provider_id: str, type: AuthType, interaction: AuthInteraction) -> Credential:
        cancel = operation_cancel(interaction.cancel)

        async def task() -> Credential:
            credential = await self._models.login(provider_id, type, _CancelBoundInteraction(interaction, cancel))
            await self._synchronize_credential_state(provider_id, "login", credential, cancel)
            return credential

        return await self._enqueue_credential_operation(provider_id, cancel, task)

    async def logout(self, provider_id: str, options: AuthOperationOptions | None = None) -> None:
        cancel = operation_cancel(options.cancel if options is not None else None)

        async def task() -> None:
            await self._models.logout(provider_id, AuthOperationOptions(cancel=cancel))
            await self._synchronize_credential_state(provider_id, "logout", None, cancel)

        await self._enqueue_credential_operation(provider_id, cancel, task)

    def _request_refresh(self) -> None:
        """Ask for a refresh without racing awaited ones (see __init__ notes)."""
        self._refresh_requested = True
        tonio.spawn.without_tracking(self._drain_refresh_requests())

    async def _drain_refresh_requests(self) -> None:
        if not self._refresh_requested:
            return  # A refresh that started after the request already ran.
        await self.refresh(ModelsRefreshOptions(allow_network=False), _requested_only=True)

    async def refresh(
        self, options: ModelsRefreshOptions | None = None, *, _requested_only: bool = False
    ) -> ModelsRefreshResult:
        options = options if options is not None else ModelsRefreshOptions(allow_network=self._model_network_enabled)
        cancel = options.cancel
        if options.providers is not None:
            # Provider-scoped refreshes must interleave freely (pi runs each
            # login/logout synchronization concurrently); cross-refresh
            # consistency comes from the per-provider generation guards.
            config = await ModelConfig.load(self._models_path)
            with self._composition_guard:
                self._config = config
                self._publish_composition()
                for provider_id in dict.fromkeys(options.providers):
                    self._recompose_provider(provider_id)
                self._update_model_snapshot()
            result = await self._models.refresh(options)
            errors = dict(result.errors)
            with self._composition_guard:
                self._update_model_snapshot()
        else:
            # One full refresh at a time: rebuild, registry refresh and snapshot
            # must see one provider generation (see `_refresh_serial` in __init__).
            async with self._refresh_serial:
                # Requests made before this run starts are satisfied by it; a
                # request landing mid-run sets the flag again and spawns its own
                # drain, which will run after this one releases the lock. A
                # drain that read its request before losing the lock race to a
                # full run re-checks here: its request is already satisfied,
                # and rebuilding again would supersede (cancel) provider
                # refreshes that started after that run returned.
                if _requested_only and not self._refresh_requested:
                    return ModelsRefreshResult(aborted=False, errors={})
                self._refresh_requested = False
                config = await ModelConfig.load(self._models_path)
                with self._composition_guard:
                    self._config = config
                    self._rebuild_providers()
                result = await self._models.refresh(options)
                errors = dict(result.errors)
                with self._composition_guard:
                    self._update_model_snapshot()
        if options.providers is not None:

            async def refresh_one_availability(provider_id: str) -> None:
                try:
                    await self._refresh_provider_availability(provider_id, operation_cancel(cancel))
                except Exception as error:
                    if cancel is None or not cancel.cancelled:
                        errors[provider_id] = error

            unique_providers = list(dict.fromkeys(options.providers))
            if unique_providers:
                await tonio.map(refresh_one_availability, unique_providers)
        else:
            try:
                await self._queue_availability_refresh(cancel)
            except Exception:
                pass  # Availability errors are recorded by the latest pass; refreshed models remain usable.
        return ModelsRefreshResult(
            aborted=result.aborted or (cancel.cancelled if cancel is not None else False), errors=errors
        )

    def register_native_provider(self, provider: Provider) -> None:
        if not provider.id.strip():
            raise Exception("Provider id must not be empty.")
        with self._composition_guard:
            self._extension_providers.pop(provider.id, None)
            self._native_extension_providers[provider.id] = provider
            self._recompose_provider(provider.id)
            self._update_model_snapshot()
        self._request_refresh()

    def register_provider(self, provider_id: str, config: ProviderConfigInput) -> None:
        with self._composition_guard:
            # Validate the incoming registration on its own, like the legacy registry:
            # a broken re-registration must throw without touching the stored config.
            validate_extension_provider(
                provider_id, self._builtins.get(provider_id), self._config.get_provider(provider_id), config
            )
            self._native_extension_providers.pop(provider_id, None)
            # Re-registration merges defined values over the previous registration and
            # preserves undefined ones, matching the legacy ModelRegistry contract.
            previous = self._extension_providers.get(provider_id)
            effective: ProviderConfigInput = {**(previous or {}), **config}
            self._extension_providers[provider_id] = effective
            self._recompose_provider(provider_id)
            self._update_model_snapshot()
            configured = configured_request_auth_status(self._config.get_provider(provider_id), effective)
            with self._availability_guard:
                if provider_id in self._snapshot.stored_providers or (configured is not None and configured.configured):
                    configured_providers = set(self._snapshot.configured_providers) | {provider_id}
                    auth = dict(self._snapshot.auth)
                    # Provisional entry until the async refresh lands; never clobber a real check result.
                    if not auth.get(provider_id):
                        auth[provider_id] = AuthCheck(
                            type="oauth" if effective.get("oauth") and not effective.get("apiKey") else "api_key",
                            source="configured provider",
                        )
                    self._snapshot = replace(
                        self._snapshot,
                        auth=auth,
                        configured_providers=configured_providers,
                        available=[model for model in self._snapshot.all if model.provider in configured_providers],
                    )
        self._request_refresh()

    def unregister_provider(self, provider_id: str) -> None:
        with self._composition_guard:
            self._extension_providers.pop(provider_id, None)
            self._native_extension_providers.pop(provider_id, None)
            self._recompose_provider(provider_id)
            self._update_model_snapshot()
        self._request_refresh()
