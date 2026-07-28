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
from dataclasses import dataclass, field, replace

import tonio.colored as tonio
from tonio.colored import sync

from pidrei_ai.api.lazy import lazy_stream
from pidrei_ai.auth.resolve import AuthResolutionOverrides, ModelsError
from pidrei_ai.auth.types import (
    AuthCheck,
    AuthInteraction,
    AuthResult,
    AuthType,
    Credential,
    CredentialInfo,
    CredentialStore,
)
from pidrei_ai.models_store import ModelsStore
from pidrei_ai.providers.all import builtin_providers, get_builtin_model_data_generated_at
from pidrei_ai.registry import ModelsRefreshOptions, ModelsRefreshResult, Provider, create_models
from pidrei_ai.types import Context, Model, SimpleStreamOptions, StreamOptions
from pidrei_ai.utils.cancel import CancelToken
from pidrei_ai.utils.headers import merge_headers

from ..config import get_agent_dir
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


def _unwrap_spawn_error(error: Exception) -> Exception:
    """tonio surfaces child failures as ExceptionGroup('SpawnExceptionGroup');
    pi's contract exposes the underlying error — unwrap single-child groups."""
    while (
        isinstance(error, ExceptionGroup) and len(error.exceptions) == 1 and isinstance(error.exceptions[0], Exception)
    ):
        error = error.exceptions[0]
    return error


class _AvailabilityRun:
    __slots__ = ("done", "error")

    def __init__(self):
        self.done = tonio.Event()
        self.error: Exception | None = None


@dataclass(slots=True, kw_only=True)
class ModelRuntimeAuthOverrides:
    api_key: str | None = None
    env: dict[str, str] | None = None


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
        self._availability_refresh: _AvailabilityRun | None = None
        self._availability_error: str | None = None
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
        cancel = CancelToken() if refresh_from_network else None
        settled = tonio.Event()
        if cancel is not None:
            timeout_s = (model_refresh_timeout_ms if model_refresh_timeout_ms is not None else 15_000) / 1000

            async def watchdog() -> None:
                await settled.wait(timeout_s)
                if not settled.is_set():
                    cancel.cancel()

            tonio.spawn.without_tracking(watchdog())
        try:
            await runtime.refresh(ModelsRefreshOptions(allow_network=refresh_from_network, cancel=cancel))
        finally:
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

    def _recompose_provider(self, provider_id: str) -> None:
        with self._composition_guard:
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
            self._update_model_snapshot()

    def _update_model_snapshot(self) -> None:
        all_models = list(self._models.get_models())
        self._snapshot = replace(
            self._snapshot,
            all=all_models,
            available=[model for model in all_models if model.provider in self._snapshot.configured_providers],
        )

    # -- availability ----------------------------------------------------------

    async def _run_availability_refresh(self) -> None:
        providers = self._models.get_providers()

        async def check_one(provider: Provider) -> tuple[str, AuthCheck | None]:
            return provider.id, await self._models.check_auth(provider.id)

        async def check_all() -> list[tuple[str, AuthCheck | None]]:
            if not providers:
                return []
            return await tonio.map(check_one, providers)

        available, checks, credentials = await tonio.spawn(
            self._models.get_available(), check_all(), self._credentials.list()
        )
        auth = dict(checks)
        configured_providers = {provider_id for provider_id, check in checks if check is not None}
        self._snapshot = _Snapshot(
            all=list(self._models.get_models()),
            available=list(available),
            configured_providers=configured_providers,
            stored_providers={entry.provider_id for entry in credentials},
            auth=auth,
        )
        self._availability_error = None

    def _queue_availability_refresh(self, after: _AvailabilityRun | None) -> _AvailabilityRun:
        run = _AvailabilityRun()

        async def task() -> None:
            if after is not None:
                await after.done.wait()
            try:
                await self._run_availability_refresh()
            except Exception as error:
                unwrapped = _unwrap_spawn_error(error)
                self._availability_error = str(unwrapped)
                run.error = unwrapped
            finally:
                if self._availability_refresh is run:
                    self._availability_refresh = None
                run.done.set()

        self._availability_refresh = run
        tonio.spawn.without_tracking(task())
        return run

    async def _await_run(self, run: _AvailabilityRun) -> None:
        await run.done.wait()
        if run.error is not None:
            raise run.error

    def _refresh_availability(self) -> _AvailabilityRun:
        """Coalesce concurrent readers onto the pending refresh."""
        return self._availability_refresh or self._queue_availability_refresh(None)

    def _force_refresh_availability(self) -> _AvailabilityRun:
        """Mutations must not observe an in-flight refresh started before them."""
        return self._queue_availability_refresh(self._availability_refresh)

    # -- reads -----------------------------------------------------------------

    def get_providers(self) -> list[Provider]:
        return self._models.get_providers()

    def get_provider(self, provider_id: str) -> Provider | None:
        return self._models.get_provider(provider_id)

    def get_models(self, provider_id: str | None = None) -> list[Model]:
        return self._models.get_models(provider_id)

    def get_model(self, provider_id: str, model_id: str) -> Model | None:
        return self._models.get_model(provider_id, model_id)

    async def check_auth(self, provider_id: str) -> AuthCheck | None:
        return await self._models.check_auth(provider_id)

    async def get_available(self, provider_id: str | None = None) -> list[Model]:
        if provider_id:
            run = self._availability_refresh
            if run is not None:
                await self._await_run(run)
                return [model for model in self._snapshot.available if model.provider == provider_id]
            try:
                return await self._models.get_available(provider_id)
            except Exception as error:
                unwrapped = _unwrap_spawn_error(error)
                self._availability_error = str(unwrapped)
                raise unwrapped from error
        await self._await_run(self._refresh_availability())
        return self._snapshot.available

    def get_available_snapshot(self) -> list[Model]:
        return self._snapshot.available

    def get_error(self) -> str | None:
        errors: list[str] = []
        config_error = self._config.get_error()
        if config_error:
            errors.append(config_error)
        for provider_id, error in self._composition_errors.items():
            errors.append(f'Provider "{provider_id}": {error}')
        if self._availability_error:
            errors.append(f"Availability refresh: {self._availability_error}")
        return "\n\n".join(errors) if errors else None

    def get_registered_provider_config(self, provider_id: str) -> ProviderConfigInput | None:
        return self._extension_providers.get(provider_id)

    def get_registered_provider_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for provider_id in (*self._extension_providers.keys(), *self._native_extension_providers.keys()):
            seen[provider_id] = None
        return list(seen.keys())

    def get_registered_native_provider(self, provider_id: str) -> Provider | None:
        return self._native_extension_providers.get(provider_id)

    async def get_compatibility_request_config(self, model: Model) -> CompatibilityRequestConfig:
        """Compatibility fallback for ModelRegistry when provider auth is unconfigured."""
        return await resolve_compatibility_request_config(
            model,
            self._config.get_provider(model.provider),
            self._extension_providers.get(model.provider),
        )

    def is_using_oauth(self, provider_id: str) -> bool:
        check = self._snapshot.auth.get(provider_id)
        return check is not None and check.type == "oauth"

    def has_configured_auth(self, provider_id: str) -> bool:
        return provider_id in self._snapshot.configured_providers

    # -- auth ------------------------------------------------------------------

    async def get_auth(
        self,
        provider_or_model: str | Model,
        overrides: ModelRuntimeAuthOverrides | None = None,
    ) -> AuthResult | None:
        overrides = overrides if overrides is not None else ModelRuntimeAuthOverrides()
        resolution_overrides = AuthResolutionOverrides(api_key=overrides.api_key, env=overrides.env)
        if isinstance(provider_or_model, str):
            return await self._models.get_auth(provider_or_model, resolution_overrides)
        resolution = await self._models.get_auth(provider_or_model, resolution_overrides)
        if resolution is None:
            return None
        configured_headers = await resolve_configured_model_headers(
            provider_or_model,
            self._config.get_provider(provider_or_model.provider),
            self._extension_providers.get(provider_or_model.provider),
            {**(dict(resolution.env) if resolution.env else {}), **(overrides.env or {})},
        )
        return replace(
            resolution,
            auth=replace(resolution.auth, headers=merge_headers(resolution.auth.headers, configured_headers)),
        )

    async def set_runtime_api_key(
        self,
        provider_id: str,
        api_key: str,
        refresh_options: ModelsRefreshOptions | None = None,
    ) -> None:
        self._credentials.set_runtime_api_key(provider_id, api_key)
        auth = dict(self._snapshot.auth)
        auth[provider_id] = AuthCheck(type="api_key", source="runtime API key")
        configured_providers = set(self._snapshot.configured_providers) | {provider_id}
        stored_providers = set(self._snapshot.stored_providers) | {provider_id}
        self._snapshot = replace(
            self._snapshot,
            auth=auth,
            configured_providers=configured_providers,
            stored_providers=stored_providers,
            available=[model for model in self._snapshot.all if model.provider in configured_providers],
        )
        await self.refresh(refresh_options if refresh_options is not None else ModelsRefreshOptions())

    async def remove_runtime_api_key(self, provider_id: str) -> None:
        self._credentials.remove_runtime_api_key(provider_id)
        await self.refresh(ModelsRefreshOptions(allow_network=self._model_network_enabled))

    async def list_credentials(self) -> list[CredentialInfo]:
        return await self._credentials.list()

    def get_provider_auth_status(self, provider_id: str) -> AuthStatus:
        if self._credentials.has_runtime_api_key(provider_id):
            return AuthStatus(configured=True, source="runtime")
        if provider_id in self._snapshot.stored_providers:
            return AuthStatus(configured=True, source="stored")
        configured = configured_request_auth_status(
            self._config.get_provider(provider_id),
            self._extension_providers.get(provider_id),
        )
        if configured is not None:
            return configured
        check = self._snapshot.auth.get(provider_id)
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
        resolution = await self.get_auth(
            model,
            ModelRuntimeAuthOverrides(
                api_key=options.api_key if options is not None else None,
                env=options.env if options is not None else None,
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
        async def setup():
            provider, request_model, request_options = await self._prepare_request(model, options)
            return provider.stream(request_model, context, request_options)

        return lazy_stream(model, setup)

    async def complete(self, model: Model, context: Context, options: StreamOptions | None = None):
        return await self.stream(model, context, options).result()

    def stream_simple(self, model: Model, context: Context, options: SimpleStreamOptions | None = None):
        async def setup():
            provider, request_model, request_options = await self._prepare_request(model, options)
            return provider.stream_simple(request_model, context, request_options)

        return lazy_stream(model, setup)

    async def complete_simple(self, model: Model, context: Context, options: SimpleStreamOptions | None = None):
        return await self.stream_simple(model, context, options).result()

    # -- lifecycle -------------------------------------------------------------

    async def login(self, provider_id: str, type: AuthType, interaction: AuthInteraction) -> Credential:
        credential = await self._models.login(provider_id, type, interaction)
        await self.refresh(ModelsRefreshOptions(allow_network=self._model_network_enabled))
        return credential

    async def logout(self, provider_id: str) -> None:
        await self._models.logout(provider_id)
        # Reset credential-dependent compatibility projections before the unconfigured provider is skipped by refresh.
        self._recompose_provider(provider_id)
        await self.refresh(ModelsRefreshOptions(allow_network=self._model_network_enabled))

    def _request_refresh(self) -> None:
        """Ask for a refresh without racing awaited ones (see __init__ notes)."""
        self._refresh_requested = True
        tonio.spawn.without_tracking(self._drain_refresh_requests())

    async def _drain_refresh_requests(self) -> None:
        if not self._refresh_requested:
            return  # A refresh that started after the request already ran.
        await self.refresh(ModelsRefreshOptions(allow_network=False))

    async def refresh(self, options: ModelsRefreshOptions | None = None) -> ModelsRefreshResult:
        options = options if options is not None else ModelsRefreshOptions(allow_network=self._model_network_enabled)
        # One refresh at a time: rebuild, registry refresh and snapshot must
        # see one provider generation (see `_refresh_serial` in __init__).
        async with self._refresh_serial:
            # Requests made before this run starts are satisfied by it; a
            # request landing mid-run sets the flag again and spawns its own
            # drain, which will run after this one releases the lock.
            self._refresh_requested = False
            config = await ModelConfig.load(self._models_path)
            with self._composition_guard:
                self._config = config
                self._rebuild_providers()
            result = await self._models.refresh(options)
            with self._composition_guard:
                self._update_model_snapshot()
        try:
            await self._await_run(self._force_refresh_availability())
        except Exception:
            pass  # Availability errors are recorded by the refresh run; refreshed models remain usable.
        return result

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
