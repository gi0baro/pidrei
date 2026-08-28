"""Tests for the models registry port (registry.py)."""

import pytest
import tonio.colored as tonio

from pidrei_ai.auth.credential_store import InMemoryCredentialStore
from pidrei_ai.auth.resolve import AuthResolutionOverrides, ModelsError
from pidrei_ai.auth.types import (
    ApiKeyAuth,
    AuthCheck,
    AuthOperationOptions,
    AuthResult,
    ModelAuth,
    OAuthAuth,
    OAuthCredential,
    ProviderAuth,
)
from pidrei_ai.builders import UsageBuilder
from pidrei_ai.models_store import InMemoryModelsStore, ModelsStoreEntry
from pidrei_ai.providers.faux import faux_assistant_message
from pidrei_ai.registry import (
    Models,
    ModelsPublication,
    ModelsRefreshOptions,
    calculate_cost,
    clamp_thinking_level,
    create_models,
    create_provider,
    get_supported_thinking_levels,
    models_are_equal,
)
from pidrei_ai.types import (
    Context,
    DoneEvent,
    Model,
    ModelCost,
    ModelCostTier,
    SimpleStreamOptions,
    StartEvent,
    StreamOptions,
)
from pidrei_ai.utils.cancel import AbortError, CancelToken
from pidrei_ai.utils.event_stream import AssistantMessageEventStream


def make_model(provider: str = "test", id: str = "model-1", **overrides) -> Model:
    defaults: dict = {
        "id": id,
        "name": id,
        "api": "test-api",
        "provider": provider,
        "base_url": "https://api.test.example",
        "reasoning": False,
        "input": ["text"],
        "cost": ModelCost(),
        "context_window": 100_000,
        "max_tokens": 8_000,
    }
    defaults.update(overrides)
    return Model(**defaults)


def static_auth(key: str = "static-key") -> ProviderAuth:
    async def resolve(ctx, credential, cancel):
        stored = credential.key if credential is not None and credential.key else None
        return AuthResult(auth=ModelAuth(api_key=stored or key), source="static")

    return ApiKeyAuth(name="static", resolve=resolve)


def unconfigured_auth() -> ProviderAuth:
    async def resolve(ctx, credential, cancel):
        return None

    return ProviderAuth(api_key=ApiKeyAuth(name="unconfigured", resolve=resolve))


async def _noop_refresh(_context) -> None:
    return


def _far_future_ms() -> int:
    import time

    return int(time.time() * 1000) + 60_000


class DynamicTestProvider:
    """pi's `testProvider({id, refreshModels})` double: a raw provider object
    with a scripted `refresh_models`."""

    def __init__(self, id: str, refresh_models, auth: ProviderAuth | None = None, models: list | None = None):
        self.id = id
        self.name = id
        self.base_url = None
        self.headers = None
        self.auth = auth if auth is not None else ProviderAuth(api_key=static_auth())
        self.filter_models = None
        self._refresh_models = refresh_models
        self._models = models if models is not None else []

    @property
    def has_dynamic_models(self) -> bool:
        return True

    def get_models(self) -> list:
        return list(self._models)

    async def refresh_models(self, context) -> None:
        await self._refresh_models(context)

    def stream(self, model, context, options=None):
        raise RuntimeError("not used")

    def stream_simple(self, model, context, options=None):
        raise RuntimeError("not used")


class EchoApi:
    """ProviderStreams fake: records calls, emits one done message."""

    def __init__(self):
        self.calls: list[tuple[Model, Context, StreamOptions | None]] = []

    def _respond(self, model, context, options):
        self.calls.append((model, context, options))
        stream = AssistantMessageEventStream()
        message = faux_assistant_message("echo")
        stream.push(StartEvent(partial=message))
        stream.push(DoneEvent(reason="stop", message=message))
        return stream

    def stream(self, model, context, options=None):
        return self._respond(model, context, options)

    def stream_simple(self, model, context, options=None):
        return self._respond(model, context, options)


def make_models_with_echo(**provider_kwargs) -> tuple[Models, EchoApi]:
    api = EchoApi()
    models = create_models()
    models.set_provider(
        create_provider(
            id="test",
            auth=ProviderAuth(api_key=static_auth()),
            models=[make_model()],
            api=api,
            **provider_kwargs,
        )
    )
    return models, api


# -- provider collection -------------------------------------------------------


def test_provider_collection_crud():
    models, _api = make_models_with_echo()
    assert [provider.id for provider in models.get_providers()] == ["test"]
    assert models.get_provider("test") is not None
    assert models.get_model("test", "model-1") is not None
    assert models.get_model("test", "missing") is None
    assert models.get_models("unknown") == []

    models.delete_provider("test")
    assert models.get_providers() == []


def test_get_models_is_best_effort():
    models = create_models()

    class ThrowingProvider:
        id = "bad"

        def get_models(self):
            raise RuntimeError("boom")

    models.set_provider(ThrowingProvider())
    assert models.get_models() == []
    assert models.get_models("bad") == []


def test_create_provider_merges_dynamic_overlay():
    provider = create_provider(
        id="test",
        auth=ProviderAuth(api_key=static_auth()),
        models=[make_model(id="a"), make_model(id="b")],
        api=EchoApi(),
    )
    overlay_b = make_model(id="b", name="b-updated")
    provider._dynamic_models = [overlay_b, make_model(id="c")]

    merged = provider.get_models()
    assert [model.id for model in merged] == ["a", "b", "c"]
    assert merged[1].name == "b-updated"


# -- streaming through Models --------------------------------------------------


@pytest.mark.tonio
async def test_complete_applies_auth_and_dispatches():
    models, api = make_models_with_echo()
    result = await models.complete(make_model(), Context(messages=[]))

    assert result.content[0].text == "echo"
    assert len(api.calls) == 1
    _model, _context, options = api.calls[0]
    assert options is not None
    assert options.api_key == "static-key"


@pytest.mark.tonio
async def test_complete_simple_defaults_to_simple_options():
    # pidrei-only regression: pi passes untyped `options ?? {}` everywhere, but
    # the dataclass split means a None default here must build SimpleStreamOptions
    # (adapters read `.reasoning`), not StreamOptions.
    models, api = make_models_with_echo()
    await models.complete_simple(make_model(), Context(messages=[]))

    options = api.calls[0][2]
    assert isinstance(options, SimpleStreamOptions)
    assert options.reasoning is None
    assert options.api_key == "static-key"


@pytest.mark.tonio
async def test_explicit_api_key_option_wins():
    models, api = make_models_with_echo()
    await models.complete(make_model(), Context(), StreamOptions(api_key="explicit"))

    options = api.calls[0][2]
    assert options.api_key == "explicit"


@pytest.mark.tonio
async def test_headers_merge_and_transform():
    models, api = make_models_with_echo()
    model = make_model(headers={"X-Model": "m", "X-Base": "model"})

    async def transform(headers):
        return {**headers, "X-Transformed": "yes"}

    await models.complete(model, Context(), StreamOptions(headers={"x-base": "option"}, transform_headers=transform))

    options = api.calls[0][2]
    assert options.headers == {"X-Model": "m", "x-base": "option", "X-Transformed": "yes"}
    assert options.transform_headers is None


@pytest.mark.tonio
async def test_unknown_provider_streams_error_event():
    models = create_models()
    result = await models.complete(make_model(provider="ghost"), Context())

    assert result.stop_reason == "error"
    assert result.error_message == "Unknown provider: ghost"


@pytest.mark.tonio
async def test_unconfigured_provider_streams_error_event():
    models = create_models()
    models.set_provider(create_provider(id="test", auth=unconfigured_auth(), models=[make_model()], api=EchoApi()))
    result = await models.complete(make_model(), Context())

    assert result.stop_reason == "error"
    assert result.error_message == "Provider is not configured: test"


@pytest.mark.tonio
async def test_api_map_dispatch_and_missing_api_error():
    api = EchoApi()
    models = create_models()
    models.set_provider(
        create_provider(
            id="test",
            auth=ProviderAuth(api_key=static_auth()),
            models=[make_model()],
            api={"test-api": api},
        )
    )

    ok = await models.complete(make_model(), Context())
    assert ok.stop_reason == "stop"

    missing = await models.complete(make_model(id="other", api="other-api"), Context())
    assert missing.stop_reason == "error"
    assert missing.error_message == 'Provider test has no API implementation for "other-api"'


@pytest.mark.tonio
async def test_lazily_exposes_only_declared_deferred_capabilities():
    from pidrei_ai.api.lazy import lazy_api
    from pidrei_ai.types import DeferredHandle

    loads = 0
    streams = EchoApi()
    streams.fetch_deferred = lambda model, handle, options=None: streams.stream_simple(model, Context())

    async def load():
        nonlocal loads
        loads += 1
        return streams

    api = lazy_api(load, {"fetch_deferred": True})
    model = make_model(api="api-a", id="model-a")
    handle = DeferredHandle(provider=model.provider, model_id=model.id, api=model.api, id="response-1")

    assert loads == 0
    assert getattr(api, "cancel_deferred", None) is None
    assert (await api.fetch_deferred(model, handle).result()).stop_reason == "stop"
    assert loads == 1


@pytest.mark.tonio
async def test_applies_resolved_request_options_to_deferred_fetch_and_cancellation():
    from pidrei_ai.types import DeferredFetchOptions, DeferredHandle, ProviderRequestOptions

    captured: dict = {}
    api = EchoApi()

    def fetch_deferred(model, handle, options=None):
        captured["fetch_model"] = model
        captured["fetch_options"] = options
        return api.stream_simple(model, Context())

    async def cancel_deferred(model, handle, options=None):
        captured["cancel_options"] = options

    api.fetch_deferred = fetch_deferred
    api.cancel_deferred = cancel_deferred

    async def resolve(_ctx, _credential, _cancel):
        return AuthResult(
            auth=ModelAuth(
                api_key="provider-key",
                base_url="https://resolved.test/v1",
                headers={"Authorization": "Bearer provider", "X-Shared": "provider"},
            ),
            env={"PROVIDER_ONLY": "provider", "SHARED": "provider"},
        )

    deferred_model = make_model(provider="deferred-provider", id="model-a", api="api-a")
    models = create_models()
    models.set_provider(
        create_provider(
            id="deferred-provider",
            auth=ProviderAuth(api_key=ApiKeyAuth(name="Test", resolve=resolve)),
            models=[deferred_model],
            api=api,
        )
    )
    handle = DeferredHandle(
        provider=deferred_model.provider, model_id=deferred_model.id, api=deferred_model.api, id="response-1"
    )

    async def transform_fetch(headers):
        return {**headers, "X-Transformed": "yes"}

    async def transform_cancel(headers):
        return {**headers, "X-Cancel": "yes"}

    await models.fetch_deferred(
        deferred_model,
        handle,
        DeferredFetchOptions(
            wait=50,
            timeout_ms=100,
            api_key="request-key",
            headers={"X-Request": "request", "x-shared": "request"},
            env={"REQUEST_ONLY": "request", "SHARED": "request"},
            transform_headers=transform_fetch,
        ),
    )
    await models.cancel_deferred(
        deferred_model,
        handle,
        ProviderRequestOptions(timeout_ms=200, transform_headers=transform_cancel),
    )

    assert captured["fetch_model"].base_url == "https://resolved.test/v1"
    fetch_options = captured["fetch_options"]
    assert fetch_options.wait == 50
    assert fetch_options.timeout_ms == 100
    assert fetch_options.api_key == "request-key"
    assert fetch_options.headers == {
        "Authorization": "Bearer provider",
        "X-Request": "request",
        "x-shared": "request",
        "X-Transformed": "yes",
    }
    assert fetch_options.env == {"PROVIDER_ONLY": "provider", "REQUEST_ONLY": "request", "SHARED": "request"}
    cancel_options = captured["cancel_options"]
    assert cancel_options.timeout_ms == 200
    assert cancel_options.api_key == "provider-key"
    assert cancel_options.headers == {
        "Authorization": "Bearer provider",
        "X-Shared": "provider",
        "X-Cancel": "yes",
    }
    assert cancel_options.env == {"PROVIDER_ONLY": "provider", "SHARED": "provider"}


# -- refresh -------------------------------------------------------------------


@pytest.mark.tonio
async def test_refresh_fetches_persists_and_merges():
    store = InMemoryModelsStore()
    fetched = [make_model(id="dynamic-1")]
    fetch_calls = 0

    async def fetch_models(context):
        nonlocal fetch_calls
        fetch_calls += 1
        return fetched

    models = create_models(models_store=store)
    models.set_provider(
        create_provider(
            id="test",
            auth=ProviderAuth(api_key=static_auth()),
            models=[make_model(id="static-1")],
            fetch_models=fetch_models,
            api=EchoApi(),
        )
    )

    result = await models.refresh()
    assert result.aborted is False
    assert result.errors == {}
    assert fetch_calls == 1
    assert [model.id for model in models.get_models("test")] == ["static-1", "dynamic-1"]

    entry = await store.read("test")
    assert entry is not None
    assert [model.id for model in entry.models] == ["dynamic-1"]


@pytest.mark.tonio
async def test_refresh_collects_errors_and_restores_cache():
    store = InMemoryModelsStore()
    await store.write("test", ModelsStoreEntry(models=[make_model(id="cached-1")]))

    async def fetch_models(context):
        if context.allow_network:
            raise RuntimeError("network down")
        return []

    models = create_models(models_store=store)
    models.set_provider(
        create_provider(
            id="test",
            auth=ProviderAuth(api_key=static_auth()),
            models=[make_model(id="static-1")],
            fetch_models=fetch_models,
            api=EchoApi(),
        )
    )

    result = await models.refresh()
    assert result.aborted is False
    assert set(result.errors) == {"test"}
    assert str(result.errors["test"]) == "network down"
    # The cache-restore pass still loaded the stored overlay.
    assert [model.id for model in models.get_models("test")] == ["static-1", "cached-1"]


@pytest.mark.tonio
async def test_refresh_skips_unconfigured_providers():
    calls = 0

    async def fetch_models(context):
        nonlocal calls
        calls += 1
        return []

    models = create_models()
    models.set_provider(
        create_provider(
            id="test",
            auth=unconfigured_auth(),
            models=[],
            fetch_models=fetch_models,
            api=EchoApi(),
        )
    )

    result = await models.refresh(ModelsRefreshOptions())
    assert result.errors == {}
    assert calls == 0


@pytest.mark.tonio
async def test_restricts_refresh_work_to_selected_providers():
    calls: list[str] = []
    models = create_models()
    for provider_id in ("one", "two"):

        async def refresh(context, provider_id=provider_id):
            calls.append(f"{provider_id}:{'network' if context.allow_network else 'cache'}")

        models.set_provider(DynamicTestProvider(provider_id, refresh))

    result = await models.refresh(ModelsRefreshOptions(providers=["two", "unknown"]))

    assert result.errors == {}
    assert calls == ["two:cache", "two:network"]


@pytest.mark.tonio
async def test_restores_cached_models_before_waiting_for_network_auth():
    store = InMemoryModelsStore()
    await store.write("dynamic", ModelsStoreEntry(models=[make_model(provider="dynamic", id="cached")]))
    auth_started = tonio.Event()
    blocked_auth = tonio.Event()

    async def blocked_resolve(_ctx, _credential, _cancel):
        auth_started.set()
        await blocked_auth.wait()
        return AuthResult(auth=ModelAuth(api_key="key"))

    async def fetch_models(_context):
        raise RuntimeError("must not fetch")

    provider = create_provider(
        id="dynamic",
        auth=ProviderAuth(api_key=ApiKeyAuth(name="Blocked auth", resolve=blocked_resolve)),
        models=[],
        fetch_models=fetch_models,
        api=EchoApi(),
    )
    models = create_models(models_store=store)
    models.set_provider(provider)
    controller = CancelToken()
    outcome: dict = {}

    async def run_refresh() -> None:
        outcome["result"] = await models.refresh(ModelsRefreshOptions(providers=["dynamic"], cancel=controller))

    async def drive() -> None:
        await auth_started.wait()
        assert models.get_model("dynamic", "cached") is not None
        controller.cancel()

    await tonio.spawn(run_refresh(), drive())
    assert outcome["result"].aborted is True
    blocked_auth.set()


@pytest.mark.tonio
async def test_lets_providers_choose_persistent_deletion_and_ephemeral_publication_atomically():
    state = {"entry": ModelsStoreEntry(models=[make_model(provider="dynamic", id="stored")]), "phase": "initial"}

    class RecordingStore:
        async def read(self, _provider_id, options=None):
            return state["entry"]

        async def write(self, _provider_id, entry, options=None):
            state["entry"] = entry

        async def delete(self, _provider_id, options=None):
            state["entry"] = None

    models = create_models(models_store=RecordingStore())

    async def refresh(context):
        assert context.stored is not None
        assert context.stored.models[0].id == "stored"

        def deleted() -> None:
            assert state["entry"] is None
            state["phase"] = "deleted"

        assert await context.publish(ModelsPublication(persist=None, update=deleted)) is True

        def ephemeral() -> None:
            state["phase"] = "ephemeral"

        assert await context.publish(ModelsPublication(update=ephemeral)) is True

    models.set_provider(DynamicTestProvider("dynamic", refresh))

    result = await models.refresh(ModelsRefreshOptions(allow_network=False))

    assert result.errors == {}
    assert state["entry"] is None
    assert state["phase"] == "ephemeral"


@pytest.mark.tonio
async def test_always_gives_providers_a_concrete_cancel():
    received: dict = {}
    models = create_models()

    async def refresh(context):
        received["cancel"] = context.cancel

    models.set_provider(DynamicTestProvider("dynamic", refresh))

    result = await models.refresh()
    assert result.aborted is False
    assert isinstance(received["cancel"], CancelToken)
    assert received["cancel"].cancelled is False


@pytest.mark.tonio
async def test_binds_model_store_waits_to_the_provider_refresh_cancel():
    storage_cancels: list = []

    class RecordingStore:
        async def read(self, _provider_id, options=None):
            storage_cancels.append(options.cancel if options is not None else None)

        async def write(self, _provider_id, _entry, options=None):
            storage_cancels.append(options.cancel if options is not None else None)

        async def delete(self, _provider_id, options=None):
            storage_cancels.append(options.cancel if options is not None else None)

    received: dict = {}
    models = create_models(models_store=RecordingStore())

    async def refresh(context):
        received["cancel"] = context.cancel
        if not context.allow_network:
            return
        await context.publish(
            ModelsPublication(persist=ModelsStoreEntry(models=[make_model(provider="dynamic", id="fresh")]))
        )

    models.set_provider(DynamicTestProvider("dynamic", refresh))

    result = await models.refresh(ModelsRefreshOptions(providers=["dynamic"]))

    assert result.errors == {}
    assert len(storage_cancels) == 3
    assert all(cancel is received["cancel"] for cancel in storage_cancels)


@pytest.mark.tonio
async def test_returns_aborted_state_without_reporting_cancellation_as_a_provider_error():
    controller = CancelToken()
    models = create_models()

    async def refresh(context):
        controller.cancel()
        if context.cancel.cancelled:
            return

    models.set_provider(DynamicTestProvider("dynamic", refresh))

    result = await models.refresh(ModelsRefreshOptions(cancel=controller))
    assert result.aborted is True
    assert result.errors == {}


@pytest.mark.tonio
async def test_stops_waiting_on_abort_when_a_provider_ignores_its_cancel():
    controller = CancelToken()
    started = tonio.Event()
    release = tonio.Event()
    state = {"calls": 0, "fail_late": False}

    async def refresh(_context):
        state["calls"] += 1
        if state["calls"] != 1:
            return
        started.set()
        await release.wait()
        if state["fail_late"]:
            raise RuntimeError("late provider failure")

    models = create_models()
    models.set_provider(DynamicTestProvider("dynamic", refresh))
    outcome: dict = {}

    async def run_refresh() -> None:
        outcome["result"] = await models.refresh(ModelsRefreshOptions(cancel=controller))

    async def drive() -> None:
        await started.wait()
        controller.cancel()

    await tonio.spawn(run_refresh(), drive())
    result = outcome["result"]
    assert result.aborted is True
    assert result.errors == {}

    state["fail_late"] = True
    release.set()
    await tonio.time.sleep(0.01)
    assert result.errors == {}


@pytest.mark.tonio
async def test_rejects_late_publication_from_a_superseded_non_cooperative_provider():
    store = InMemoryModelsStore()
    state = {"value": "initial", "calls": 0}
    first_started = tonio.Event()
    first_blocked = tonio.Event()
    first_finished = tonio.Event()

    async def refresh(context):
        if not context.allow_network:
            return
        state["calls"] += 1
        current = state["calls"]
        if current == 1:
            first_started.set()
            await first_blocked.wait()
        value = f"generation-{current}"

        def apply(value: str = value) -> None:
            state["value"] = value

        try:
            await context.publish(
                ModelsPublication(
                    persist=ModelsStoreEntry(models=[make_model(provider="dynamic", id=value)]),
                    update=apply,
                )
            )
        finally:
            if current == 1:
                first_finished.set()

    models = create_models(models_store=store)
    models.set_provider(DynamicTestProvider("dynamic", refresh))

    async def run_first() -> None:
        await models.refresh(ModelsRefreshOptions(providers=["dynamic"]))

    async def drive() -> None:
        await first_started.wait()
        await models.refresh(ModelsRefreshOptions(providers=["dynamic"]))

    await tonio.spawn(run_first(), drive())
    first_blocked.set()
    # Wait for the superseded provider's late publication to settle — a parked
    # leftover would outlive the test (and wedge interpreter shutdown).
    await first_finished.wait()
    await tonio.time.sleep(0.01)

    assert state["value"] == "generation-2"
    entry = await store.read("dynamic")
    assert entry is not None
    assert entry.models[0].id == "generation-2"


@pytest.mark.tonio
async def test_lets_a_newer_dynamic_refresh_bypass_and_supersede_older_network_work():
    state = {"fetches": 0}
    first_started = tonio.Event()
    first_blocked = tonio.Event()

    async def fetch_models(_context):
        state["fetches"] += 1
        current = state["fetches"]
        if current == 1:
            first_started.set()
            await first_blocked.wait()
        return [make_model(provider="dynamic", id=f"listed-{current}", api="api-a")]

    provider = create_provider(
        id="dynamic",
        auth=ProviderAuth(api_key=static_auth()),
        models=[],
        fetch_models=fetch_models,
        api=EchoApi(),
    )
    store = InMemoryModelsStore()
    models = create_models(models_store=store)
    models.set_provider(provider)
    assert provider.get_models() == []
    outcome: dict = {}

    async def run_first() -> None:
        outcome["first"] = await models.refresh(ModelsRefreshOptions(providers=["dynamic"]))

    async def drive() -> None:
        await first_started.wait()
        outcome["second"] = await models.refresh(ModelsRefreshOptions(providers=["dynamic"]))

    await tonio.spawn(run_first(), drive())
    assert outcome["second"].aborted is False
    assert outcome["first"].aborted is False
    assert state["fetches"] == 2
    assert [model.id for model in provider.get_models()] == ["listed-2"]
    assert [model.id for model in (await store.read("dynamic")).models] == ["listed-2"]

    first_blocked.set()
    await tonio.time.sleep(0.01)
    assert [model.id for model in provider.get_models()] == ["listed-2"]
    assert [model.id for model in (await store.read("dynamic")).models] == ["listed-2"]


@pytest.mark.tonio
async def test_passes_caller_cancels_to_provider_auth_callbacks():
    controller = CancelToken()
    received: list = []

    async def login(interaction):
        received.append(interaction.cancel)
        return __import__("pidrei_ai.auth.types", fromlist=["ApiKeyCredential"]).ApiKeyCredential(key="saved")

    async def check(_ctx, _credential, cancel):
        received.append(cancel)
        return AuthCheck(type="api_key")

    async def resolve(_ctx, _credential, cancel):
        received.append(cancel)
        return AuthResult(auth=ModelAuth(api_key="resolved"))

    api_key = ApiKeyAuth(name="Signal auth", login=login, check=check, resolve=resolve)
    models = create_models()
    models.set_provider(DynamicTestProvider("p1", _noop_refresh, auth=ProviderAuth(api_key=api_key)))

    class Interaction:
        cancel = controller

        async def prompt(self, prompt):
            return "unused"

        def notify(self, event):
            pass

    await models.check_auth("p1", AuthOperationOptions(cancel=controller))
    await models.get_auth("p1", AuthResolutionOverrides(cancel=controller))
    await models.login("p1", "api_key", Interaction())

    assert received == [controller, controller, controller]


@pytest.mark.tonio
async def test_stops_waiting_for_non_cooperative_auth_callbacks():
    check_started = tonio.Event()
    blocked_check = tonio.Event()
    resolve_started = tonio.Event()
    blocked_resolve = tonio.Event()

    async def check(_ctx, _credential, _cancel):
        check_started.set()
        await blocked_check.wait()
        return AuthCheck(type="api_key")

    async def resolve(_ctx, _credential, _cancel):
        resolve_started.set()
        await blocked_resolve.wait()
        return AuthResult(auth=ModelAuth(api_key="key"))

    models = create_models()
    models.set_provider(
        DynamicTestProvider(
            "p1",
            _noop_refresh,
            auth=ProviderAuth(api_key=ApiKeyAuth(name="Blocked auth", check=check, resolve=resolve)),
        )
    )

    available_controller = CancelToken()
    available_outcome: dict = {}

    async def run_available() -> None:
        try:
            await models.get_available(None, AuthOperationOptions(cancel=available_controller))
            available_outcome["error"] = None
        except BaseException as error:
            available_outcome["error"] = error

    async def drive_available() -> None:
        await check_started.wait()
        available_controller.cancel()

    await tonio.spawn(run_available(), drive_available())
    assert isinstance(available_outcome["error"], AbortError)

    auth_controller = CancelToken()
    auth_outcome: dict = {}

    async def run_auth() -> None:
        try:
            await models.get_auth("p1", AuthResolutionOverrides(cancel=auth_controller))
            auth_outcome["error"] = None
        except BaseException as error:
            auth_outcome["error"] = error

    async def drive_auth() -> None:
        await resolve_started.wait()
        auth_controller.cancel()

    await tonio.spawn(run_auth(), drive_auth())
    assert isinstance(auth_outcome["error"], AbortError)

    blocked_check.set()
    blocked_resolve.set()


@pytest.mark.tonio
async def test_cancels_queued_credential_mutations_without_running_them_later():
    credentials = InMemoryCredentialStore()
    first_entered = tonio.Event()
    first_blocked = tonio.Event()
    state = {"second_ran": False}

    from pidrei_ai.auth.types import ApiKeyCredential

    async def first_fn(_current):
        first_entered.set()
        await first_blocked.wait()
        return ApiKeyCredential(key="first")

    async def second_fn(_current):
        state["second_ran"] = True
        return ApiKeyCredential(key="second")

    controller = CancelToken()
    outcome: dict = {}

    async def run_first() -> None:
        outcome["first"] = await credentials.modify("p1", first_fn)

    async def run_second() -> None:
        # pi registers both mutations on the chain synchronously; here the
        # ordering handshake is explicit.
        await first_entered.wait()
        try:
            await credentials.modify("p1", second_fn, AuthOperationOptions(cancel=controller))
            outcome["second_error"] = None
        except BaseException as error:
            outcome["second_error"] = error

    async def drive() -> None:
        await first_entered.wait()
        await tonio.time.sleep(0.01)
        controller.cancel()
        first_blocked.set()

    await tonio.spawn(run_first(), run_second(), drive())
    await tonio.time.sleep(0.01)

    assert isinstance(outcome["second_error"], AbortError)
    assert state["second_ran"] is False
    assert await credentials.read("p1") == ApiKeyCredential(key="first")


@pytest.mark.tonio
async def test_passes_cancellation_to_oauth_refresh_and_preserves_the_previous_credential():
    credentials = InMemoryCredentialStore()
    previous = OAuthCredential(access="old", refresh="old-refresh", expires=0)

    async def seed(_current):
        return previous

    await credentials.modify("p1", seed)
    refresh_started = tonio.Event()
    blocked_refresh = tonio.Event()
    received: dict = {}

    async def refresh(_credential, cancel):
        received["cancel"] = cancel
        refresh_started.set()
        await blocked_refresh.wait()
        return OAuthCredential(access="new", refresh="old-refresh", expires=_far_future_ms())

    async def login(_interaction):
        raise RuntimeError("not used")

    async def to_auth(credential):
        return ModelAuth(api_key=credential.access)

    models = create_models(credentials=credentials)
    models.set_provider(
        DynamicTestProvider(
            "p1",
            _noop_refresh,
            auth=ProviderAuth(oauth=OAuthAuth(name="Test OAuth", login=login, refresh=refresh, to_auth=to_auth)),
        )
    )
    controller = CancelToken()
    outcome: dict = {}

    async def run_auth() -> None:
        try:
            await models.get_auth("p1", AuthResolutionOverrides(cancel=controller))
            outcome["error"] = None
        except BaseException as error:
            outcome["error"] = error

    async def drive() -> None:
        await refresh_started.wait()
        controller.cancel()

    await tonio.spawn(run_auth(), drive())

    assert isinstance(outcome["error"], AbortError)
    # d3da2e968: the refresh receives a composite token carrying the caller's reason.
    assert isinstance(received["cancel"], CancelToken)
    assert received["cancel"].cancelled is True
    assert received["cancel"].reason is controller.reason
    blocked_refresh.set()
    await tonio.time.sleep(0.01)
    assert await credentials.read("p1") == previous


# -- availability --------------------------------------------------------------


@pytest.mark.tonio
async def test_get_available_filters_unconfigured_and_applies_filter_models():
    models = create_models()
    models.set_provider(
        create_provider(
            id="configured",
            auth=ProviderAuth(api_key=static_auth()),
            models=[make_model("configured", "a"), make_model("configured", "b")],
            filter_models=lambda entries, credential: [entry for entry in entries if entry.id == "a"],
            api=EchoApi(),
        )
    )
    models.set_provider(
        create_provider(
            id="unconfigured",
            auth=unconfigured_auth(),
            models=[make_model("unconfigured", "c")],
            api=EchoApi(),
        )
    )

    available = await models.get_available()
    assert [(model.provider, model.id) for model in available] == [("configured", "a")]

    check = await models.check_auth("configured")
    assert check is not None and check.type == "api_key" and check.source == "static"
    assert await models.check_auth("unconfigured") is None
    assert await models.check_auth("ghost") is None


# -- model helpers -------------------------------------------------------------


def test_calculate_cost_base_rates():
    model = make_model(cost=ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75))
    # Step 2 translation (PROPER_MT_DESIGN.md): calculate_cost is producer-side
    # and mutates a usage *builder*; the frozen Usage is a value.
    usage = UsageBuilder(input=1_000_000, output=100_000, cache_read=2_000_000, cache_write=500_000)

    cost = calculate_cost(model, usage)
    assert cost.input == pytest.approx(3.0)
    assert cost.output == pytest.approx(1.5)
    assert cost.cache_read == pytest.approx(0.6)
    assert cost.cache_write == pytest.approx(1.875)
    assert cost.total == pytest.approx(3.0 + 1.5 + 0.6 + 1.875)
    assert usage.cost is cost  # mutates in place, pi-style


def test_calculate_cost_applies_highest_matching_tier():
    model = make_model(
        cost=ModelCost(
            input=1.0,
            output=2.0,
            cache_read=0.1,
            cache_write=1.25,
            tiers=[
                ModelCostTier(input=2.0, output=4.0, cache_read=0.2, cache_write=2.5, input_tokens_above=200_000),
                ModelCostTier(input=3.0, output=6.0, cache_read=0.3, cache_write=3.75, input_tokens_above=400_000),
            ],
        )
    )
    # input + cacheRead + cacheWrite = 450k -> second tier applies to the whole request.
    usage = UsageBuilder(input=250_000, output=10_000, cache_read=150_000, cache_write=50_000)

    cost = calculate_cost(model, usage)
    assert cost.input == pytest.approx(3.0 / 1e6 * 250_000)
    assert cost.output == pytest.approx(6.0 / 1e6 * 10_000)


def test_calculate_cost_1h_cache_writes_cost_double_input_rate():
    # Anthropic: 1h cache writes bill at 2x the *input* rate, not the cacheWrite rate.
    model = make_model(cost=ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75))
    usage = UsageBuilder(cache_write=1_000, cache_write_1h=400)

    cost = calculate_cost(model, usage)
    expected = (3.75 * 600 + 3.0 * 2 * 400) / 1e6
    assert cost.cache_write == pytest.approx(expected)


def test_thinking_levels_non_reasoning_model():
    assert get_supported_thinking_levels(make_model(reasoning=False)) == ["off"]


def test_thinking_levels_default_reasoning_model():
    # Without a map: xhigh/max need explicit entries, the rest are supported.
    model = make_model(reasoning=True)
    assert get_supported_thinking_levels(model) == ["off", "minimal", "low", "medium", "high"]


def test_thinking_levels_map_none_disables_and_explicit_enables():
    model = make_model(
        reasoning=True,
        thinking_level_map={"minimal": None, "xhigh": "ultra", "max": None},
    )
    assert get_supported_thinking_levels(model) == ["off", "low", "medium", "high", "xhigh"]


def test_clamp_thinking_level():
    model = make_model(reasoning=True, thinking_level_map={"xhigh": "ultra"})
    assert clamp_thinking_level(model, "high") == "high"
    assert clamp_thinking_level(model, "max") == "xhigh"  # clamps down to nearest supported

    non_reasoning = make_model(reasoning=False)
    assert clamp_thinking_level(non_reasoning, "high") == "off"


def test_models_are_equal():
    a = make_model("p1", "m1")
    assert models_are_equal(a, make_model("p1", "m1")) is True
    assert models_are_equal(a, make_model("p2", "m1")) is False
    assert models_are_equal(a, make_model("p1", "m2")) is False
    assert models_are_equal(a, None) is False
    assert models_are_equal(None, None) is False


def test_models_error_carries_code():
    error = ModelsError("provider", "Unknown provider: x")
    assert error.code == "provider"
    assert str(error) == "Unknown provider: x"


# -- one stream per request ----------------------------------------------------


@pytest.mark.tonio
async def test_adapters_produce_straight_into_the_stream_the_caller_holds():
    """Models → Provider → LazyApi → adapter is one stream object, not a
    chain of forwarded ones: what `Models.stream` returns is what the adapter
    pushed into."""
    from pidrei_ai.api.lazy import lazy_api

    seen: list[AssistantMessageEventStream] = []

    class _Module:
        @staticmethod
        def stream(model, context, options=None, *, into=None):
            assert into is not None
            seen.append(into)
            message = faux_assistant_message("echo")
            into.push(StartEvent(partial=message))
            into.push(DoneEvent(reason="stop", message=message))
            return into

    async def load():
        return _Module

    models = create_models()
    models.set_provider(
        create_provider(id="test", auth=ProviderAuth(api_key=static_auth()), models=[make_model()], api=lazy_api(load))
    )

    stream = models.stream(make_model(), Context())
    events = [event.type async for event in stream]

    assert events == ["start", "done"]
    assert seen == [stream]


@pytest.mark.tonio
async def test_a_provider_that_ignores_into_is_still_forwarded():
    models, _api = make_models_with_echo()  # EchoApi returns its own stream

    stream = models.stream(make_model(), Context())
    events = [event.type async for event in stream]

    assert events == ["start", "done"]
    assert (await stream.result()).stop_reason == "stop"
