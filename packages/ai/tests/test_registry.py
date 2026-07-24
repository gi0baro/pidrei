"""Tests for the models registry port (registry.py)."""

import pytest

from pidrei_ai.auth.resolve import ModelsError
from pidrei_ai.auth.types import ApiKeyAuth, AuthResult, ModelAuth, ProviderAuth
from pidrei_ai.models_store import InMemoryModelsStore, ModelsStoreEntry
from pidrei_ai.providers.faux import faux_assistant_message
from pidrei_ai.registry import (
    Models,
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
    StartEvent,
    StreamOptions,
    Usage,
)
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
    async def resolve(ctx, credential):
        stored = credential.key if credential is not None and credential.key else None
        return AuthResult(auth=ModelAuth(api_key=stored or key), source="static")

    return ApiKeyAuth(name="static", resolve=resolve)


def unconfigured_auth() -> ProviderAuth:
    async def resolve(ctx, credential):
        return None

    return ProviderAuth(api_key=ApiKeyAuth(name="unconfigured", resolve=resolve))


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
async def test_explicit_api_key_option_wins():
    models, api = make_models_with_echo()
    await models.complete(make_model(), Context(), StreamOptions(api_key="explicit"))

    options = api.calls[0][2]
    assert options.api_key == "explicit"


@pytest.mark.tonio
async def test_headers_merge_and_transform():
    models, api = make_models_with_echo()
    model = make_model(headers={"X-Model": "m", "X-Base": "model"})

    def transform(headers):
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
    usage = Usage(input=1_000_000, output=100_000, cache_read=2_000_000, cache_write=500_000)

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
    usage = Usage(input=250_000, output=10_000, cache_read=150_000, cache_write=50_000)

    cost = calculate_cost(model, usage)
    assert cost.input == pytest.approx(3.0 / 1e6 * 250_000)
    assert cost.output == pytest.approx(6.0 / 1e6 * 10_000)


def test_calculate_cost_1h_cache_writes_cost_double_input_rate():
    # Anthropic: 1h cache writes bill at 2x the *input* rate, not the cacheWrite rate.
    model = make_model(cost=ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75))
    usage = Usage(cache_write=1_000, cache_write_1h=400)

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
