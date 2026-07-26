"""Mirror of pi's images-models.test.ts."""

import time
from types import SimpleNamespace

import pytest
import tonio.colored as tonio

from pidrei_ai.auth.resolve import ModelsError
from pidrei_ai.auth.types import ApiKeyAuth, AuthResult, ModelAuth, ProviderAuth
from pidrei_ai.images_models import create_images_models, create_images_provider
from pidrei_ai.providers.all import builtin_images_models
from pidrei_ai.types import AssistantImages, ImagesContext, ImagesModel, ImagesOptions, ModelCost, TextContent


CONTEXT = ImagesContext(input=[TextContent(text="draw")])


def make_image_model(provider: str, id: str) -> ImagesModel:
    return ImagesModel(
        id=id,
        name=id,
        api="test-images",
        provider=provider,
        base_url="https://example.invalid",
        input=["text"],
        output=["image"],
        cost=ModelCost(),
    )


def ok_result(model: ImagesModel) -> AssistantImages:
    return AssistantImages(
        api=model.api,
        provider=model.provider,
        model=model.id,
        output=[],
        stop_reason="stop",
        timestamp=int(time.time() * 1000),
    )


class _FakeAuthContext:
    def __init__(self, env: dict[str, str]):
        self._env = env

    async def env(self, name: str) -> str | None:
        return self._env.get(name)

    async def file_exists(self, _path: str) -> bool:
        return False


def env_key_auth(env_var: str) -> ApiKeyAuth:
    async def resolve(ctx, _credential):
        key = await ctx.env(env_var)
        return AuthResult(auth=ModelAuth(api_key=key), source=env_var) if key else None

    return ApiKeyAuth(name="Test key", resolve=resolve)


def make_provider(*, id: str, models=None, env_var: str = "TEST_KEY", calls: list | None = None):
    async def generate_images(model, _context, options=None):
        if calls is not None:
            calls.append({"model": model, "options": options})
        return ok_result(model)

    api = SimpleNamespace(generate_images=generate_images)

    return create_images_provider(
        id=id,
        auth=ProviderAuth(api_key=env_key_auth(env_var)),
        models=models if models is not None else [make_image_model(id, "model-a")],
        api=api,
    )


def test_registers_providers_and_reads_models_synchronously():
    models = create_images_models()
    models.set_provider(make_provider(id="p1", models=[make_image_model("p1", "m1"), make_image_model("p1", "m2")]))
    models.set_provider(make_provider(id="p2", models=[make_image_model("p2", "m3")]))

    assert [p.id for p in models.get_providers()] == ["p1", "p2"]
    assert [m.id for m in models.get_models()] == ["m1", "m2", "m3"]
    assert [m.id for m in models.get_models("p1")] == ["m1", "m2"]
    assert models.get_model("p2", "m3").id == "m3"
    assert models.get_model("p2", "missing") is None

    models.delete_provider("p1")
    assert models.get_provider("p1") is None


@pytest.mark.tonio
async def test_resolves_auth_through_the_provider_and_merges_it_explicit_options_win():
    calls: list = []
    models = create_images_models(auth_context=_FakeAuthContext({"TEST_KEY": "env-key"}))
    models.set_provider(make_provider(id="p1", env_var="TEST_KEY", calls=calls))
    model = models.get_model("p1", "model-a")

    assert (await models.get_auth(model)).auth.api_key == "env-key"
    assert (await models.get_auth(model.provider)).auth.api_key == "env-key"

    result = await models.generate_images(model, CONTEXT)
    assert result.stop_reason == "stop"
    assert calls[0]["options"].api_key == "env-key"

    await models.generate_images(model, CONTEXT, ImagesOptions(api_key="explicit"))
    assert calls[1]["options"].api_key == "explicit"


@pytest.mark.tonio
async def test_merges_provider_resolved_env_into_image_options():
    calls: list = []

    async def resolve(_ctx, _credential):
        return AuthResult(
            auth=ModelAuth(api_key="provider-key"),
            env={"PROVIDER_ONLY": "provider", "SHARED": "provider"},
        )

    async def generate_images(model, _context, options=None):
        calls.append({"model": model, "options": options})
        return ok_result(model)

    api = SimpleNamespace(generate_images=generate_images)

    models = create_images_models()
    models.set_provider(
        create_images_provider(
            id="p1",
            auth=ProviderAuth(api_key=ApiKeyAuth(name="Test key", resolve=resolve)),
            models=[make_image_model("p1", "model-a")],
            api=api,
        )
    )
    model = models.get_model("p1", "model-a")

    await models.generate_images(
        model,
        CONTEXT,
        ImagesOptions(api_key="request-key", env={"REQUEST_ONLY": "request", "SHARED": "request"}),
    )

    assert calls[0]["options"].api_key == "request-key"
    assert calls[0]["options"].env == {
        "PROVIDER_ONLY": "provider",
        "REQUEST_ONLY": "request",
        "SHARED": "request",
    }


@pytest.mark.tonio
async def test_returns_an_error_result_for_unknown_providers_and_unconfigured_auth():
    models = create_images_models(auth_context=_FakeAuthContext({}))
    ghost = await models.generate_images(make_image_model("ghost", "m"), CONTEXT)
    assert ghost.stop_reason == "error"
    assert "Unknown provider: ghost" in ghost.error_message

    # Unconfigured (resolve -> None) still dispatches; the provider decides.
    calls: list = []
    models.set_provider(make_provider(id="p1", env_var="MISSING", calls=calls))
    model = models.get_model("p1", "model-a")
    assert await models.get_auth(model) is None
    await models.generate_images(model, CONTEXT)
    assert calls[0]["options"] is None or calls[0]["options"].api_key is None


@pytest.mark.tonio
async def test_supports_dynamic_providers_via_refresh_with_in_flight_dedupe():
    fetches = []

    async def refresh_models():
        fetches.append(1)
        await tonio.sleep(0.005)
        return [make_image_model("dyn", "listed")]

    async def generate_images(model, _context, options=None):
        return ok_result(model)

    api = SimpleNamespace(generate_images=generate_images)

    async def resolve(_ctx, _credential):
        return AuthResult(auth=ModelAuth())

    models = create_images_models()
    models.set_provider(
        create_images_provider(
            id="dyn",
            auth=ProviderAuth(api_key=ApiKeyAuth(name="Test", resolve=resolve)),
            models=[],
            refresh_models=refresh_models,
            api=api,
        )
    )

    assert models.get_models("dyn") == []
    await tonio.spawn(models.refresh("dyn"), models.refresh("dyn"))
    assert len(fetches) == 1
    assert models.get_model("dyn", "listed") is not None

    async def failing_refresh():
        raise RuntimeError("fetch failed")

    models.set_provider(
        create_images_provider(
            id="flaky",
            auth=ProviderAuth(api_key=ApiKeyAuth(name="Test", resolve=resolve)),
            models=[],
            refresh_models=failing_refresh,
            api=api,
        )
    )

    with pytest.raises(ModelsError) as excinfo:
        await models.refresh("flaky")
    assert excinfo.value.code == "model_source"

    # Refreshing everything is best-effort and never raises.
    assert await models.refresh() is None


@pytest.mark.tonio
async def test_builtin_images_models_registers_the_openrouter_provider_with_its_catalog():
    models = builtin_images_models(auth_context=_FakeAuthContext({"OPENROUTER_API_KEY": "or-key"}))

    assert [p.id for p in models.get_providers()] == ["openrouter"]

    listed = models.get_models("openrouter")
    assert len(listed) > 0
    assert all(m.api == "openrouter-images" for m in listed)

    assert (await models.get_auth(listed[0])).auth.api_key == "or-key"
