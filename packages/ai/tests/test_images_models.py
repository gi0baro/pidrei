"""Mirror of pi's images-models.test.ts.

pi's two compat-module reads (`getModels` from src/compat.ts) have no pidrei
counterpart (the compat registry is unported), so those assertions are dropped.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from pidrei_ai.auth.types import ApiKeyAuth, AuthResult, ModelAuth, ProviderAuth
from pidrei_ai.models_store import InMemoryModelsStore
from pidrei_ai.providers.all import (
    builtin_models,
    get_all_builtin_models,
    get_builtin_classifier_models,
    get_builtin_image_model,
    get_builtin_image_models,
    get_builtin_models,
)
from pidrei_ai.registry import (
    ModelsRefreshOptions,
    create_models,
    create_provider,
    get_model_type,
    has_api,
    is_model_type,
)
from pidrei_ai.types import (
    AssistantImages,
    Context,
    ImageModel,
    ImagesContext,
    ImagesOptions,
    Model,
    ModelCost,
    SimpleStreamOptions,
    TextContent,
)
from pidrei_ai.utils.event_stream import AssistantMessageEventStream
from pidrei_utils.cancel import CancelToken


class _FakeAuthContext:
    def __init__(self, env: dict[str, str]):
        self._env = env

    async def env(self, name: str) -> str | None:
        return self._env.get(name)

    async def file_exists(self, _path: str) -> bool:
        return False


def image_model(provider: str, id: str) -> ImageModel:
    return ImageModel(
        id=id,
        name=id,
        api="test-images",
        provider=provider,
        base_url="https://example.test/v1",
        input=["text"],
        output=["image"],
        cost=ModelCost(),
    )


def chat_model(provider: str, id: str) -> Model:
    return Model(
        id=id,
        name=id,
        api="test-chat",
        provider=provider,
        base_url="https://example.test/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(),
        context_window=1000,
        max_tokens=100,
    )


def ok_result(model: ImageModel) -> AssistantImages:
    return AssistantImages(
        api=model.api, provider=model.provider, model=model.id, output=[], stop_reason="stop", timestamp=0
    )


def _chat_streams() -> SimpleNamespace:
    return SimpleNamespace(
        stream=lambda *_args: AssistantMessageEventStream(),
        stream_simple=lambda *_args: AssistantMessageEventStream(),
    )


def _no_auth() -> ProviderAuth:
    async def resolve(_ctx, _credential, _cancel):
        return AuthResult(auth=ModelAuth())

    return ProviderAuth(api_key=ApiKeyAuth(name="Test", resolve=resolve))


def make_provider(*, id: str, models=None, env_var: str | None = None, calls: list | None = None, images=None):
    async def generate_images(model, _context, options=None):
        if calls is not None:
            calls.append({"model": model, "options": options})
        return ok_result(model)

    async def resolve(ctx, _credential, _cancel):
        key = await ctx.env(env_var) if env_var else None
        if env_var and not key:
            return None
        return AuthResult(auth=ModelAuth(api_key=key))

    image_apis = list((images or {"test-images": True}).keys())
    return create_provider(
        id=id,
        auth=ProviderAuth(api_key=ApiKeyAuth(name="Test key", resolve=resolve)),
        models=models if models is not None else [image_model(id, "model-a")],
        api={"test-chat": _chat_streams()},
        images={api: SimpleNamespace(generate_images=generate_images) for api in image_apis},
    )


CONTEXT = ImagesContext(input=[TextContent(text="a red circle")])


# --- model discriminants ---------------------------------------------------


def test_treats_models_without_a_type_as_chat_models():
    chat = chat_model("p", "c")
    image = image_model("p", "i")

    assert chat.type is None
    assert get_model_type(chat) == "chat"
    assert get_model_type(replace(chat, type="chat")) == "chat"
    assert get_model_type(image) == "image"
    assert is_model_type(chat, "chat") is True
    assert is_model_type(chat, "image") is False
    assert is_model_type(image, "image") is True

    # has_api never matches an image model, even on an equal api string
    assert has_api(replace(image, api="test-chat"), "test-chat") is False
    assert has_api(chat, "test-chat") is True


# --- Models with image models ----------------------------------------------


@pytest.mark.tonio
async def test_lists_models_without_a_type_as_chat_models_at_create_provider_boundaries():
    async def fetch_models(_context):
        return [chat_model("legacy", "dynamic")]

    provider = create_provider(
        id="legacy",
        auth=_no_auth(),
        models=[chat_model("legacy", "static"), image_model("legacy", "static")],
        fetch_models=fetch_models,
        api=_chat_streams(),
    )
    models = create_models()
    models.set_provider(provider)

    assert [model.id for model in provider.get_models()] == ["static"]
    await models.refresh(ModelsRefreshOptions(providers=[provider.id]))
    assert [model.id for model in provider.get_models()] == ["static", "dynamic"]
    assert [model.type for model in provider.get_models()] == [None, None]
    assert [model.id for model in models.get_models_of_type("image", "legacy")] == ["static"]


def test_lists_chat_image_and_all_models_through_typed_accessors():
    models = create_models()
    models.set_provider(
        make_provider(id="p1", models=[chat_model("p1", "c1"), image_model("p1", "i1"), image_model("p1", "i2")])
    )
    models.set_provider(make_provider(id="p2", models=[image_model("p2", "i3")]))

    assert [m.id for m in models.get_models()] == ["c1"]
    assert [m.id for m in models.get_models_of_type("chat")] == ["c1"]
    assert [m.id for m in models.get_models_of_type("image")] == ["i1", "i2", "i3"]
    assert [m.id for m in models.get_models_of_type("image", "p1")] == ["i1", "i2"]
    assert models.get_models_of_type("classifier") == []
    assert [m.id for m in models.get_all_models()] == ["c1", "i1", "i2", "i3"]

    assert models.get_model("p1", "c1").id == "c1"
    assert models.get_model("p1", "i1") is None
    assert models.get_model_of_type("chat", "p1", "c1").id == "c1"
    assert models.get_model_of_type("image", "p1", "i1").id == "i1"
    assert models.get_model_of_type("image", "p1", "c1") is None


@pytest.mark.tonio
async def test_splits_available_models_by_type():
    models = create_models(auth_context=_FakeAuthContext({"KEY": "k"}))
    models.set_provider(make_provider(id="p1", env_var="KEY", models=[chat_model("p1", "c1"), image_model("p1", "i1")]))
    models.set_provider(make_provider(id="p2", env_var="MISSING", models=[image_model("p2", "i2")]))

    assert [m.id for m in await models.get_available()] == ["c1"]
    assert [m.id for m in await models.get_available_of_type("chat")] == ["c1"]
    assert [m.id for m in await models.get_available_of_type("image")] == ["i1"]
    assert [m.id for m in await models.get_all_available()] == ["c1", "i1"]


@pytest.mark.tonio
async def test_resolves_auth_through_the_provider_and_merges_it_into_image_requests_explicit_options_win():
    calls: list = []
    models = create_models(auth_context=_FakeAuthContext({"TEST_KEY": "env-key"}))
    models.set_provider(make_provider(id="p1", env_var="TEST_KEY", calls=calls))
    model = models.get_model_of_type("image", "p1", "model-a")

    assert (await models.get_auth(model)).auth.api_key == "env-key"
    assert (await models.get_auth(model.provider)).auth.api_key == "env-key"

    result = await models.generate_images(model, CONTEXT)
    assert result.stop_reason == "stop"
    assert calls[0]["options"].api_key == "env-key"

    await models.generate_images(model, CONTEXT, ImagesOptions(api_key="explicit"))
    assert calls[1]["options"].api_key == "explicit"


@pytest.mark.tonio
async def test_merges_provider_resolved_env_and_applies_header_transforms():
    calls: list = []

    async def resolve(_ctx, _credential, _cancel):
        return AuthResult(
            auth=ModelAuth(api_key="provider-key", headers={"x-base": "1"}),
            env={"PROVIDER_ONLY": "provider", "SHARED": "provider"},
        )

    async def generate_images(model, _context, options=None):
        calls.append({"model": model, "options": options})
        return ok_result(model)

    models = create_models()
    models.set_provider(
        create_provider(
            id="p1",
            auth=ProviderAuth(api_key=ApiKeyAuth(name="Test key", resolve=resolve)),
            models=[image_model("p1", "model-a")],
            images={"test-images": SimpleNamespace(generate_images=generate_images)},
        )
    )
    model = models.get_model_of_type("image", "p1", "model-a")

    async def transform_headers(headers):
        return {**headers, "x-extra": "2"}

    await models.generate_images(
        model,
        CONTEXT,
        ImagesOptions(
            api_key="request-key",
            env={"REQUEST_ONLY": "request", "SHARED": "request"},
            transform_headers=transform_headers,
        ),
    )

    assert calls[0]["options"].api_key == "request-key"
    assert calls[0]["options"].env == {
        "PROVIDER_ONLY": "provider",
        "REQUEST_ONLY": "request",
        "SHARED": "request",
    }
    assert calls[0]["options"].headers == {"x-base": "1", "x-extra": "2"}


@pytest.mark.tonio
async def test_returns_error_results_instead_of_raising():
    models = create_models(auth_context=_FakeAuthContext({}))

    ghost = await models.generate_images(image_model("ghost", "m"), CONTEXT)
    assert ghost.stop_reason == "error"
    assert "Unknown provider: ghost" in ghost.error_message

    # Unconfigured auth is an error, matching stream().
    calls: list = []
    models.set_provider(make_provider(id="p1", env_var="MISSING", calls=calls))
    model = models.get_model_of_type("image", "p1", "model-a")
    assert await models.get_auth(model) is None
    unconfigured = await models.generate_images(model, CONTEXT)
    assert unconfigured.stop_reason == "error"
    assert "not configured" in unconfigured.error_message
    assert calls == []

    cancel = CancelToken()
    cancel.cancel()
    cancelled = await models.generate_images(model, CONTEXT, ImagesOptions(cancel=cancel))
    assert cancelled.stop_reason == "aborted"
    assert calls == []

    # A provider without any images implementation rejects image models it lists.
    models.set_provider(
        create_provider(id="chat-only", auth=_no_auth(), models=[image_model("chat-only", "i")], api=_chat_streams())
    )
    unsupported = await models.generate_images(models.get_model_of_type("image", "chat-only", "i"), CONTEXT)
    assert unsupported.stop_reason == "error"
    assert "does not support image generation" in unsupported.error_message

    # An images map without the model's api yields a provider error result.
    models.set_provider(
        make_provider(id="wrong-api", models=[image_model("wrong-api", "i")], images={"other-images": True})
    )
    missing_api = await models.generate_images(models.get_model_of_type("image", "wrong-api", "i"), CONTEXT)
    assert missing_api.stop_reason == "error"
    assert 'no image generation implementation for "test-images"' in missing_api.error_message


@pytest.mark.tonio
async def test_rejects_chat_models_at_the_image_entry_point_at_runtime():
    models = create_models()
    chat = chat_model("p1", "chat")
    models.set_provider(make_provider(id="p1", models=[chat], images={"test-chat": True}))

    result = await models.generate_images(chat, CONTEXT)
    assert result.stop_reason == "error"
    assert "is not an image model" in result.error_message


@pytest.mark.tonio
async def test_rejects_image_models_at_the_stream_entry_points_at_runtime():
    models = create_models()
    models.set_provider(make_provider(id="p1"))
    image = models.get_model_of_type("image", "p1", "model-a")

    result = await models.stream_simple(image, Context(messages=[]), SimpleStreamOptions()).result()
    assert result.stop_reason == "error"
    assert "is not a chat model" in result.error_message


@pytest.mark.parametrize(
    "implementations",
    [{}, {"api": {}}, {"images": {}}, {"classifiers": {}}],
    ids=["none", "empty api", "empty images", "empty classifiers"],
)
def test_requires_at_least_one_concrete_operation_implementation(implementations):
    with pytest.raises(Exception, match='at least one of "api", "images", or "classifiers"'):
        create_provider(id="empty", auth=_no_auth(), models=[], **implementations)


@pytest.mark.tonio
async def test_supports_dynamic_providers_listing_image_models_via_refresh():
    fetches: list = []
    models_store = InMemoryModelsStore()
    models = create_models(models_store=models_store)

    async def fetch_models(_context):
        fetches.append(1)
        return [image_model("dyn", "listed"), chat_model("dyn", "chat")]

    async def generate_images(model, _context, options=None):
        return ok_result(model)

    models.set_provider(
        create_provider(
            id="dyn",
            auth=_no_auth(),
            models=[],
            fetch_models=fetch_models,
            images={"test-images": SimpleNamespace(generate_images=generate_images)},
        )
    )

    assert models.get_all_models("dyn") == []
    result = await models.refresh(ModelsRefreshOptions(providers=["dyn"]))
    assert result.errors == {}
    assert len(fetches) == 1
    assert models.get_model_of_type("image", "dyn", "listed") is not None
    assert models.get_model("dyn", "chat") is not None
    stored = await models_store.read("dyn")
    assert [model.id for model in stored.models] == ["listed", "chat"]


def test_keeps_existing_built_in_model_reads_chat_only():
    chat = get_builtin_models("openrouter")
    images = get_builtin_image_models("openrouter")
    all_models = get_all_builtin_models("openrouter")

    assert all(is_model_type(model, "chat") for model in chat)
    assert all(is_model_type(model, "image") for model in images)
    assert any(is_model_type(model, "image") for model in all_models)
    assert all(model.context_window > 0 for model in chat)
    assert len(chat) + len(images) + len(get_builtin_classifier_models("openrouter")) == len(all_models)
    assert get_builtin_image_model("openrouter", "black-forest-labs/flux.2-pro").type == "image"


@pytest.mark.tonio
async def test_builtin_models_exposes_openrouter_image_models_under_the_openrouter_provider():
    models = builtin_models(auth_context=_FakeAuthContext({"OPENROUTER_API_KEY": "or-key"}))
    provider = models.get_provider("openrouter")
    images = models.get_models_of_type("image", "openrouter")
    assert len(images) > 0
    assert all(is_model_type(model, "chat") for model in provider.get_models())
    assert any(is_model_type(model, "image") for model in provider.get_all_models())
    assert all(m.type == "image" and m.api == "openrouter-images" for m in images)
    assert all(m.provider == "openrouter" for m in models.get_models_of_type("image"))

    # One upstream id can expose separate chat and image operations.
    chat = models.get_model("openrouter", "google/gemini-3-pro-image")
    image = models.get_model_of_type("image", "openrouter", "google/gemini-3-pro-image")
    assert chat.api == "openai-completions"
    assert image.api == "openrouter-images"

    # One credential covers both.
    assert (await models.get_auth(images[0])).auth.api_key == "or-key"
    assert (await models.get_auth(chat)).auth.api_key == "or-key"
    assert provider.generate_images is not None
