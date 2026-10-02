"""Mirror of pi coding-agent test/model-runtime-images.test.ts.

Extension registrations are pi-shaped camelCase dicts, as elsewhere in
pidrei's provider composer; `images`/`classifiers` implementations are
objects with `generate_images`/`classify`.
"""

import json
from types import SimpleNamespace

import pytest

from pidrei.core.auth_storage import AuthStorage
from pidrei.core.model_runtime import ModelRuntime
from pidrei_ai.auth.types import ApiKeyAuth, AuthResult, ModelAuth, ProviderAuth
from pidrei_ai.models_store import InMemoryModelsStore
from pidrei_ai.registry import create_provider, get_model_type, is_model_type
from pidrei_ai.types import (
    AssistantImages,
    ClassifierBoolAnswer,
    ClassifierBoolQuestion,
    ClassifierContext,
    ClassifierModel,
    ClassifierResult,
    Context,
    DeferredHandle,
    ImageContent,
    ImageModel,
    ImagesContext,
    ImagesOptions,
    Model,
    ModelCost,
    TextContent,
)
from pidrei_utils.cancel import CancelToken


def image_model(provider: str, id: str) -> ImageModel:
    return ImageModel(
        id=id,
        name=id,
        api="test-images",
        provider=provider,
        base_url="https://images.test/v1",
        input=["text"],
        output=["image"],
        cost=ModelCost(),
    )


def classifier_model(provider: str, id: str) -> ClassifierModel:
    return ClassifierModel(
        id=id,
        name=id,
        api="test-classifier",
        provider=provider,
        base_url="https://classifier.test/v1",
        input=["text"],
        cost=ModelCost(),
        context_window=1000,
    )


def image_definition(id: str, **overrides) -> dict:
    return {
        "type": "image",
        "id": id,
        "name": id,
        "api": "test-images",
        "baseUrl": "https://images.test/v1",
        "input": ["text"],
        "output": ["image"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        **overrides,
    }


def classifier_definition(id: str, **overrides) -> dict:
    return {
        "type": "classifier",
        "id": id,
        "name": id,
        "api": "test-classifier",
        "baseUrl": "https://classifier.test/v1",
        "input": ["text"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 1000,
        **overrides,
    }


CONTEXT = ImagesContext(input=[TextContent(text="a red circle")])
CLASSIFIER_CONTEXT = ClassifierContext(
    state={"text": "Looks good"},
    questions={
        "approved": ClassifierBoolQuestion(
            instructions="Does this express approval?", criteria={"true": "Approval", "false": "No approval"}
        )
    },
)


def ok_image_result(model: ImageModel) -> AssistantImages:
    return AssistantImages(
        api=model.api,
        provider=model.provider,
        model=model.id,
        output=[ImageContent(data="aGk=", mime_type="image/png")],
        stop_reason="stop",
        timestamp=0,
    )


def _stored_key_auth(name: str) -> ProviderAuth:
    async def resolve(_ctx, credential, _cancel):
        if credential is not None and credential.key:
            return AuthResult(auth=ModelAuth(api_key=credential.key), source="stored")
        return None

    return ProviderAuth(api_key=ApiKeyAuth(name=name, resolve=resolve))


async def create_runtime(tmp_path=None, models_json: dict | None = None) -> ModelRuntime:
    models_path = None
    if models_json is not None:
        models_path = str(tmp_path / "models.json")
        (tmp_path / "models.json").write_text(json.dumps(models_json))
    return await ModelRuntime(
        credentials=AuthStorage.in_memory(),
        models_store=InMemoryModelsStore(),
        models_path=models_path,
        allow_model_network=False,
    )


@pytest.mark.tonio
async def test_lists_built_in_openrouter_image_models_separately_from_chat_models():
    runtime = await create_runtime()
    images = runtime.get_models_of_type("image", "openrouter")
    assert len(images) > 0
    assert all(is_model_type(model, "chat") for model in runtime.get_models("openrouter"))
    assert runtime.get_model_of_type("image", "openrouter", images[0].id) is images[0]
    assert runtime.get_model("openrouter", "google/gemini-3-pro-image").api == "openai-completions"
    assert runtime.get_model_of_type("image", "openrouter", "google/gemini-3-pro-image").type == "image"
    classifiers = runtime.get_models_of_type("classifier", "openrouter")
    assert len(runtime.get_all_models("openrouter")) == (
        len(runtime.get_models("openrouter")) + len(images) + len(classifiers)
    )


@pytest.mark.tonio
async def test_extension_model_lists_replace_undeclared_models_of_every_operation():
    runtime = await create_runtime()
    chat = Model(
        id="built-in-chat",
        name="Built-in chat",
        api="test-chat",
        provider="mixed",
        base_url="https://built-in.test/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(),
        context_window=1000,
        max_tokens=100,
    )

    async def resolve(_ctx, _credential, _cancel):
        return AuthResult(auth=ModelAuth())

    async def generate_images(model, _context, _options=None):
        return ok_image_result(model)

    runtime.register_native_provider(
        create_provider(
            id="mixed",
            auth=ProviderAuth(api_key=ApiKeyAuth(name="Mixed key", resolve=resolve)),
            models=[chat, image_model("mixed", "built-in-image"), classifier_model("mixed", "built-in-classifier")],
            images={"test-images": SimpleNamespace(generate_images=generate_images)},
        )
    )

    runtime.register_provider(
        "mixed",
        {
            "apiKey": "extension-secret",
            "models": [
                {
                    "id": "extension-chat",
                    "name": "Extension chat",
                    "api": "test-chat",
                    "baseUrl": "https://chat-proxy.test/v1",
                    "reasoning": False,
                    "input": ["text"],
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    "contextWindow": 1000,
                    "maxTokens": 100,
                }
            ],
        },
    )

    assert [(get_model_type(model), model.id) for model in runtime.get_all_models("mixed")] == [
        ("chat", "extension-chat")
    ]
    assert runtime.get_model("mixed", "extension-chat").base_url == "https://chat-proxy.test/v1"
    assert runtime.get_models_of_type("image", "mixed") == []
    assert runtime.get_models_of_type("classifier", "mixed") == []


@pytest.mark.tonio
async def test_registers_extension_image_and_classifier_models_with_their_implementations():
    runtime = await create_runtime()
    observed: list = []

    async def generate_images(model, _context, options=None):
        observed.append({"api_key": options.api_key, "headers": options.headers})
        return ok_image_result(model)

    async def classify(model, _context, options=None):
        observed.append({"api_key": options.api_key, "headers": options.headers})
        return ClassifierResult(
            api=model.api,
            provider=model.provider,
            model=model.id,
            answers={"approved": ClassifierBoolAnswer(probability=0.9)},
            stop_reason="stop",
            timestamp=0,
        )

    runtime.register_provider(
        "extension-operations",
        {
            "apiKey": "extension-secret",
            "models": [
                image_definition("shared", headers={"X-Operation": "image"}),
                classifier_definition("shared", headers={"X-Operation": "classifier"}),
            ],
            "images": {"test-images": SimpleNamespace(generate_images=generate_images)},
            "classifiers": {"test-classifier": SimpleNamespace(classify=classify)},
        },
    )

    image = runtime.get_model_of_type("image", "extension-operations", "shared")
    classifier = runtime.get_model_of_type("classifier", "extension-operations", "shared")
    assert (await runtime.generate_images(image, CONTEXT)).stop_reason == "stop"
    assert (await runtime.classify(classifier, CLASSIFIER_CONTEXT)).stop_reason == "stop"
    assert observed == [
        {"api_key": "extension-secret", "headers": {"X-Operation": "image"}},
        {"api_key": "extension-secret", "headers": {"X-Operation": "classifier"}},
    ]


@pytest.mark.tonio
async def test_generates_images_through_a_native_provider_with_runtime_resolved_auth():
    runtime = await create_runtime()
    calls: list = []

    async def generate_images(model, _context, options=None):
        calls.append({"model": model, "options": options})
        return ok_image_result(model)

    runtime.register_native_provider(
        create_provider(
            id="pixels",
            auth=_stored_key_auth("Pixels key"),
            models=[image_model("pixels", "flux")],
            images={"test-images": SimpleNamespace(generate_images=generate_images)},
        )
    )
    model = runtime.get_model_of_type("image", "pixels", "flux")

    unconfigured = await runtime.generate_images(model, CONTEXT)
    assert unconfigured.stop_reason == "error"
    assert "not configured" in unconfigured.error_message
    assert calls == []

    cancel = CancelToken()
    cancel.cancel()
    cancelled = await runtime.generate_images(model, CONTEXT, ImagesOptions(cancel=cancel))
    assert cancelled.stop_reason == "aborted"
    assert calls == []

    await runtime.set_runtime_api_key("pixels", "sk-pixels")
    assert [entry.id for entry in await runtime.get_available_of_type("image", "pixels")] == ["flux"]
    result = await runtime.generate_images(model, CONTEXT)
    assert result.stop_reason == "stop"
    assert calls[0]["options"].api_key == "sk-pixels"


class _RejectingStreams:
    """Chat adapter that must never be reached with an image model."""

    def __init__(self):
        self.dispatches = 0

    def _reject(self, *_args, **_kwargs):
        self.dispatches += 1
        raise RuntimeError("image reached chat adapter")

    stream = stream_simple = fetch_deferred = _reject

    async def cancel_deferred(self, *args, **kwargs):
        self._reject(*args, **kwargs)


@pytest.mark.tonio
async def test_rejects_image_models_at_every_chat_entry_point_before_provider_dispatch():
    runtime = await create_runtime()
    streams = _RejectingStreams()
    runtime.register_native_provider(
        create_provider(
            id="hybrid",
            auth=_stored_key_auth("Hybrid key"),
            models=[image_model("hybrid", "shared")],
            api=streams,
        )
    )
    await runtime.set_runtime_api_key("hybrid", "sk-hybrid")
    image = runtime.get_model_of_type("image", "hybrid", "shared")
    chat_context = Context(messages=[])
    handle = DeferredHandle(provider="hybrid", model_id=image.id, api=image.api, id="response-1")

    results = [
        await runtime.stream(image, chat_context).result(),
        await runtime.complete(image, chat_context),
        await runtime.stream_simple(image, chat_context).result(),
        await runtime.complete_simple(image, chat_context),
        await runtime.stream_deferred(image, handle).result(),
        await runtime.fetch_deferred(image, handle),
    ]
    for result in results:
        assert result.stop_reason == "error"
        assert "is not a chat model" in result.error_message
    with pytest.raises(Exception, match="is not a chat model"):
        await runtime.cancel_deferred(image, handle)
    assert streams.dispatches == 0


@pytest.mark.tonio
async def test_keeps_image_generation_on_a_built_in_provider_composed_with_models_json_overrides(tmp_path):
    runtime = await create_runtime(
        tmp_path,
        {
            "providers": {
                "openrouter": {
                    "headers": {"X-Title": "pi"},
                    "modelOverrides": {
                        "openrouter/auto": {"name": "Auto (renamed)"},
                        "google/gemini-3-pro-image": {"headers": {"X-Chat-Only": "yes"}},
                    },
                }
            }
        },
    )
    provider = runtime.get_provider("openrouter")
    assert provider.generate_images is not None
    assert runtime.get_model("openrouter", "openrouter/auto").name == "Auto (renamed)"
    assert len(runtime.get_models_of_type("image", "openrouter")) > 0

    # Auth is provider-scoped: the same key serves chat and image models.
    await runtime.set_runtime_api_key("openrouter", "sk-or")
    chat = runtime.get_model("openrouter", "google/gemini-3-pro-image")
    image = runtime.get_model_of_type("image", "openrouter", "google/gemini-3-pro-image")
    chat_auth = await runtime.get_auth(chat)
    image_auth = await runtime.get_auth(image)
    assert image_auth.auth.api_key == "sk-or"
    assert {"X-Title": "pi", "X-Chat-Only": "yes"}.items() <= dict(chat_auth.auth.headers).items()
    assert image_auth.auth.headers == {"X-Title": "pi"}


@pytest.mark.tonio
async def test_does_not_add_image_generation_or_classification_to_composed_chat_only_providers(tmp_path):
    runtime = await create_runtime(tmp_path, {"providers": {"anthropic": {"headers": {"X-Title": "pi"}}}})
    provider = runtime.get_provider("anthropic")
    assert provider.generate_images is None
    assert provider.classify is None
    assert runtime.get_provider("openrouter").generate_images is not None
    assert runtime.get_provider("typesafe").classify is not None
