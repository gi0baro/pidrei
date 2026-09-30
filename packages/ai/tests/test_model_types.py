"""Mirror of pi's model-types.test.ts.

pi's "built-in catalog getters return model shapes that can be reassigned
within one api" case is a TypeScript compile-time regression check with no
runtime behavior, so it is not mirrored.
"""

from dataclasses import replace

import pytest

from pidrei_ai.models_store import InMemoryModelsStore, ModelsStoreEntry
from pidrei_ai.providers.faux import faux_assistant_message, faux_provider
from pidrei_ai.registry import (
    ModelsRefreshOptions,
    create_models,
    create_provider,
    get_model_type,
    has_api,
    is_model_type,
    models_are_equal,
)
from pidrei_ai.types import Context, ImageModel, Model, ModelCost, UserMessage
from tests.test_images_models import _chat_streams, _no_auth


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


class _HandwrittenProvider:
    """A provider object written by hand, without the optional members
    (pi: an object literal omitting `getAllModels` and friends)."""

    def __init__(self, faux):
        self.id = "handwritten"
        self.name = "Handwritten"
        self.base_url = None
        self.headers = None
        self.auth = faux.provider.auth
        self.filter_models = None
        self.get_all_models = None
        self.filter_all_models = None
        self.generate_images = None
        self.classify = None
        self._faux = faux
        self.stream = faux.provider.stream
        self.stream_simple = faux.provider.stream_simple
        self.supports_fetch_deferred = False
        self.supports_cancel_deferred = False
        self.has_dynamic_models = False

    def get_models(self) -> list[Model]:
        return self._faux.models


@pytest.mark.tonio
async def test_chat_models_without_a_type_work_through_a_handwritten_provider_without_get_all_models():
    faux = faux_provider(provider="handwritten")
    models = create_models()
    models.set_provider(_HandwrittenProvider(faux))

    model = models.get_model("handwritten", faux.models[0].id)
    assert model is not None
    assert model.type is None
    assert get_model_type(model) == "chat"
    assert has_api(model, model.api) is True
    assert models_are_equal(model, replace(model, type="chat")) is True
    assert models_are_equal(model, image_model(model.provider, model.id)) is False
    assert models.get_models_of_type("chat", "handwritten") == faux.models
    assert models.get_all_models("handwritten") == faux.models
    assert models.get_models_of_type("image", "handwritten") == []

    faux.set_responses([faux_assistant_message("hi")])
    result = await models.complete(model, Context(messages=[UserMessage(content="hi", timestamp=0)]))
    assert result.stop_reason == "stop"


def test_chat_models_without_a_type_narrow_mixed_lists_with_is_model_type():
    mixed = [chat_model("p", "c"), replace(chat_model("p", "typed"), type="chat"), image_model("p", "i")]
    assert [model.id for model in mixed if is_model_type(model, "chat")] == ["c", "typed"]
    assert [model.id for model in mixed if is_model_type(model, "image")] == ["i"]
    assert [model for model in mixed if is_model_type(model, "classifier")] == []


@pytest.mark.tonio
async def test_stored_and_fetched_models_of_unknown_types_are_dropped_instead_of_failing_the_refresh():
    models_store = InMemoryModelsStore()
    stored = ModelsStoreEntry(
        models=[
            chat_model("dyn", "stored-chat"),
            image_model("dyn", "stored-image"),
            replace(chat_model("dyn", "future-embedding"), type="embedding"),
            replace(image_model("dyn", "future-video"), type="video"),
        ]
    )
    await models_store.write("dyn", stored)

    fetched: list = []

    async def fetch_models(_context):
        return fetched

    models = create_models(models_store=models_store)
    models.set_provider(
        create_provider(id="dyn", auth=_no_auth(), models=[], fetch_models=fetch_models, api=_chat_streams())
    )

    restored = await models.refresh(ModelsRefreshOptions(providers=["dyn"], allow_network=False))
    assert restored.errors == {}
    assert [model.id for model in models.get_all_models("dyn")] == ["stored-chat", "stored-image"]

    fetched = [chat_model("dyn", "fetched-chat"), replace(image_model("dyn", "fetched-video"), type="video")]
    refreshed = await models.refresh(ModelsRefreshOptions(providers=["dyn"]))
    assert refreshed.errors == {}
    assert [model.id for model in models.get_all_models("dyn")] == ["fetched-chat"]
    assert [model.id for model in (await models_store.read("dyn")).models] == ["fetched-chat"]
