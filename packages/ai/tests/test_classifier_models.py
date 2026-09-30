"""Mirror of pi's classifier-models.test.ts.

pi injects `options.fetch`; here the one POST is stubbed at
`system_one_shared._SystemOneClient` (tests/system_one_helpers.py).
"""

from types import SimpleNamespace

import pytest

from pidrei_ai.auth.types import ApiKeyAuth, AuthResult, ModelAuth, ProviderAuth
from pidrei_ai.providers.all import (
    builtin_models,
    get_all_builtin_models,
    get_builtin_classifier_model,
    get_builtin_classifier_models,
)
from pidrei_ai.registry import create_models, create_provider, get_model_type
from pidrei_ai.types import (
    ClassifierBoolAnswer,
    ClassifierBoolQuestion,
    ClassifierContext,
    ClassifierModel,
    ClassifierOptions,
    ClassifierResult,
    Model,
    ModelCost,
)
from pidrei_ai.utils.event_stream import AssistantMessageEventStream
from tests.system_one_helpers import respond_json, stub_system_one


def classifier_model(provider: str, id: str) -> ClassifierModel:
    return ClassifierModel(
        id=id,
        name=id,
        api="test-classifier",
        provider=provider,
        base_url="https://example.test/v1",
        input=["text"],
        cost=ModelCost(),
        context_window=1000,
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


def _no_auth() -> ProviderAuth:
    async def resolve(_ctx, _credential, _cancel):
        return AuthResult(auth=ModelAuth())

    return ProviderAuth(api_key=ApiKeyAuth(name="Test", resolve=resolve))


def _chat_streams() -> SimpleNamespace:
    return SimpleNamespace(
        stream=lambda *_args: AssistantMessageEventStream(),
        stream_simple=lambda *_args: AssistantMessageEventStream(),
    )


CONTEXT = ClassifierContext(
    state={"text": "yes"},
    questions={
        "approved": ClassifierBoolQuestion(
            instructions="Does this express approval?", criteria={"true": "Approval", "false": "No approval"}
        )
    },
)


@pytest.mark.tonio
async def test_keeps_chat_and_classifier_entries_with_the_same_provider_and_id_separate():
    chat = chat_model("test", "shared")
    classifier = classifier_model("test", "shared")

    async def classify(model, _context, _options=None):
        return ClassifierResult(
            api=model.api,
            provider=model.provider,
            model=model.id,
            answers={"approved": ClassifierBoolAnswer(probability=0.9)},
            stop_reason="stop",
            timestamp=0,
        )

    provider = create_provider(
        id="test",
        auth=_no_auth(),
        models=[chat, classifier],
        api={"test-chat": _chat_streams()},
        classifiers={"test-classifier": SimpleNamespace(classify=classify)},
    )
    models = create_models()
    models.set_provider(provider)

    listed_chat = models.get_model("test", "shared")
    assert get_model_type(listed_chat) == "chat"
    assert models.get_model_of_type("classifier", "test", "shared").type == "classifier"
    assert models.get_models_of_type("classifier") == [classifier]
    assert len(models.get_all_models()) == 2
    assert await models.get_available_of_type("classifier") == [classifier]
    assert (await models.classify(classifier, CONTEXT)).answers["approved"] == ClassifierBoolAnswer(probability=0.9)


@pytest.mark.tonio
async def test_rejects_chat_models_at_the_classifier_entry_point_at_runtime():
    chat = chat_model("test", "chat")
    models = create_models()
    models.set_provider(create_provider(id="test", auth=_no_auth(), models=[chat], api=_chat_streams()))

    result = await models.classify(chat, CONTEXT)
    assert result.stop_reason == "error"
    assert "is not a classifier model" in result.error_message


def test_exposes_jev_only_through_classifier_catalog_accessors():
    jev = get_builtin_classifier_model("typesafe", "jev-latest")
    assert jev is not None
    assert (jev.type, jev.api, jev.provider, jev.context_window) == (
        "classifier",
        "typesafe-system-one",
        "typesafe",
        64000,
    )
    assert get_builtin_classifier_models("typesafe") == [jev]
    assert get_all_builtin_models("typesafe") == [jev]

    models = builtin_models()
    assert models.get_model("typesafe", "jev-latest") is None
    assert models.get_model_of_type("classifier", "typesafe", "jev-latest") == jev


@pytest.mark.tonio
@pytest.mark.parametrize(
    ("provider", "id", "url"),
    [
        ("vercel-ai-gateway", "typesafe-ai/jev", "https://ai-gateway.vercel.sh/typesafe/v1/systemone"),
        ("opencode", "jev-1.13", "https://opencode.ai/zen/v1/systemone"),
        ("opencode", "jev-1.13-free", "https://opencode.ai/zen/v1/systemone"),
    ],
)
async def test_routes_jev_to_its_typesafe_compatible_endpoint(provider, id, url):
    models = builtin_models()
    jev = models.get_model_of_type("classifier", provider, id)
    assert jev is not None, f"missing {provider} Jev model"
    assert (jev.api, jev.context_window) == ("typesafe-system-one", 32000)
    assert models.get_model(provider, id) is None

    with stub_system_one(
        respond_json({"model": id, "answers": {"approved": {"type": "noul", "noul": 0.8}}})
    ) as requests:
        result = await models.classify(jev, CONTEXT, ClassifierOptions(api_key="secret"))

    assert [(request.url, request.headers["authorization"]) for request in requests] == [(url, "Bearer secret")]
    assert requests[0].payload["model"] == id
    assert requests[0].payload["state"] == CONTEXT.state
    assert result.stop_reason == "stop"
    assert result.answers["approved"] == ClassifierBoolAnswer(probability=0.8)


def test_routes_openrouter_classifier_models_through_the_system_one_api():
    models = builtin_models()
    for model in get_builtin_classifier_models("openrouter"):
        assert (model.api, model.base_url) == ("typesafe-system-one", "https://openrouter.ai/api/v1")
        assert models.get_model("openrouter", model.id) is None
        assert models.get_model_of_type("classifier", "openrouter", model.id) == model
