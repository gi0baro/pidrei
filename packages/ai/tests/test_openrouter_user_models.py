"""Mirror of pi ai test/openrouter-user-models.test.ts.

pi spies on the global `fetch`; the request here goes through the HTTP seam's
`default_fetch`, which the cases replace with a fake response.
"""

import json

import pytest

from pidrei_ai.auth.types import ApiKeyCredential
from pidrei_ai.providers.openrouter import personalize_openrouter_catalog
from pidrei_ai.registry import RefreshModelsContext
from pidrei_ai.types import ClassifierModel, ImageModel, Model, ModelCost
from pidrei_http import http
from pidrei_utils.cancel import CancelToken


def _zero_cost() -> ModelCost:
    return ModelCost(input=0, output=0, cache_read=0, cache_write=0)


def _chat_model(id: str) -> Model:
    return Model(
        id=id,
        name=id,
        api="openai-completions",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=1, output=2, cache_read=0.1, cache_write=0),
        context_window=200_000,
        max_tokens=50_000,
    )


IMAGE_MODEL = ImageModel(
    id="image",
    name="image",
    api="openrouter-images",
    provider="openrouter",
    base_url="https://openrouter.ai/api/v1",
    input=["text"],
    output=["image"],
    cost=_zero_cost(),
)

CLASSIFIER_MODEL = ClassifierModel(
    id="classifier",
    name="classifier",
    api="typesafe-system-one",
    provider="openrouter",
    base_url="https://openrouter.ai/api/v1",
    input=["text"],
    cost=_zero_cost(),
    context_window=64_000,
)


def _refresh_context() -> RefreshModelsContext:
    async def publish(_publication) -> bool:
        return True

    return RefreshModelsContext(
        credential=ApiKeyCredential(key="sk-or-test"),
        publish=publish,
        allow_network=True,
        cancel=CancelToken(),
    )


class _FakeResponse:
    def __init__(self, status_code: int, body: bytes):
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self._body = body

    async def read(self) -> bytes:
        return self._body


@pytest.fixture
def user_models(monkeypatch):
    """pi's `stubUserModels`: the next request answers with `(status, body)`; records the request."""
    requests = []
    reply: dict = {}

    async def fake_fetch(request, *, env=None):
        requests.append(request)
        return _FakeResponse(reply["status"], reply["body"])

    monkeypatch.setattr(http, "default_fetch", fake_fetch)

    def stub(status: int, body: bytes) -> list:
        reply.update(status=status, body=body)
        return requests

    return stub


@pytest.mark.tonio
async def test_keeps_only_the_keys_chat_models_with_the_keys_limits_and_prices(user_models):
    data = [
        {
            "id": "listed",
            "context_length": 1_000_000,
            "top_provider": {"context_length": 262_144, "max_completion_tokens": 32_768},
            "pricing": {"prompt": "0.000003", "completion": "0.000015"},
        },
        # A missing limit keeps the catalog value, so generator fixes for missing OpenRouter data survive.
        {
            "id": "no-limits",
            "context_length": 0,
            "top_provider": {"context_length": None, "max_completion_tokens": None},
        },
        # Router models report -1, meaning the price depends on the model they pick.
        {"id": "router", "pricing": {"prompt": "-1", "completion": "-1"}},
    ]
    requests = user_models(200, json.dumps({"data": data}).encode())
    catalog = [
        _chat_model("listed"),
        _chat_model("no-limits"),
        _chat_model("router"),
        _chat_model("blocked"),
        IMAGE_MODEL,
        CLASSIFIER_MODEL,
    ]

    result = await personalize_openrouter_catalog(catalog, _refresh_context())

    assert [model.id for model in result] == ["listed", "no-limits", "router", "image", "classifier"]
    assert result[0].context_window == 262_144
    assert result[0].max_tokens == 32_768
    assert result[0].cost == ModelCost(input=3, output=15, cache_read=0, cache_write=0)
    assert result[1].context_window == 200_000
    assert result[1].max_tokens == 50_000
    assert result[1].cost == _chat_model("x").cost
    assert result[2].cost == _zero_cost()
    # /models/user lists text models only, so image and classifier models are not filtered.
    assert result[3] is IMAGE_MODEL
    assert result[4] is CLASSIFIER_MODEL
    assert str(requests[0].url) == "https://openrouter.ai/api/v1/models/user"
    assert requests[0].headers["authorization"] == "Bearer sk-or-test"


@pytest.mark.tonio
@pytest.mark.parametrize("status", [401, 403, 404])
async def test_returns_the_catalog_unchanged_when_the_endpoint_rejects_the_key_or_is_missing(user_models, status):
    # For example a proxy configured in models.json that rejects the key or has no /models/user:
    # the user keeps the full list instead of a failing refresh.
    user_models(status, b"")
    catalog = [_chat_model("a")]

    assert await personalize_openrouter_catalog(catalog, _refresh_context()) is catalog
