"""Mirror of pi coding-agent test/model-runtime-classifiers.test.ts.

pi passes `options.fetch`; here the System One POST is stubbed at
`classifier_shared._ClassifierClient`.
"""

import contextlib
import json
from dataclasses import dataclass, replace

import pytest

from pidrei.core.auth_storage import AuthStorage
from pidrei.core.model_runtime import ModelRuntime
from pidrei_ai.api import classifier_shared
from pidrei_ai.models_store import InMemoryModelsStore
from pidrei_ai.types import ClassifierBoolAnswer, ClassifierBoolQuestion, ClassifierContext, ImageContent


CONTEXT = ClassifierContext(
    state={"text": "Looks good"},
    questions={
        "approved": ClassifierBoolQuestion(
            instructions="Does this express approval?", criteria={"true": "Approval", "false": "No approval"}
        )
    },
)

IMAGE = ImageContent(data="cG5n", mime_type="image/png")


@dataclass(slots=True)
class _Request:
    url: str
    body: dict
    # Names lower-cased: pi reads them through `new Headers(...)`, which folds case.
    headers: dict[str, str]


@contextlib.contextmanager
def _stub_system_one(body):
    requests: list[_Request] = []

    class _StubClient:
        def __init__(self, env=None, fetch=None):
            pass

        async def post(self, url, payload, headers, _cancel):
            requests.append(
                _Request(
                    url=url,
                    body=json.loads(json.dumps(payload)),
                    headers={name.lower(): value for name, value in headers.items()},
                )
            )
            return 200, {}, json.dumps(body)

    original = classifier_shared._ClassifierClient
    classifier_shared._ClassifierClient = _StubClient
    try:
        yield requests
    finally:
        classifier_shared._ClassifierClient = original


async def _create_runtime(tmp_path=None, models_json: dict | None = None) -> ModelRuntime:
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


async def _capture_image_request(runtime: ModelRuntime, provider: str, id: str) -> _Request:
    """Classifies one image and returns the request the service received."""
    await runtime.set_runtime_api_key(provider, "secret")
    model = runtime.get_model_of_type("classifier", provider, id)
    with _stub_system_one({"answers": {"approved": {"type": "noul", "noul": 0.8}}}) as requests:
        result = await runtime.classify(model, replace(CONTEXT, images=[IMAGE]))
    assert result.stop_reason == "stop"
    assert len(requests) == 1
    return requests[0]


@pytest.mark.tonio
async def test_lists_jev_separately_and_classifies_with_runtime_resolved_auth():
    runtime = await _create_runtime()
    jev = runtime.get_model_of_type("classifier", "typesafe", "jev-latest")
    assert jev is not None
    assert jev.type == "classifier"
    assert runtime.get_model("typesafe", "jev-latest") is None

    unconfigured = await runtime.classify(jev, CONTEXT)
    assert unconfigured.stop_reason == "error"
    assert "not configured" in unconfigured.error_message

    await runtime.set_runtime_api_key("typesafe", "sk-typesafe")
    assert await runtime.get_available_of_type("classifier", "typesafe") == [jev]
    with _stub_system_one({"answers": {"approved": {"type": "noul", "noul": 0.8}}}) as requests:
        result = await runtime.classify(jev, CONTEXT)

    assert requests[0].headers["authorization"] == "Bearer sk-typesafe"
    assert result.answers["approved"] == ClassifierBoolAnswer(probability=0.8)


OPEN_ROUTER_IMAGE_STATE = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,cG5n"}}, CONTEXT.state]


@pytest.mark.tonio
async def test_keeps_openrouters_image_form_when_models_json_routes_openrouter_through_a_proxy(tmp_path):
    runtime = await _create_runtime(
        tmp_path, {"providers": {"openrouter": {"baseUrl": "https://proxy.example.com/v1"}}}
    )

    request = await _capture_image_request(runtime, "openrouter", "typesafe/jev-1.14")

    # OpenRouter answers TypeSafe `assets` with 200 but drops the images.
    assert request.url == "https://proxy.example.com/v1/systemone"
    assert request.body["state"] == OPEN_ROUTER_IMAGE_STATE
    assert "assets" not in request.body
    assert "typesafe-danger-zone-images" not in request.headers


@pytest.mark.tonio
async def test_applies_provider_compat_from_models_json_to_classifier_models(tmp_path):
    runtime = await _create_runtime(
        tmp_path,
        {
            "providers": {
                "typesafe": {
                    "baseUrl": "https://proxy.example.com/v1",
                    "compat": {"imageInput": "state-image-urls", "sendImageOptInHeader": False},
                }
            }
        },
    )

    request = await _capture_image_request(runtime, "typesafe", "jev-latest")

    assert request.url == "https://proxy.example.com/v1/systemone"
    assert request.body["state"] == OPEN_ROUTER_IMAGE_STATE
    assert "typesafe-danger-zone-images" not in request.headers
