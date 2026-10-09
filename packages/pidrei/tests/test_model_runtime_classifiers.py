"""Mirror of pi coding-agent test/model-runtime-classifiers.test.ts.

pi passes `options.fetch`; here the System One POST is stubbed at
`classifier_shared._ClassifierClient`.
"""

import contextlib
import json

import pytest

from pidrei.core.auth_storage import AuthStorage
from pidrei.core.model_runtime import ModelRuntime
from pidrei_ai.api import classifier_shared
from pidrei_ai.models_store import InMemoryModelsStore
from pidrei_ai.types import ClassifierBoolAnswer, ClassifierBoolQuestion, ClassifierContext


CONTEXT = ClassifierContext(
    state={"text": "Looks good"},
    questions={
        "approved": ClassifierBoolQuestion(
            instructions="Does this express approval?", criteria={"true": "Approval", "false": "No approval"}
        )
    },
)


@contextlib.contextmanager
def _stub_system_one(body):
    headers_seen: list[dict[str, str]] = []

    class _StubClient:
        def __init__(self, env=None, fetch=None):
            pass

        async def post(self, _url, _payload, headers, _cancel):
            headers_seen.append(headers)
            return 200, {}, json.dumps(body)

    original = classifier_shared._ClassifierClient
    classifier_shared._ClassifierClient = _StubClient
    try:
        yield headers_seen
    finally:
        classifier_shared._ClassifierClient = original


@pytest.mark.tonio
async def test_lists_jev_separately_and_classifies_with_runtime_resolved_auth():
    runtime = await ModelRuntime(
        credentials=AuthStorage.in_memory(),
        models_store=InMemoryModelsStore(),
        models_path=None,
        allow_model_network=False,
    )
    jev = runtime.get_model_of_type("classifier", "typesafe", "jev-latest")
    assert jev is not None
    assert jev.type == "classifier"
    assert runtime.get_model("typesafe", "jev-latest") is None

    unconfigured = await runtime.classify(jev, CONTEXT)
    assert unconfigured.stop_reason == "error"
    assert "not configured" in unconfigured.error_message

    await runtime.set_runtime_api_key("typesafe", "sk-typesafe")
    assert await runtime.get_available_of_type("classifier", "typesafe") == [jev]
    with _stub_system_one({"answers": {"approved": {"type": "noul", "noul": 0.8}}}) as headers_seen:
        result = await runtime.classify(jev, CONTEXT)

    assert headers_seen[0]["authorization"] == "Bearer sk-typesafe"
    assert result.answers["approved"] == ClassifierBoolAnswer(probability=0.8)
