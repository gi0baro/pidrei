"""Mirror of pi's cloudflare-workers-ai-system-one.test.ts.

pi injects `options.fetch`; here the one POST is stubbed at
`system_one_shared._SystemOneClient` (tests/system_one_helpers.py).
"""

import pytest

from pidrei_ai.providers.all import get_builtin_classifier_model
from pidrei_ai.providers.cloudflare_workers_ai import cloudflare_workers_ai_provider
from pidrei_ai.registry import create_models
from pidrei_ai.types import (
    ClassifierBoolAnswer,
    ClassifierBoolQuestion,
    ClassifierChoiceQuestion,
    ClassifierContext,
    ClassifierOptions,
)
from tests.system_one_helpers import respond_json, stub_system_one


CONTEXT = ClassifierContext(
    state={"message": "Help! My payouts have been failing for 3 days."},
    questions={
        "is_urgent": ClassifierBoolQuestion(
            instructions="Does this convey urgency?",
            criteria={"true": "Explicitly time-sensitive", "false": "No urgency expressed"},
        ),
        "department": ClassifierChoiceQuestion(
            instructions="Which team should handle this?", criteria={"billing": "Payments", "technical": "Bugs"}
        ),
    },
)

# Model output from https://developers.cloudflare.com/ai/models/typesafe/jev/
JEV_OUTPUT = {
    "model": "jev-1.13.0",
    "answers": {
        "is_urgent": {"type": "noul", "noul": 0.95},
        "department": {
            "type": "choice",
            "choice": "billing",
            "confidence": 0.8,
            "probabilities": {"billing": 0.87, "technical": 0.13},
        },
    },
    "usage": {"input_tokens": 426, "output_tokens": 73},
}


def rest_response(state: str, result=JEV_OUTPUT) -> dict:
    """REST envelope observed from the live /ai/run endpoint."""
    return {
        "result": {"state": state, "result": result, "gatewayMetadata": {"keySource": "Unified"}},
        "success": True,
        "errors": [],
        "messages": [],
    }


# Cloudflare-hosted output observed from a live /ai/run call, question ids renamed to match `CONTEXT`.
# The envelope carries the output directly, without a run record.
CLEF_OUTPUT = {
    "model": "clef",
    "answers": {
        "is_urgent": {"type": "noul", "noul": 0.9912},
        "department": {
            "type": "choice",
            "choice": "technical",
            "probabilities": {"billing": 0.1632, "technical": 0.8368},
            "confidence": 0.4538,
        },
    },
    "usage": {"input_tokens": 222, "output_tokens": 0},
}


def setup():
    models = create_models()
    models.set_provider(cloudflare_workers_ai_provider())
    jev = models.get_model_of_type("classifier", "cloudflare-workers-ai", "typesafe/jev")
    assert jev is not None, "missing Cloudflare Jev model"
    return models, jev


def auth() -> ClassifierOptions:
    return ClassifierOptions(api_key="cf-key", env={"CLOUDFLARE_ACCOUNT_ID": "account-id"})


def test_exposes_jev_only_through_classifier_catalog_accessors():
    models, jev = setup()
    assert jev == get_builtin_classifier_model("cloudflare-workers-ai", "typesafe/jev")
    assert jev.type == "classifier"
    assert jev.api == "cloudflare-workers-ai-system-one"
    assert models.get_model("cloudflare-workers-ai", "typesafe/jev") is None


@pytest.mark.tonio
async def test_runs_jev_through_the_account_scoped_ai_run_endpoint():
    models, jev = setup()

    with stub_system_one(respond_json(rest_response("Completed"))) as requests:
        result = await models.classify(jev, CONTEXT, auth())

    payload = requests[0].payload
    assert payload["model"] == "typesafe/jev"
    assert payload["input"]["state"] == CONTEXT.state
    assert payload["input"]["questions"]["is_urgent"]["type"] == "noul"
    assert payload["input"]["questions"]["department"]["type"] == "choice"
    assert requests[0].headers["authorization"] == "Bearer cf-key"
    assert requests[0].url == "https://api.cloudflare.com/client/v4/accounts/account-id/ai/run"
    assert result.stop_reason == "stop"
    assert result.answers["is_urgent"] == ClassifierBoolAnswer(probability=0.95)
    department = result.answers["department"]
    assert (department.type, department.choice, department.confidence) == ("choice", "billing", 0.8)
    assert (result.usage.input, result.usage.output, result.usage.total_tokens) == (426, 73, 499)


@pytest.mark.tonio
@pytest.mark.parametrize(
    ("model_id", "input_price"), [("@cf/cloudflare/clef", 0.24), ("@cf/cloudflare/clef-flash", 0.09)]
)
async def test_runs_clef_through_ai_run_and_parses_its_direct_output(model_id, input_price):
    models, _jev = setup()
    clef = models.get_model_of_type("classifier", "cloudflare-workers-ai", model_id)
    assert clef is not None, f"missing Cloudflare {model_id} model"

    envelope = {"result": CLEF_OUTPUT, "success": True, "errors": [], "messages": []}
    with stub_system_one(respond_json(envelope)) as requests:
        result = await models.classify(clef, CONTEXT, auth())

    payload = requests[0].payload
    assert payload["model"] == model_id
    assert payload["input"]["state"] == CONTEXT.state
    assert payload["input"]["questions"]["is_urgent"]["type"] == "noul"
    assert requests[0].url == "https://api.cloudflare.com/client/v4/accounts/account-id/ai/run"
    assert result.stop_reason == "stop"
    assert result.answers["is_urgent"] == ClassifierBoolAnswer(probability=0.9912)
    department = result.answers["department"]
    assert (department.type, department.choice, department.confidence) == ("choice", "technical", 0.4538)
    assert (result.usage.input, result.usage.output, result.usage.total_tokens) == (222, 0, 222)
    assert result.usage.cost.input == pytest.approx((222 * input_price) / 1_000_000)


@pytest.mark.tonio
async def test_reports_runs_that_did_not_complete():
    models, jev = setup()
    with stub_system_one(respond_json(rest_response("Queued", None))):
        result = await models.classify(jev, CONTEXT, auth())

    assert result.stop_reason == "error"
    assert "run did not complete (state: Queued)" in result.error_message


@pytest.mark.tonio
async def test_reports_cloudflare_envelope_errors():
    models, jev = setup()
    envelope = {"success": False, "errors": [{"code": 5007, "message": "No such model"}], "result": None}
    with stub_system_one(respond_json(envelope)):
        result = await models.classify(jev, CONTEXT, auth())

    assert result.stop_reason == "error"
    assert "Cloudflare Workers AI error: No such model" in result.error_message
