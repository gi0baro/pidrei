"""Mirror of pi's typesafe-system-one.test.ts.

pi injects `options.fetch`; here the one POST is stubbed at
`system_one_shared._SystemOneClient` (tests/system_one_helpers.py). pi's
"preserves prototype-sensitive question IDs" case is JS-only (`__proto__` on a
plain object); a Python dict has no prototype, so it is not mirrored.
"""

from dataclasses import replace

import pytest

from pidrei_ai.api.typesafe_system_one import classify
from pidrei_ai.types import (
    ClassifierBoolAnswer,
    ClassifierBoolQuestion,
    ClassifierChoiceQuestion,
    ClassifierContext,
    ClassifierModel,
    ClassifierOptions,
    ClassifierScoreAnswer,
    ClassifierScoreQuestion,
    ModelCost,
)
from tests.system_one_helpers import json_response, respond_json, stub_system_one


MODEL = ClassifierModel(
    id="jev-latest",
    name="Jev",
    api="typesafe-system-one",
    provider="typesafe",
    base_url="https://api.typesafe.ai/v1/",
    input=["text"],
    cost=ModelCost(),
    context_window=64000,
)

CONTEXT = ClassifierContext(
    state={"text": "The deployment succeeded, thank you."},
    questions={
        "category": ClassifierChoiceQuestion(
            instructions="Classify the message", criteria={"success": "Successful", "failure": "Failed"}
        ),
        "satisfaction": ClassifierScoreQuestion(instructions="Score satisfaction", criteria=["low", "neutral", "high"]),
        "approved": ClassifierBoolQuestion(
            instructions="Does the user approve?", criteria={"true": "Approval", "false": "No approval"}
        ),
    },
)

WIRE_ANSWERS = {
    "category": {
        "type": "choice",
        "choice": "success",
        "probabilities": {"success": 0.9, "failure": 0.1},
        "confidence": 0.8,
    },
    "satisfaction": {"type": "score", "score": 2, "confidence": 0.7},
    "approved": {"type": "noul", "noul": 0.95},
}


@pytest.mark.tonio
async def test_maps_public_bool_questions_and_answers_to_typesafe_noul_values():
    with stub_system_one(respond_json({"answers": WIRE_ANSWERS})) as requests:
        result = await classify(MODEL, CONTEXT, ClassifierOptions(api_key="secret", temperature=1.5))
    with stub_system_one(respond_json({"answers": WIRE_ANSWERS, "usage": {"input_tokens": 308, "output_tokens": 23}})):
        priced_result = await classify(
            replace(MODEL, cost=ModelCost(input=0.042)), CONTEXT, ClassifierOptions(api_key="secret")
        )

    assert len(requests) == 1
    payload = requests[0].payload
    assert payload["model"] == "jev-latest"
    assert payload["questions"]["category"]["type"] == "choice"
    assert payload["questions"]["satisfaction"]["type"] == "score"
    assert payload["questions"]["approved"]["type"] == "noul"
    # System One has no temperature field; the option is ignored.
    assert "temperature" not in payload
    assert requests[0].headers["authorization"] == "Bearer secret"
    assert requests[0].url == "https://api.typesafe.ai/v1/systemone"
    assert result.stop_reason == "stop"
    assert result.answers["approved"] == ClassifierBoolAnswer(probability=0.95)
    assert result.answers["category"].type == "choice"
    assert result.answers["category"].choice == "success"
    assert result.answers["satisfaction"] == ClassifierScoreAnswer(score=2, confidence=0.7)
    assert result.usage is None
    assert (priced_result.usage.input, priced_result.usage.output, priced_result.usage.total_tokens) == (308, 23, 331)
    assert priced_result.usage.cost.total == pytest.approx(0.000012936, abs=1e-12)


@pytest.mark.tonio
async def test_posts_openrouter_system_one_requests_to_its_typesafe_compatible_endpoint():
    # Response shape observed from the live OpenRouter endpoint.
    response = {
        "id": "gen-dec-1",
        "provider": "TypeSafe",
        "answers": WIRE_ANSWERS,
        "usage": {"input_tokens": 308, "output_tokens": 23, "cost": 0.000012936},
    }
    open_router_model = replace(
        MODEL,
        id="typesafe/jev-1.13",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        cost=ModelCost(input=0.042),
    )

    with stub_system_one(respond_json(response)) as requests:
        result = await classify(open_router_model, CONTEXT, ClassifierOptions(api_key="secret"))

    assert requests[0].payload["model"] == "typesafe/jev-1.13"
    assert requests[0].payload["state"] == CONTEXT.state
    assert requests[0].url == "https://openrouter.ai/api/v1/systemone"
    assert result.stop_reason == "stop"
    assert result.answers["approved"] == ClassifierBoolAnswer(probability=0.95)
    # Priced from the catalog like chat usage; matches OpenRouter's reported cost.
    assert result.usage.cost.total == pytest.approx(0.000012936, abs=1e-12)


@pytest.mark.tonio
async def test_rejects_models_for_other_classifier_apis():
    with stub_system_one(respond_json({"answers": WIRE_ANSWERS})) as requests:
        result = await classify(
            replace(MODEL, api="cloudflare-workers-ai-system-one"), CONTEXT, ClassifierOptions(api_key="secret")
        )

    assert requests == []
    assert result.stop_reason == "error"
    assert "Unsupported classifier API: cloudflare-workers-ai-system-one" in result.error_message


@pytest.mark.tonio
async def test_merges_headers_case_insensitively_and_supports_none_suppression():
    model_with_headers = replace(MODEL, headers={"authorization": "Bearer model", "X-Source": "model"})

    with stub_system_one(respond_json({"answers": WIRE_ANSWERS})) as requests:
        await classify(
            model_with_headers,
            CONTEXT,
            ClassifierOptions(api_key="secret", headers={"Authorization": "Bearer request", "x-source": "request"}),
        )
        await classify(
            model_with_headers, CONTEXT, ClassifierOptions(api_key="secret", headers={"Authorization": None})
        )

    first = {name.lower(): value for name, value in requests[0].headers.items()}
    assert first["authorization"] == "Bearer request"
    assert first["x-source"] == "request"
    assert len([name for name in requests[0].headers if name.lower() == "authorization"]) == 1
    assert "authorization" not in {name.lower() for name in requests[1].headers}


@pytest.mark.tonio
async def test_reports_request_timeouts_separately_from_caller_cancellation():
    async def hang(request):
        await request.cancel.wait()
        raise request.cancel.reason

    with stub_system_one(hang):
        result = await classify(MODEL, CONTEXT, ClassifierOptions(api_key="secret", timeout_ms=5, max_retries=0))

    assert result.stop_reason == "error"
    assert result.error_message == "Request timed out after 5ms"


@pytest.mark.tonio
async def test_creates_a_fresh_timeout_for_every_retry_attempt():
    attempts: list = []

    async def handler(request):
        attempts.append(request)
        if len(attempts) == 1:
            return 500, {"retry-after-ms": "0"}, "retry"
        return json_response({"answers": WIRE_ANSWERS})

    with stub_system_one(handler):
        result = await classify(MODEL, CONTEXT, ClassifierOptions(api_key="secret", timeout_ms=1000, max_retries=1))

    assert result.stop_reason == "stop"
    assert len(attempts) == 2
    assert all(attempt.cancel is not None for attempt in attempts)
    assert attempts[0].cancel is not attempts[1].cancel


@pytest.mark.tonio
async def test_returns_malformed_responses_as_classifier_errors():
    with stub_system_one(respond_json({"answers": {}, "usage": {"input_tokens": 10, "output_tokens": 2}})):
        result = await classify(MODEL, CONTEXT, ClassifierOptions(api_key="secret"))

    assert result.stop_reason == "error"
    assert result.answers == {}
    assert "did not return an answer for category" in result.error_message
    # The request was billed, so its usage is kept.
    assert (result.usage.input, result.usage.output) == (10, 2)


@pytest.mark.tonio
async def test_ignores_malformed_usage():
    with stub_system_one(
        respond_json({"answers": WIRE_ANSWERS, "usage": {"input_tokens": "many", "output_tokens": 3}})
    ):
        result = await classify(MODEL, CONTEXT, ClassifierOptions(api_key="secret"))
    with stub_system_one(respond_json({"answers": WIRE_ANSWERS, "usage": {"cost": 0.1}})):
        without_tokens = await classify(MODEL, CONTEXT, ClassifierOptions(api_key="secret"))

    assert result.stop_reason == "stop"
    assert (result.usage.input, result.usage.output, result.usage.total_tokens) == (0, 3, 3)
    assert without_tokens.stop_reason == "stop"
    assert without_tokens.usage is None
