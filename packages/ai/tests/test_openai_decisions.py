"""Mirror of pi's openai-decisions.test.ts.

pi injects `options.fetch`; here the one POST is stubbed at
`classifier_shared._ClassifierClient` (tests/system_one_helpers.py). pi's
"preserves prototype-sensitive question IDs in answers" case is JS-only
(`__proto__` on a plain object); a Python dict has no prototype, so it is not
mirrored.
"""

import json
from dataclasses import replace

import pytest

from pidrei_ai.api.openai_decisions import classify
from pidrei_ai.types import (
    ClassifierBoolAnswer,
    ClassifierBoolQuestion,
    ClassifierChoiceAnswer,
    ClassifierChoiceQuestion,
    ClassifierContext,
    ClassifierModel,
    ClassifierOptions,
    ClassifierScoreAnswer,
    ClassifierScoreQuestion,
    ImageContent,
    ModelCost,
    ModelCostTier,
)
from tests.system_one_helpers import json_response, respond_json, stub_system_one


MODEL = ClassifierModel(
    id="gpt-6-luna",
    name="GPT-6 Luna",
    api="openai-decisions",
    provider="openai",
    base_url="https://api.openai.com/v1",
    input=["text", "image"],
    cost=ModelCost(
        input=0.1,
        tiers=[ModelCostTier(input=0.2, output=0, cache_read=0, cache_write=0, input_tokens_above=272000)],
    ),
    context_window=922000,
)

CONTEXT = ClassifierContext(
    state={"text": "The deployment succeeded, thank you."},
    questions={
        "category": ClassifierChoiceQuestion(
            instructions="Classify the message", criteria={"success": "Successful", "failure": ""}
        ),
        "satisfaction": ClassifierScoreQuestion(instructions="Score satisfaction", criteria=["low", "neutral", "high"]),
        "approved": ClassifierBoolQuestion(
            instructions="Does the user approve?", criteria={"true": "Approval", "false": "No approval"}
        ),
    },
)

# Response shape from the API reference and live `gpt-6-luna` requests.
WIRE_ANSWERS = [
    {
        "type": "choice",
        "name": "category",
        "choice": "success",
        "probabilities": [{"value": "success", "probability": 0.9}, {"value": "failure", "probability": 0.1}],
        "confidence": 0.8,
    },
    {
        "type": "score",
        "name": "satisfaction",
        "score": 1.8,
        "probabilities": [
            {"value": 0, "label": "low", "probability": 0.05},
            {"value": 1, "label": "neutral", "probability": 0.1},
            {"value": 2, "label": "high", "probability": 0.85},
        ],
        "confidence": 0.7,
    },
    {"type": "predicate", "name": "approved", "probability": 0.95},
]

WIRE_USAGE = {
    "input_tokens": 164,
    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
    "output_tokens": 0,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 164,
}

IMAGE = ImageContent(data="aW1hZ2U=", mime_type="image/png")

# pi: JSON.stringify(context.state).
STATE_JSON = json.dumps(CONTEXT.state, separators=(",", ":"))


@pytest.mark.tonio
async def test_maps_questions_to_decisions_types_and_answers_back_by_name():
    # Answers out of question order: they are matched by name.
    response = {"model": "gpt-6-luna", "answers": list(reversed(WIRE_ANSWERS)), "usage": WIRE_USAGE}
    with stub_system_one(respond_json(response)) as requests:
        result = await classify(MODEL, CONTEXT, ClassifierOptions(api_key="secret", temperature=1.5))

    assert len(requests) == 1
    assert requests[0].url == "https://api.openai.com/v1/decisions"
    assert requests[0].headers["authorization"] == "Bearer secret"
    assert requests[0].payload == {
        "model": "gpt-6-luna",
        "input": STATE_JSON,
        "questions": [
            {
                "type": "choice",
                "name": "category",
                "instructions": "Classify the message",
                # Empty descriptions are omitted.
                "choices": [{"value": "success", "description": "Successful"}, {"value": "failure"}],
            },
            {
                "type": "score",
                "name": "satisfaction",
                "instructions": "Score satisfaction",
                "levels": [{"label": "low"}, {"label": "neutral"}, {"label": "high"}],
            },
            {
                "type": "predicate",
                "name": "approved",
                "instructions": "Does the user approve?\n\nTrue means: Approval\nFalse means: No approval",
            },
        ],
    }
    assert result.stop_reason == "stop"
    assert result.answers == {
        "category": ClassifierChoiceAnswer(
            choice="success", probabilities={"success": 0.9, "failure": 0.1}, confidence=0.8
        ),
        "satisfaction": ClassifierScoreAnswer(score=1.8, confidence=0.7),
        "approved": ClassifierBoolAnswer(probability=0.95),
    }
    usage = result.usage
    assert (usage.input, usage.output, usage.cache_read, usage.total_tokens) == (164, 0, 0, 164)
    assert usage.cost.total == pytest.approx(0.0000164, abs=1e-12)


@pytest.mark.tonio
async def test_prices_long_context_requests_at_the_long_context_input_rate():
    with stub_system_one(
        respond_json({"answers": WIRE_ANSWERS, "usage": {"input_tokens": 300000, "output_tokens": 0}})
    ):
        result = await classify(MODEL, CONTEXT, ClassifierOptions(api_key="secret"))

    assert result.usage.cost.total == pytest.approx(0.06, abs=1e-12)


@pytest.mark.tonio
async def test_sends_images_after_the_state_in_one_user_message():
    context = replace(CONTEXT, images=[IMAGE, replace(IMAGE, mime_type="image/jpeg")])
    with stub_system_one(respond_json({"answers": WIRE_ANSWERS})) as requests:
        result = await classify(MODEL, context, ClassifierOptions(api_key="secret"))

    assert result.stop_reason == "stop"
    assert requests[0].payload["input"] == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": STATE_JSON},
                {"type": "input_image", "image_url": "data:image/png;base64,aW1hZ2U="},
                {"type": "input_image", "image_url": "data:image/jpeg;base64,aW1hZ2U="},
            ],
        }
    ]


@pytest.mark.tonio
async def test_rejects_more_than_128_images_before_sending():
    with stub_system_one(respond_json({"answers": WIRE_ANSWERS})) as requests:
        result = await classify(MODEL, replace(CONTEXT, images=[IMAGE] * 129), ClassifierOptions(api_key="secret"))

    assert requests == []
    assert result.stop_reason == "error"
    assert "at most 128 images, got 129" in result.error_message


@pytest.mark.tonio
async def test_fails_the_result_when_a_question_is_refused_and_keeps_the_billed_usage():
    response = {
        "answers": [WIRE_ANSWERS[0], WIRE_ANSWERS[1], {"type": "refusal", "name": "approved"}],
        "usage": WIRE_USAGE,
    }
    with stub_system_one(respond_json(response)):
        result = await classify(MODEL, CONTEXT, ClassifierOptions(api_key="secret"))

    assert result.stop_reason == "error"
    assert result.answers == {}
    assert result.error_message == "OpenAI Decisions refused to answer approved"
    assert result.usage.input == 164


@pytest.mark.tonio
async def test_returns_missing_and_mistyped_answers_as_classifier_errors():
    with stub_system_one(respond_json({"answers": WIRE_ANSWERS[:2]})):
        missing = await classify(MODEL, CONTEXT, ClassifierOptions(api_key="secret"))
    mistyped_answers = [WIRE_ANSWERS[0], WIRE_ANSWERS[1], {"type": "score", "name": "approved"}]
    with stub_system_one(respond_json({"answers": mistyped_answers})):
        mistyped = await classify(MODEL, CONTEXT, ClassifierOptions(api_key="secret"))

    assert missing.stop_reason == "error"
    assert "did not return an answer for approved" in missing.error_message
    assert mistyped.stop_reason == "error"
    assert "did not return a predicate answer for approved" in mistyped.error_message


@pytest.mark.tonio
async def test_does_not_retry_gateway_timeouts_and_explains_them_instead_of_returning_the_html_page():
    async def gateway_timeout(_request):
        return 504, {"retry-after-ms": "0"}, "<!DOCTYPE html><html>Gateway time-out</html>"

    # Default retries: the same input would time out again.
    with stub_system_one(gateway_timeout) as requests:
        result = await classify(MODEL, CONTEXT, ClassifierOptions(api_key="secret"))

    assert len(requests) == 1
    assert result.stop_reason == "error"
    assert "OpenAI Decisions error (504): the request timed out at the gateway" in result.error_message
    assert "<html>" not in result.error_message


@pytest.mark.tonio
async def test_still_retries_other_server_errors():
    attempts = 0

    async def busy_then_ok(_request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return 503, {"retry-after-ms": "0"}, "busy"
        return json_response({"answers": WIRE_ANSWERS})

    with stub_system_one(busy_then_ok):
        result = await classify(MODEL, CONTEXT, ClassifierOptions(api_key="secret"))

    assert attempts == 2
    assert result.stop_reason == "stop"


@pytest.mark.tonio
async def test_includes_the_api_error_body_for_other_http_failures():
    error = {"error": {"message": "Decision input exceeds the token limit.", "type": "invalid_request_error"}}

    async def bad_request(_request):
        return json_response(error, 400)

    with stub_system_one(bad_request):
        result = await classify(MODEL, CONTEXT, ClassifierOptions(api_key="secret", max_retries=0))

    assert result.stop_reason == "error"
    assert "OpenAI Decisions error (400)" in result.error_message
    assert "Decision input exceeds the token limit." in result.error_message


@pytest.mark.tonio
async def test_rejects_models_for_other_classifier_apis_and_missing_api_keys():
    with stub_system_one(respond_json({"answers": WIRE_ANSWERS})) as requests:
        other_api = await classify(
            replace(MODEL, api="typesafe-system-one"), CONTEXT, ClassifierOptions(api_key="secret")
        )
        no_key = await classify(MODEL, CONTEXT, ClassifierOptions())

    assert requests == []
    assert "Unsupported classifier API: typesafe-system-one" in other_api.error_message
    assert "No API key for provider: openai" in no_key.error_message
