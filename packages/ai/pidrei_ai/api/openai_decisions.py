"""Port of pi's OpenAI Decisions classifier (packages/ai/src/api/openai-decisions.ts).

OpenAI's Decisions API: `POST /v1/decisions` with `{model, input, questions}`.
https://developers.openai.com/api/docs/guides/decisions

The state is sent as JSON text. With images, the input becomes one user message with the state
as `input_text` followed by `input_image` data URLs. Questions map to Decisions types:
`choice` to `choice`, `score` to `score`, and `bool` to `predicate`. Predicates have no
criteria field, so the meanings of true and false are appended to the instructions.

Only OpenAI API keys work: Sign in with ChatGPT tokens are rejected on this route.
"""

import json
from typing import Any

from pidrei_ai.api.classifier_shared import (
    ClassifierHttpError,
    is_record,
    parse_classifier_usage,
    post_classifier_request,
    required_number,
)
from pidrei_ai.types import (
    ClassifierAnswer,
    ClassifierBoolAnswer,
    ClassifierBoolQuestion,
    ClassifierChoiceAnswer,
    ClassifierContext,
    ClassifierModel,
    ClassifierOptions,
    ClassifierQuestion,
    ClassifierResult,
    ClassifierScoreAnswer,
)
from pidrei_ai.utils.error_body import format_provider_error, normalize_provider_error
from pidrei_utils import clock


_LABEL = "OpenAI Decisions"

# The endpoint accepts at most this many image parts per request.
_MAX_IMAGES = 128


def _predicate_instructions(question: ClassifierBoolQuestion) -> str:
    meanings = [
        meaning
        for meaning in (
            f"True means: {question.criteria['true']}" if question.criteria.get("true") else "",
            f"False means: {question.criteria['false']}" if question.criteria.get("false") else "",
        )
        if meaning
    ]
    return f"{question.instructions}\n\n" + "\n".join(meanings) if meanings else question.instructions


def _wire_question(name: str, question: ClassifierQuestion) -> dict[str, Any]:
    if question.type == "choice":
        return {
            "type": "choice",
            "name": name,
            "instructions": question.instructions,
            "choices": [
                {"value": value, "description": description} if description else {"value": value}
                for value, description in question.criteria.items()
            ],
        }
    if question.type == "score":
        return {
            "type": "score",
            "name": name,
            "instructions": question.instructions,
            "levels": [{"label": label} for label in question.criteria],
        }
    return {"type": "predicate", "name": name, "instructions": _predicate_instructions(question)}


def _wire_input(context: ClassifierContext) -> Any:
    # pi: JSON.stringify(context.state).
    state = json.dumps(context.state, separators=(",", ":"), ensure_ascii=False)
    images = context.images or []
    if not images:
        return state
    if len(images) > _MAX_IMAGES:
        raise RuntimeError(f"{_LABEL} accepts at most {_MAX_IMAGES} images, got {len(images)}")
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": state},
                *(
                    {"type": "input_image", "image_url": f"data:{image.mime_type};base64,{image.data}"}
                    for image in images
                ),
            ],
        }
    ]


def _choice_probabilities(value: Any, id: str) -> dict[str, float]:
    if not isinstance(value, list):
        raise RuntimeError(f"{_LABEL} returned invalid probabilities for {id}")  # noqa: TRY004 - pi throws a plain Error
    probabilities: dict[str, float] = {}
    for entry in value:
        if not is_record(entry) or not isinstance(entry.get("value"), str):
            raise RuntimeError(f"{_LABEL} returned invalid probabilities for {id}")
        probabilities[entry["value"]] = required_number(
            _LABEL, entry.get("probability"), f"probability for {id}.{entry['value']}"
        )
    return probabilities


def _parse_answer(id: str, question: ClassifierQuestion, answer: dict[str, Any]) -> ClassifierAnswer:
    if answer.get("type") == "refusal":
        raise RuntimeError(f"{_LABEL} refused to answer {id}")
    if question.type == "choice":
        if answer.get("type") != "choice" or not isinstance(answer.get("choice"), str):
            raise RuntimeError(f"{_LABEL} did not return a choice answer for {id}")
        return ClassifierChoiceAnswer(
            choice=answer["choice"],
            probabilities=_choice_probabilities(answer.get("probabilities"), id),
            confidence=required_number(_LABEL, answer.get("confidence"), f"confidence for {id}"),
        )
    if question.type == "score":
        if answer.get("type") != "score":
            raise RuntimeError(f"{_LABEL} did not return a score answer for {id}")
        return ClassifierScoreAnswer(
            score=required_number(_LABEL, answer.get("score"), f"score for {id}"),
            confidence=required_number(_LABEL, answer.get("confidence"), f"confidence for {id}"),
        )
    if answer.get("type") != "predicate":
        raise RuntimeError(f"{_LABEL} did not return a predicate answer for {id}")
    return ClassifierBoolAnswer(probability=required_number(_LABEL, answer.get("probability"), f"probability for {id}"))


def _parse_answers(value: Any, context: ClassifierContext) -> dict[str, ClassifierAnswer]:
    if not isinstance(value, list):
        raise RuntimeError(f"{_LABEL} returned an unexpected response")  # noqa: TRY004 - pi throws a plain Error
    by_name: dict[str, dict[str, Any]] = {}
    for answer in value:
        if is_record(answer) and isinstance(answer.get("name"), str):
            by_name[answer["name"]] = answer
    answers: dict[str, ClassifierAnswer] = {}
    for id, question in context.questions.items():
        answer = by_name.get(id)
        if answer is None:
            raise RuntimeError(f"{_LABEL} did not return an answer for {id}")
        answers[id] = _parse_answer(id, question, answer)
    return answers


# Cloudflare in front of api.openai.com answers 504 with an HTML page when a request runs longer
# than about five seconds. Large inputs, currently above roughly 600K tokens, hit this limit, and
# retrying the same input runs into it again, so 504 is not retried.
_NO_RETRY_STATUSES = (504,)


def _error_message(error: BaseException) -> str:
    if isinstance(error, ClassifierHttpError) and error.status == 504:
        return (
            f"{_LABEL} error (504): the request timed out at the gateway. Very large inputs "
            "(above roughly 600K tokens) currently exceed its time limit."
        )
    return format_provider_error(normalize_provider_error(error), f"{_LABEL} error")


async def classify(
    model: ClassifierModel, context: ClassifierContext, options: ClassifierOptions | None = None
) -> ClassifierResult:
    """Classification through OpenAI's Decisions API. Never raises."""
    output = ClassifierResult(
        api=model.api,
        provider=model.provider,
        model=model.id,
        answers={},
        stop_reason="stop",
        timestamp=clock.now_ms(),
    )
    caller_cancel = options.cancel if options is not None else None

    try:
        if model.api != "openai-decisions":
            raise RuntimeError(f"Unsupported classifier API: {model.api}")
        body = await post_classifier_request(
            _LABEL,
            f"{model.base_url.rstrip('/')}/decisions",
            model,
            {
                "model": model.id,
                "input": _wire_input(context),
                "questions": [_wire_question(id, question) for id, question in context.questions.items()],
            },
            options,
            _NO_RETRY_STATUSES,
        )
        if not is_record(body):
            raise RuntimeError(f"{_LABEL} returned an unexpected response")
        # Set before parsing answers: a request with malformed or refused answers was still billed.
        usage = parse_classifier_usage(body.get("usage"), model)
        if usage is not None:
            output.usage = usage
        output.answers = _parse_answers(body.get("answers"), context)
        return output
    except Exception as error:
        output.answers = {}
        output.stop_reason = "aborted" if caller_cancel is not None and caller_cancel.cancelled else "error"
        output.error_message = _error_message(error)
        return output
