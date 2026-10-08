"""Port of pi's System One classification core (packages/ai/src/api/system-one-shared.ts).

TypeSafe's System One protocol, shared by every service that serves it; a
`SystemOneTransport` carries the per-service differences (URL, request and
response envelopes). The HTTP request itself is `classifier_shared`'s.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pidrei_ai.api.classifier_shared import (
    is_record,
    parse_classifier_usage,
    post_classifier_request,
    required_number,
)
from pidrei_ai.types import (
    ClassifierAnswer,
    ClassifierApi,
    ClassifierBoolAnswer,
    ClassifierChoiceAnswer,
    ClassifierContext,
    ClassifierModel,
    ClassifierOptions,
    ClassifierResult,
    ClassifierScoreAnswer,
)
from pidrei_ai.utils.error_body import format_provider_error, normalize_provider_error
from pidrei_utils import clock


@dataclass(slots=True, frozen=True)
class SystemOneTransport:
    """Differences between services that serve System One models."""

    # Classifier API implemented by this transport.
    api: ClassifierApi
    # Service name used in error messages.
    label: str
    # Absolute request URL.
    url: Callable[[ClassifierModel], str]
    # Wraps the System One request (`{state, questions}`) in the service's request envelope.
    payload: Callable[[ClassifierModel, dict[str, Any]], Any]
    # Extracts the System One output (`{answers, usage}`) from the service's response envelope.
    output: Callable[[Any], dict[str, Any]]


def _probabilities(label: str, value: Any, id: str) -> dict[str, float]:
    if not is_record(value):
        raise RuntimeError(f"{label} returned invalid probabilities for {id}")
    return {
        key: required_number(label, probability, f"probability for {id}.{key}") for key, probability in value.items()
    }


def _parse_answers(label: str, value: Any, context: ClassifierContext) -> dict[str, ClassifierAnswer]:
    if not is_record(value):
        raise RuntimeError(f"{label} returned an unexpected response")
    answers: dict[str, ClassifierAnswer] = {}
    for id, question in context.questions.items():
        answer = value.get(id)
        if not is_record(answer):
            raise RuntimeError(f"{label} did not return an answer for {id}")
        if question.type == "choice":
            if answer.get("type") != "choice" or not isinstance(answer.get("choice"), str):
                raise RuntimeError(f"{label} did not return a choice answer for {id}")
            answers[id] = ClassifierChoiceAnswer(
                choice=answer["choice"],
                probabilities=_probabilities(label, answer.get("probabilities"), id),
                confidence=required_number(label, answer.get("confidence"), f"confidence for {id}"),
            )
        elif question.type == "score":
            if answer.get("type") != "score":
                raise RuntimeError(f"{label} did not return a score answer for {id}")
            answers[id] = ClassifierScoreAnswer(
                score=required_number(label, answer.get("score"), f"score for {id}"),
                confidence=required_number(label, answer.get("confidence"), f"confidence for {id}"),
            )
        else:
            if answer.get("type") != "noul":
                raise RuntimeError(f"{label} did not return a bool answer for {id}")
            answers[id] = ClassifierBoolAnswer(
                probability=required_number(label, answer.get("noul"), f"probability for {id}")
            )
    return answers


def _wire_request(context: ClassifierContext) -> dict[str, Any]:
    """Maps public `bool` questions to TypeSafe's wire-level `noul` type."""
    return {
        "state": context.state,
        "questions": {
            id: {
                "type": "noul" if question.type == "bool" else question.type,
                "instructions": question.instructions,
                "criteria": question.criteria,
            }
            for id, question in context.questions.items()
        },
    }


async def classify_system_one(
    transport: SystemOneTransport,
    model: ClassifierModel,
    context: ClassifierContext,
    options: ClassifierOptions | None,
) -> ClassifierResult:
    """Runs one System One classification over the given transport. Never raises."""
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
        if model.api != transport.api:
            raise RuntimeError(f"Unsupported classifier API: {model.api}")
        if context.images:
            raise RuntimeError(f"{transport.label} does not support image input")
        body = await post_classifier_request(
            transport.label,
            transport.url(model),
            model,
            transport.payload(model, _wire_request(context)),
            options,
        )
        result = transport.output(body)
        # Set before parsing answers: a request with malformed answers was still billed.
        usage = parse_classifier_usage(result.get("usage"), model)
        if usage is not None:
            output.usage = usage
        output.answers = _parse_answers(transport.label, result.get("answers"), context)
        return output
    except Exception as error:
        output.stop_reason = "aborted" if caller_cancel is not None and caller_cancel.cancelled else "error"
        output.error_message = format_provider_error(normalize_provider_error(error), f"{transport.label} error")
        return output
