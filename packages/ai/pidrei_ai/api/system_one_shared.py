"""Port of pi's System One classification core (packages/ai/src/api/system-one-shared.ts).

TypeSafe's System One protocol, shared by every service that serves it; a
`SystemOneTransport` carries the per-service differences (URL, request and
response envelopes). The HTTP request itself is `classifier_shared`'s.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
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
    ImageContent,
    SystemOneCompat,
    SystemOneImageInput,
)
from pidrei_ai.utils.error_body import format_provider_error, normalize_provider_error
from pidrei_utils import clock


# How a service takes images (`SystemOneCompat.image_input`):
# - `assets`: TypeSafe's `assets` map, `{"<image-1>": {type: "image", format, data_base64}}`,
#   with the names placed before the state: `state: ["<image-1>", state]`. PNG and JPEG only.
# - `images`: Clef's `images` list of data URLs, which the model reads before the state.
#   PNG, JPEG, and WebP, at most four.
# - `state-image-urls`: OpenRouter's form, data URL parts placed before the state:
#   `state: [{type: "image_url", image_url: {url}}, state]`. OpenRouter accepts this for every
#   model, and drops `assets` silently.


@dataclass(slots=True, frozen=True)
class ResolvedSystemOneCompat:
    """System One compatibility settings with every field resolved."""

    image_input: SystemOneImageInput
    send_image_opt_in_header: bool


# TypeSafe's opt-in header for its experimental image input.
_IMAGE_OPT_IN_HEADERS = {"TypeSafe-Danger-Zone-Images": "true"}


@dataclass(slots=True, frozen=True)
class SystemOneTransport:
    """Differences between services that serve System One models."""

    # Classifier API implemented by this transport.
    api: ClassifierApi
    # Service name used in error messages.
    label: str
    # Compatibility settings for models that do not set them in `compat`.
    detect_compat: Callable[[ClassifierModel], ResolvedSystemOneCompat]
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


_ASSET_FORMATS = {"image/png": "png", "image/jpeg": "jpeg"}
_DATA_URL_TYPES = frozenset(("image/png", "image/jpeg", "image/webp"))
# Clef accepts at most this many images per request.
_MAX_DATA_URL_IMAGES = 4


def _add_images(
    label: str,
    model: ClassifierModel,
    image_input: SystemOneImageInput,
    images: list[ImageContent],
    request: dict[str, Any],
) -> None:
    """Adds the context images to `request` in the service's form."""
    if image_input == "images":
        if len(images) > _MAX_DATA_URL_IMAGES:
            raise RuntimeError(
                f"{label} accepts at most {_MAX_DATA_URL_IMAGES} images for {model.id}, got {len(images)}"
            )
        data_urls = []
        for image in images:
            if image.mime_type not in _DATA_URL_TYPES:
                raise RuntimeError(f"{label} accepts PNG, JPEG, and WebP images for {model.id}, got {image.mime_type}")
            data_urls.append(f"data:{image.mime_type};base64,{image.data}")
        request["images"] = data_urls
        return
    if image_input == "state-image-urls":
        # OpenRouter validates the images against the model that serves the request.
        parts = [
            {"type": "image_url", "image_url": {"url": f"data:{image.mime_type};base64,{image.data}"}}
            for image in images
        ]
        request["state"] = [*parts, request["state"]]
        return
    assets: dict[str, dict[str, Any]] = {}
    for index, image in enumerate(images):
        image_format = _ASSET_FORMATS.get(image.mime_type)
        if image_format is None:
            raise RuntimeError(f"{label} accepts PNG and JPEG images for {model.id}, got {image.mime_type}")
        assets[f"<image-{index + 1}>"] = {"type": "image", "format": image_format, "data_base64": image.data}
    request["assets"] = assets
    # Every asset must be referenced from the request. The names go before the state, where Clef
    # also places its images.
    request["state"] = [*assets, request["state"]]


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
        request = _wire_request(context)
        request_options = options
        if context.images:
            if "image" not in model.input:
                raise RuntimeError(f"{transport.label} does not support image input for {model.id}")
            compat = model.compat or SystemOneCompat()
            detected = transport.detect_compat(model)
            _add_images(transport.label, model, compat.image_input or detected.image_input, context.images, request)
            send_opt_in = compat.send_image_opt_in_header
            if detected.send_image_opt_in_header if send_opt_in is None else send_opt_in:
                caller_headers = options.headers if options is not None else None
                headers = {**_IMAGE_OPT_IN_HEADERS, **(caller_headers or {})}
                request_options = (
                    replace(options, headers=headers) if options is not None else ClassifierOptions(headers=headers)
                )
        body = await post_classifier_request(
            transport.label,
            transport.url(model),
            model,
            transport.payload(model, request),
            request_options,
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
