"""Port of pi's System One classification core (packages/ai/src/api/system-one-shared.ts).

TypeSafe's System One protocol, shared by every service that serves it; a
`SystemOneTransport` carries the per-service differences (URL, request and
response envelopes).

pi posts with `fetch` (or the caller's `options.fetch`) and a per-attempt
`AbortSignal.timeout(timeoutMs)`. Here the one POST goes over the punkreq seam
through `_SystemOneClient` — the tests' interception point, like
`openrouter_images._OpenRouterImagesClient` — and the per-attempt timeout is a
timer `CancelToken` combined with the caller's, so an expired attempt surfaces
as pi's "Request timed out after Nms" while a caller cancel stays an abort.
"""

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import tonio.colored as tonio

from pidrei_ai.builders import UsageBuilder
from pidrei_ai.registry import calculate_cost
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
    ProviderEnv,
    ProviderHeaders,
    ProviderResponse,
    Usage,
)
from pidrei_ai.utils.callbacks import maybe_call
from pidrei_ai.utils.error_body import format_provider_error, normalize_provider_error
from pidrei_ai.utils.headers import provider_headers_to_record
from pidrei_ai.utils.provider_retry import retry_provider_request
from pidrei_http import http
from pidrei_utils import clock
from pidrei_utils.cancel import AbortError, CancelToken, combine_cancel_tokens, run_cancellable


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


class SystemOneHttpError(Exception):
    def __init__(self, message: str, status: int | None, headers: dict[str, str] | None, body: str):
        super().__init__(message)
        self.status = status
        self.headers = headers
        self.body = body


def _http_error(label: str, status: int, headers: dict[str, str], body: str) -> SystemOneHttpError:
    return SystemOneHttpError(f"{label} returned {status}", status, headers, body)


def _format_ms(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _timeout_error(timeout_ms: float) -> SystemOneHttpError:
    return SystemOneHttpError(f"Request timed out after {_format_ms(timeout_ms)}ms", None, None, "")


def is_record(value: Any) -> bool:
    return isinstance(value, dict)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _required_number(label: str, value: Any, field: str) -> float:
    if not _is_number(value):
        raise RuntimeError(f"{label} returned an invalid {field}")
    return value


def _probabilities(label: str, value: Any, id: str) -> dict[str, float]:
    if not is_record(value):
        raise RuntimeError(f"{label} returned invalid probabilities for {id}")
    return {
        key: _required_number(label, probability, f"probability for {id}.{key}") for key, probability in value.items()
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
                confidence=_required_number(label, answer.get("confidence"), f"confidence for {id}"),
            )
        elif question.type == "score":
            if answer.get("type") != "score":
                raise RuntimeError(f"{label} did not return a score answer for {id}")
            answers[id] = ClassifierScoreAnswer(
                score=_required_number(label, answer.get("score"), f"score for {id}"),
                confidence=_required_number(label, answer.get("confidence"), f"confidence for {id}"),
            )
        else:
            if answer.get("type") != "noul":
                raise RuntimeError(f"{label} did not return a bool answer for {id}")
            answers[id] = ClassifierBoolAnswer(
                probability=_required_number(label, answer.get("noul"), f"probability for {id}")
            )
    return answers


def _token_count(value: Any) -> float:
    return value if _is_number(value) and value > 0 else 0


def _parse_usage(value: Any, model: ClassifierModel) -> Usage | None:
    """Usage from System One's `{input_tokens, output_tokens}`, priced from the
    model catalog like chat usage. A missing or malformed usage object leaves
    the result without usage instead of failing it."""
    if not is_record(value) or ("input_tokens" not in value and "output_tokens" not in value):
        return None
    input = _token_count(value.get("input_tokens"))
    output = _token_count(value.get("output_tokens"))
    usage = UsageBuilder(input=input, output=output, total_tokens=input + output)
    calculate_cost(model, usage)
    return usage.freeze()


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


def _request_headers(model: ClassifierModel, api_key: str, options_headers: ProviderHeaders | None) -> dict[str, str]:
    return (
        provider_headers_to_record(
            {"authorization": f"Bearer {api_key}", "content-type": "application/json"},
            model.headers,
            options_headers,
        )
        or {}
    )


class _SystemOneClient:
    """One JSON POST over the punkreq seam (pi: `fetch(url, { method: "POST", ... })`)."""

    def __init__(self, env: ProviderEnv | None = None):
        self._env = env

    async def post(
        self, url: str, payload: Any, headers: dict[str, str], cancel: CancelToken | None
    ) -> tuple[int, dict[str, str], str]:
        client = http.client_for(url, self._env)

        async def _send() -> tuple[Any, bytes]:
            # The whole-request bound is the caller's per-attempt timer token.
            response = await client.post(url, json=payload, headers=headers, timeout=http.oneshot_timeout(None))
            try:
                body = await response.read()
            except BaseException:
                http.abandon_response(response)
                raise
            return response, body

        response, body = await run_cancellable(_send(), cancel)
        return (
            response.status_code,
            {name.lower(): value for name, value in response.headers.items()},
            body.decode("utf-8", "replace"),
        )


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
        if options is None or not options.api_key:
            raise RuntimeError(f"No API key for provider: {model.provider}")
        timeout_ms = options.timeout_ms
        payload = transport.payload(model, _wire_request(context))
        transformed = await maybe_call(options.on_payload, payload, model)
        if transformed is not None:
            payload = transformed
        client = _SystemOneClient(options.env)
        url = transport.url(model)
        headers = _request_headers(model, options.api_key, options.headers)

        async def attempt() -> tuple[ProviderResponse, Any]:
            # A fresh timeout for every attempt (pi: `AbortSignal.timeout` per request).
            timeout = CancelToken() if timeout_ms is not None else None
            timer = CancelToken()
            if timeout is not None:

                async def _expire() -> None:
                    try:
                        await clock.sleep_ms(timeout_ms, timer)
                    except AbortError:
                        return
                    timeout.cancel(TimeoutError("The operation timed out."))

                tonio.spawn.without_tracking(_expire())
            combined = combine_cancel_tokens(caller_cancel, timeout)
            try:
                status, response_headers, text = await client.post(url, payload, headers, combined.token)
                if not 200 <= status < 300:
                    raise _http_error(transport.label, status, response_headers, text)
                return ProviderResponse(status=status, headers=response_headers), json.loads(text)
            except Exception:
                if timeout is not None and timeout.cancelled and not (caller_cancel and caller_cancel.cancelled):
                    raise _timeout_error(timeout_ms) from None
                raise
            finally:
                combined.cleanup()
                timer.cancel()

        response, body = await retry_provider_request(
            attempt,
            max_retries=options.max_retries if options.max_retries is not None else 2,
            max_retry_delay_ms=options.max_retry_delay_ms,
            cancel=caller_cancel,
        )
        await maybe_call(options.on_response, response, model)
        result = transport.output(body)
        # Set before parsing answers: a request with malformed answers was still billed.
        usage = _parse_usage(result.get("usage"), model)
        if usage is not None:
            output.usage = usage
        output.answers = _parse_answers(transport.label, result.get("answers"), context)
        return output
    except Exception as error:
        output.stop_reason = "aborted" if caller_cancel is not None and caller_cancel.cancelled else "error"
        output.error_message = format_provider_error(normalize_provider_error(error), f"{transport.label} error")
        return output
