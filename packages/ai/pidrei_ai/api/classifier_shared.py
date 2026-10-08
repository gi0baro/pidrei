"""Port of pi's shared classifier HTTP code (packages/ai/src/api/classifier-shared.ts).

pi posts with `fetch` (or the caller's `options.fetch`) and a per-attempt
`AbortSignal.timeout(timeoutMs)`. Here the one POST goes over the punkreq seam
through `_ClassifierClient` — the tests' interception point, like
`openrouter_images._OpenRouterImagesClient` — and the per-attempt timeout is a
timer `CancelToken` combined with the caller's, so an expired attempt surfaces
as pi's "Request timed out after Nms" while a caller cancel stays an abort.
"""

import json
import math
from collections.abc import Sequence
from typing import Any

import tonio.colored as tonio

from pidrei_ai.builders import UsageBuilder
from pidrei_ai.registry import calculate_cost
from pidrei_ai.types import (
    ClassifierModel,
    ClassifierOptions,
    ProviderEnv,
    ProviderHeaders,
    ProviderResponse,
    Usage,
)
from pidrei_ai.utils.callbacks import maybe_call
from pidrei_ai.utils.headers import provider_headers_to_record
from pidrei_ai.utils.provider_retry import retry_provider_request
from pidrei_http import http
from pidrei_utils import clock
from pidrei_utils.cancel import AbortError, CancelToken, combine_cancel_tokens, run_cancellable


class ClassifierHttpError(Exception):
    """An HTTP failure in the shape `retry_provider_request` and `normalize_provider_error` understand."""

    def __init__(self, message: str, status: int | None, headers: dict[str, str] | None, body: str):
        super().__init__(message)
        self.status = status
        self.headers = headers
        self.body = body


def _http_error(label: str, status: int, headers: dict[str, str], body: str) -> ClassifierHttpError:
    return ClassifierHttpError(f"{label} returned {status}", status, headers, body)


def _format_ms(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _timeout_error(timeout_ms: float) -> ClassifierHttpError:
    return ClassifierHttpError(f"Request timed out after {_format_ms(timeout_ms)}ms", None, None, "")


def is_record(value: Any) -> bool:
    return isinstance(value, dict)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def required_number(label: str, value: Any, field: str) -> float:
    if not _is_number(value):
        raise RuntimeError(f"{label} returned an invalid {field}")
    return value


def _request_headers(model: ClassifierModel, api_key: str, options_headers: ProviderHeaders | None) -> dict[str, str]:
    return (
        provider_headers_to_record(
            {"authorization": f"Bearer {api_key}", "content-type": "application/json"},
            model.headers,
            options_headers,
        )
        or {}
    )


class _ClassifierClient:
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


async def post_classifier_request(
    label: str,
    url: str,
    model: ClassifierModel,
    body: Any,
    options: ClassifierOptions | None,
    no_retry_statuses: Sequence[int] | None = None,
) -> Any:
    """Posts one JSON classifier request with bearer auth, `on_payload`/`on_response`
    hooks, a fresh timeout per attempt, and provider retries. Returns the parsed
    response body; raises on failure. `no_retry_statuses` lists HTTP statuses
    that fail at once although they are normally retried."""
    if options is None or not options.api_key:
        raise RuntimeError(f"No API key for provider: {model.provider}")
    caller_cancel = options.cancel
    timeout_ms = options.timeout_ms
    payload = body
    transformed = await maybe_call(options.on_payload, payload, model)
    if transformed is not None:
        payload = transformed
    client = _ClassifierClient(options.env)
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
                raise _http_error(label, status, response_headers, text)
            return ProviderResponse(status=status, headers=response_headers), json.loads(text)
        except Exception:
            if timeout is not None and timeout.cancelled and not (caller_cancel and caller_cancel.cancelled):
                raise _timeout_error(timeout_ms) from None
            raise
        finally:
            combined.cleanup()
            timer.cancel()

    response, parsed = await retry_provider_request(
        attempt,
        max_retries=options.max_retries if options.max_retries is not None else 2,
        max_retry_delay_ms=options.max_retry_delay_ms,
        cancel=caller_cancel,
        no_retry_statuses=no_retry_statuses,
    )
    await maybe_call(options.on_response, response, model)
    return parsed


def _token_count(value: Any) -> float:
    return value if _is_number(value) and value > 0 else 0


def parse_classifier_usage(value: Any, model: ClassifierModel) -> Usage | None:
    """Usage from a `{input_tokens, output_tokens}` object, priced from the model
    catalog like chat usage. A missing or malformed usage object leaves the
    result without usage instead of failing it."""
    if not is_record(value) or ("input_tokens" not in value and "output_tokens" not in value):
        return None
    input = _token_count(value.get("input_tokens"))
    output = _token_count(value.get("output_tokens"))
    usage = UsageBuilder(input=input, output=output, total_tokens=input + output)
    calculate_cost(model, usage)
    return usage.freeze()
