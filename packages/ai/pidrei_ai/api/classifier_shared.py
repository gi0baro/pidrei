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
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import tonio.colored as tonio

from pidrei_ai.builders import UsageBuilder
from pidrei_ai.registry import calculate_cost
from pidrei_ai.types import (
    ClassifierModel,
    ClassifierOptions,
    ClientIdentity,
    ProviderEnv,
    ProviderHeaders,
    ProviderResponse,
)
from pidrei_ai.utils.callbacks import maybe_call
from pidrei_ai.utils.headers import provider_headers_to_record
from pidrei_ai.utils.identity import get_identity_headers
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


def _request_headers(
    model: ClassifierModel,
    api_key: str,
    options_headers: ProviderHeaders | None,
    identity: ClientIdentity | None = None,
) -> dict[str, str]:
    return (
        provider_headers_to_record(
            get_identity_headers(model, identity),
            {"authorization": f"Bearer {api_key}", "content-type": "application/json"},
            model.headers,
            options_headers,
        )
        or {}
    )


class _ClassifierClient:
    """One JSON POST over the punkreq seam (pi: `fetch(url, { method: "POST", ... })`)."""

    def __init__(self, env: ProviderEnv | None = None, fetch: http.FetchFunction | None = None):
        self._env = env
        # pi: `options.fetch ?? globalThis.fetch`, shared by the three classifier APIs.
        self._fetch = fetch if fetch is not None else http.default_fetch

    async def post(
        self,
        url: str,
        payload: Any,
        headers: dict[str, str],
        cancel: CancelToken | None,
        on_response: Callable[[ProviderResponse], Awaitable[Any]],
    ) -> tuple[int, dict[str, str], str]:
        async def _head() -> Any:
            # The whole-request bound is the caller's per-attempt timer token.
            request = http.build_request("POST", url, json=payload, headers=headers, timeout=http.oneshot_timeout(None))
            return await self._fetch(request, env=self._env)

        response = await run_cancellable(_head(), cancel)
        response_headers = {name.lower(): value for name, value in response.headers.items()}
        try:
            # Reported before the body is read, so a body that fails to arrive still reports its status.
            # Outside the cancellable waits, as pi's callback is not interrupted by the attempt's signal.
            await on_response(ProviderResponse(status=response.status_code, headers=response_headers))
            body = await run_cancellable(response.read(), cancel)
        except BaseException:
            http.abandon_response(response)
            raise
        return response.status_code, response_headers, body.decode("utf-8", "replace")


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
    client = _ClassifierClient(options.env, options.fetch)
    headers = _request_headers(model, options.api_key, options.headers, options.identity)

    def report(response: ProviderResponse) -> Awaitable[Any]:
        return maybe_call(options.on_response, response, model)

    async def attempt() -> Any:
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
            status, response_headers, text = await client.post(url, payload, headers, combined.token, report)
            if not 200 <= status < 300:
                raise _http_error(label, status, response_headers, text)
            return json.loads(text)
        except Exception:
            if timeout is not None and timeout.cancelled and not (caller_cancel and caller_cancel.cancelled):
                raise _timeout_error(timeout_ms) from None
            raise
        finally:
            combined.cleanup()
            timer.cancel()

    return await retry_provider_request(
        attempt,
        max_retries=options.max_retries if options.max_retries is not None else 2,
        max_retry_delay_ms=options.max_retry_delay_ms,
        cancel=caller_cancel,
        no_retry_statuses=no_retry_statuses,
    )


def _token_count(value: Any) -> float:
    return value if _is_number(value) and value > 0 else 0


def parse_classifier_usage(value: Any, model: ClassifierModel) -> UsageBuilder | None:
    """Usage from a `{input_tokens, output_tokens}` object, priced from the model
    catalog like chat usage. A missing or malformed usage object leaves the
    result without usage instead of failing it.
    `input_tokens_details.cached_tokens` and `cache_write_tokens` are included
    in `input_tokens`, as in the Responses API, so they are split out of
    `input`.

    Returns the builder, which the caller freezes once it has applied any
    provider-reported cost (recipe `freeze-at-seam`)."""
    if not is_record(value) or ("input_tokens" not in value and "output_tokens" not in value):
        return None
    total_input = _token_count(value.get("input_tokens"))
    output = _token_count(value.get("output_tokens"))
    raw_details = value.get("input_tokens_details")
    details = raw_details if is_record(raw_details) else {}
    cache_read = min(_token_count(details.get("cached_tokens")), total_input)
    cache_write = min(_token_count(details.get("cache_write_tokens")), total_input - cache_read)
    usage = UsageBuilder(
        input=total_input - cache_read - cache_write,
        output=output,
        cache_read=cache_read,
        cache_write=cache_write,
        total_tokens=total_input + output,
    )
    calculate_cost(model, usage)
    return usage
