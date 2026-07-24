"""Port of pi's HTTP-level retry (packages/ai/src/utils/provider-retry.ts).

Reproduces the retry behavior of the OpenAI/Anthropic SDKs (pi pins it here so
SDK backoff timers can't ignore the abort signal): honor `x-should-retry`,
retry 408/409/429/5xx and connection-level failures, respect
`retry-after(-ms)` headers, and cap server-requested delays at
`max_retry_delay_ms` (60s default; 0 disables the cap).

The backoff sleep is interruptible: it races tonio's sleep against the
`CancelToken` with `tonio.select`.
"""

import math
import random
from collections.abc import Awaitable, Callable
from email.utils import parsedate_to_datetime
from typing import Any

import tonio.colored as tonio
from tonio.colored import time as tonio_time

from pppi_ai.utils.cancel import AbortError, CancelToken


DEFAULT_MAX_RETRY_DELAY_MS = 60_000

_SLEPT = object()
_CANCELLED = object()


def _create_abort_error() -> AbortError:
    return AbortError("Request aborted")


async def abortable_sleep(ms: float, cancel: CancelToken | None = None) -> None:
    if cancel is not None and cancel.cancelled:
        raise _create_abort_error()

    seconds = max(0.0, ms) / 1000

    if cancel is None:
        await tonio_time.sleep(seconds)
        return

    async def _sleep() -> object:
        await tonio_time.sleep(seconds)
        return _SLEPT

    async def _aborted() -> object:
        await cancel.wait()
        return _CANCELLED

    winner = await tonio.select(_sleep(), _aborted())
    if winner is _CANCELLED:
        raise _create_abort_error()


def _headers_get(headers: Any, name: str) -> str | None:
    if headers is None:
        return None
    value = headers.get(name)
    if value is None:
        # Case-insensitive fallback for plain-dict header carriers.
        lowered = name.lower()
        for key in headers:
            if key.lower() == lowered:
                return headers[key]
    return value


def _is_provider_error(error: Any) -> bool:
    if not isinstance(error, BaseException):
        return False
    if not hasattr(error, "status") or not hasattr(error, "headers"):
        return False
    status = error.status
    return status is None or (isinstance(status, int) and not isinstance(status, bool))


def _is_retryable_provider_error(error: Any) -> bool:
    """Mirrors the pinned OpenAI/Anthropic SDK retry policy."""
    should_retry = _headers_get(error.headers, "x-should-retry")
    if should_retry == "true":
        return True
    if should_retry == "false":
        return False

    if error.status is None:
        return True
    return error.status in (408, 409, 429) or error.status >= 500


def _validate_server_retry_delay_ms(
    delay_ms: float,
    max_retry_delay_ms: float | None,
    provider_error_message: str,
) -> float:
    max_delay_ms = max_retry_delay_ms if max_retry_delay_ms is not None else DEFAULT_MAX_RETRY_DELAY_MS
    if max_delay_ms > 0 and delay_ms > max_delay_ms:
        raise RuntimeError(
            f"Server requested {math.ceil(delay_ms / 1000)}s retry delay "
            f"(max: {math.ceil(max_delay_ms / 1000)}s). {provider_error_message}"
        )
    return delay_ms


def _parse_http_date_delay_ms(value: str) -> float | None:
    try:
        target = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    import time as _time

    return target.timestamp() * 1000 - _time.time() * 1000


def _get_retry_delay_ms(error: Any, retry_index: int, max_retry_delay_ms: float | None) -> float:
    message = str(error)

    retry_after_ms = _headers_get(error.headers, "retry-after-ms")
    if retry_after_ms:
        try:
            return _validate_server_retry_delay_ms(float(retry_after_ms), max_retry_delay_ms, message)
        except ValueError:
            pass

    retry_after = _headers_get(error.headers, "retry-after")
    if retry_after:
        try:
            delay_ms = float(retry_after) * 1000
        except ValueError:
            parsed = _parse_http_date_delay_ms(retry_after)
            delay_ms = parsed if parsed is not None else float("nan")
        if not math.isnan(delay_ms):
            return _validate_server_retry_delay_ms(delay_ms, max_retry_delay_ms, message)

    exponential_delay = min(0.5 * 2**retry_index, 8) * 1000
    return exponential_delay * (1 - random.random() * 0.25)  # noqa: S311


async def retry_provider_request[T](
    request: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 0,
    max_retry_delay_ms: float | None = None,
    cancel: CancelToken | None = None,
) -> T:
    """Run `request` with SDK-compatible, interruptible retry."""
    retries_remaining = max_retries

    while True:
        try:
            return await request()
        except BaseException as error:
            if cancel is not None and cancel.cancelled:
                raise _create_abort_error() from error
            if retries_remaining <= 0 or not _is_provider_error(error) or not _is_retryable_provider_error(error):
                raise

            retry_index = max_retries - retries_remaining
            retries_remaining -= 1
            await abortable_sleep(_get_retry_delay_ms(error, retry_index, max_retry_delay_ms), cancel)
