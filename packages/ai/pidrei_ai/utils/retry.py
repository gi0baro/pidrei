"""Port of pi's assistant-turn retry policy (packages/ai/src/utils/retry.ts).

Classifies failed assistant messages as transient (retryable) vs deterministic
(quota/billing — fail fast) via pi's accumulated error-string patterns, and
runs whole-turn retries with exponential backoff. The backoff sleep is
interruptible; aborts during backoff are normalized to an aborted
`AssistantMessage` so callers never care when cancellation happened.
"""

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any

from tonio.colored import time as tonio_time

from pidrei_ai.types import AssistantMessage
from pidrei_ai.utils.callbacks import maybe_call
from pidrei_ai.utils.cancel import CancelToken


def _build_provider_error_pattern(patterns: list[str]) -> re.Pattern[str]:
    return re.compile("|".join(patterns), re.IGNORECASE)


_NON_RETRYABLE_PROVIDER_LIMIT_ERROR_PATTERN = _build_provider_error_pattern(
    [
        # OpenCode Go/free-tier limits returned as 429 JSON error types: these are
        # subscription/account limits, not transient throttles.
        "GoUsageLimitError",
        "FreeUsageLimitError",
        # OpenCode Go subscription-limit text.
        "Monthly usage limit reached",
        "available balance",
        # Generic quota/budget/billing exhaustion.
        "insufficient_quota",
        "out of budget",
        "quota exceeded",
        "billing",
    ]
)

_RETRYABLE_PROVIDER_ERROR_PATTERN = _build_provider_error_pattern(
    [
        # Generic provider load, HTTP status, and server-side transient failures.
        "overloaded",
        "rate.?limit",
        "too many requests",
        "429",
        "500",
        "502",
        "503",
        "504",
        "524",
        "service.?unavailable",
        "server.?error",
        "internal.?error",
        # Wrapper/provider text for transient upstream failures.
        "provider.?returned.?error",
        "exceeded request buffer limit while retrying upstream",
        # Network, proxy, and transport failures.
        "network.?error",
        "connection.?error",
        "connection.?refused",
        "connection.?lost",
        "other side closed",
        "fetch failed",
        "getaddrinfo",
        "ENOTFOUND",
        "EAI_AGAIN",
        "upstream.?connect",
        "reset before headers",
        "socket hang up",
        "socket connection was closed",
        "timed? out",
        "timeout",
        "terminated",
        # WebSocket transports can report close/error text instead of HTTP text.
        "websocket.?closed",
        "websocket.?error",
        # Premature stream endings from SDKs and transports.
        "ended without",
        "stream ended before message_stop",
        "stream ended before a terminal response event",
        "http2 request did not get a response",
        # Provider-requested retry delay cap failures flow through the outer
        # policy so callers can surface/abort the backoff.
        "retry delay",
        # Explicit retry guidance emitted mid-stream.
        "you can retry your request",
        "try your request again",
        "please retry your request",
        # gRPC based providers (e.g. NVIDIA NIM)
        "ResourceExhausted",
    ]
)


@dataclass(slots=True)
class RetryPolicy:
    """Bounded attempts with exponential backoff (`base_delay_ms * 2^(attempt-1)`)."""

    enabled: bool
    # Max retry attempts (0 = no retries). The initial call never counts as a retry.
    max_retries: int
    # Base delay in ms; per-attempt delay is `base_delay_ms * 2^(attempt-1)` before jitter.
    base_delay_ms: float


@dataclass(slots=True)
class RetryCallbacks:
    """Optional callbacks emitted by `retry_assistant_call` around each retry.

    Awaitable-returning by contract (async-only callback policy).
    """

    # Before the backoff sleep of each retry attempt (1-indexed):
    # (attempt, max_attempts, delay_ms, error_message)
    on_retry_scheduled: Callable[[int, int, float, str], Awaitable[Any]] | None = None
    # After the backoff sleep, immediately before the retried call starts.
    on_retry_attempt_start: Callable[[], Awaitable[Any]] | None = None
    # Once when the loop ends: (success, attempt, final_error?)
    on_retry_finished: Callable[..., Awaitable[Any]] | None = None


class _RetrySleepAbort(Exception):
    pass


async def _sleep(ms: float, cancel: CancelToken | None) -> None:
    seconds = ms / 1000
    if cancel is None:
        await tonio_time.sleep(seconds)
        return
    if not cancel.cancelled:
        await cancel.wait(seconds)
    if cancel.cancelled:
        raise _RetrySleepAbort()


async def retry_assistant_call(
    produce: Callable[[], Awaitable[AssistantMessage]],
    policy: RetryPolicy | None,
    cancel: CancelToken | None,
    callbacks: RetryCallbacks | None = None,
) -> AssistantMessage:
    """Run a single assistant-producing call with bounded retry on transient errors.

    Aborts are terminal and never retried; aborts during backoff are normalized
    to an aborted `AssistantMessage`. Non-retryable errors (per
    `is_retryable_assistant_error`) fail fast. With no/disabled policy this is
    equivalent to calling `produce()` directly.
    """
    callbacks = callbacks or RetryCallbacks()
    max_attempts = policy.max_retries if policy is not None and policy.enabled else 0

    attempt = 0
    last_retry: tuple[int, str] | None = None
    while True:
        response = await produce()

        # Abort: terminal but not successful. Never retry an aborted message.
        if response.stop_reason == "aborted":
            if last_retry is not None:
                await maybe_call(callbacks.on_retry_finished, False, last_retry[0])
            return response

        # Success: non-error, non-abort responses return as-is.
        if response.stop_reason != "error":
            if last_retry is not None:
                await maybe_call(callbacks.on_retry_finished, True, last_retry[0])
            return response

        # Non-retryable, or budget exhausted: return the final error message.
        if attempt >= max_attempts or not is_retryable_assistant_error(response):
            if last_retry is not None:
                await maybe_call(callbacks.on_retry_finished, False, last_retry[0], response.error_message)
            return response

        attempt += 1
        last_retry = (attempt, response.error_message or "Unknown error")
        delay_ms = policy.base_delay_ms * 2 ** (attempt - 1)  # type: ignore[union-attr]
        await maybe_call(callbacks.on_retry_scheduled, attempt, max_attempts, delay_ms, last_retry[1])

        try:
            await _sleep(delay_ms, cancel)
        except _RetrySleepAbort:
            await maybe_call(callbacks.on_retry_finished, False, attempt, last_retry[1])
            return replace(response, stop_reason="aborted", error_message=None)
        await maybe_call(callbacks.on_retry_attempt_start)


def is_retryable_assistant_error(message: AssistantMessage) -> bool:
    """Classify whether a failed assistant message looks like a transient error."""
    if message.stop_reason != "error" or not message.error_message:
        return False
    error_message = message.error_message
    if _NON_RETRYABLE_PROVIDER_LIMIT_ERROR_PATTERN.search(error_message):
        return False
    return bool(_RETRYABLE_PROVIDER_ERROR_PATTERN.search(error_message))
