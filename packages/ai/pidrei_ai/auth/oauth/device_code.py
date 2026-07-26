"""Port of pi's RFC 8628 device-code poller (packages/ai/src/auth/oauth/device-code.ts).

pi's discriminated poll result collapses into one dataclass with a `status` tag,
the same shape `AuthEvent` uses. Time goes through `utils/clock.py` so the tests
can drive the interval arithmetic on a virtual clock (pi uses fake timers).
"""

import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from pidrei_ai.utils import clock
from pidrei_ai.utils.cancel import AbortError, CancelToken


CANCEL_MESSAGE = "Login cancelled"
TIMEOUT_MESSAGE = "Device flow timed out"
SLOW_DOWN_TIMEOUT_MESSAGE = (
    "Device flow timed out after one or more slow_down responses. This is often caused by clock drift in WSL or VM "
    "environments. Please sync or restart the VM clock and try again."
)
MINIMUM_INTERVAL_MS = 1000
# RFC 8628 section 3.2: if the authorization server omits `interval`, the client must use 5 seconds.
DEFAULT_POLL_INTERVAL_SECONDS = 5
# RFC 8628 section 3.5: `slow_down` means the polling interval must increase by 5 seconds.
SLOW_DOWN_INTERVAL_INCREMENT_MS = 5000


@dataclass(slots=True)
class OAuthDeviceCodePollResult:
    """One poll outcome. `value` is set for "complete", `message` for "failed",
    `interval_seconds` for a server-provided "slow_down" interval."""

    status: Literal["pending", "slow_down", "failed", "complete"]
    value: Any = None
    message: str | None = None
    interval_seconds: float | None = None


async def _abortable_sleep(ms: float, cancel: CancelToken | None, cancel_message: str) -> None:
    try:
        await clock.sleep_ms(ms, cancel)
    except AbortError:
        raise RuntimeError(cancel_message) from None


def _positive_interval_ms(interval_seconds: float | None) -> float | None:
    """`interval_seconds` clamped to the minimum, or None when unusable.

    pi requires a finite number greater than zero before trusting a
    server-provided interval; anything else falls back to the caller's value.
    """
    if interval_seconds is None or not math.isfinite(interval_seconds) or interval_seconds <= 0:
        return None
    return max(MINIMUM_INTERVAL_MS, math.floor(interval_seconds * 1000))


async def poll_oauth_device_code_flow(
    *,
    poll: Callable[[], Awaitable[OAuthDeviceCodePollResult]],
    interval_seconds: float | None = None,
    expires_in_seconds: float | None = None,
    wait_before_first_poll: bool = False,
    cancel: CancelToken | None = None,
) -> Any:
    """Poll `poll` until it completes, fails, or the device code expires."""
    deadline = clock.now_ms() + expires_in_seconds * 1000 if expires_in_seconds is not None else math.inf
    interval_ms = max(
        MINIMUM_INTERVAL_MS,
        math.floor((interval_seconds if interval_seconds is not None else DEFAULT_POLL_INTERVAL_SECONDS) * 1000),
    )

    slow_down_responses = 0
    if wait_before_first_poll:
        remaining_ms = deadline - clock.now_ms()
        if remaining_ms > 0:
            await _abortable_sleep(min(interval_ms, remaining_ms), cancel, CANCEL_MESSAGE)

    while clock.now_ms() < deadline:
        if cancel is not None and cancel.cancelled:
            raise RuntimeError(CANCEL_MESSAGE)

        result = await poll()
        if result.status == "complete":
            return result.value
        if result.status == "failed":
            raise RuntimeError(result.message)
        if result.status == "slow_down":
            slow_down_responses += 1
            # Use the server-provided interval when given (GitHub reports the new required minimum
            # in `interval`); trusting only a client-tracked value risks polling early forever under
            # WSL/VM clock drift. Otherwise apply RFC 8628 section 3.5: increase by 5 seconds.
            server_interval_ms = _positive_interval_ms(result.interval_seconds)
            interval_ms = (
                server_interval_ms
                if server_interval_ms is not None
                else max(MINIMUM_INTERVAL_MS, interval_ms + SLOW_DOWN_INTERVAL_INCREMENT_MS)
            )

        remaining_ms = deadline - clock.now_ms()
        if remaining_ms <= 0:
            break

        await _abortable_sleep(min(interval_ms, remaining_ms), cancel, CANCEL_MESSAGE)

    raise RuntimeError(SLOW_DOWN_TIMEOUT_MESSAGE if slow_down_responses > 0 else TIMEOUT_MESSAGE)
