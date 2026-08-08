"""Mirror of pi's oauth-device-code.test.ts.

pi advances fake timers by hand and asserts what has not happened yet; the
virtual clock advances itself, so each case asserts the recorded poll times,
which pin the same intervals (see `oauth_helpers`).
"""

import pytest

from pidrei_ai.auth.oauth.device_code import OAuthDeviceCodePollResult, poll_oauth_device_code_flow
from pidrei_ai.utils import clock
from pidrei_ai.utils.cancel import CancelToken

from .oauth_helpers import DEFAULT_START_MS, virtual_clock


@pytest.mark.tonio
async def test_polls_immediately_and_returns_the_completed_value():
    poll_times: list[int] = []

    async def poll() -> OAuthDeviceCodePollResult:
        poll_times.append(clock.now_ms())
        if len(poll_times) == 1:
            return OAuthDeviceCodePollResult(status="pending")
        return OAuthDeviceCodePollResult(status="complete", value="token")

    with virtual_clock():
        result = await poll_oauth_device_code_flow(
            poll=poll, cancel=CancelToken(), interval_seconds=2, expires_in_seconds=30
        )

    assert result == "token"
    assert poll_times == [DEFAULT_START_MS, DEFAULT_START_MS + 2000]


@pytest.mark.tonio
async def test_can_wait_before_the_first_poll():
    poll_times: list[int] = []

    async def poll() -> OAuthDeviceCodePollResult:
        poll_times.append(clock.now_ms())
        return OAuthDeviceCodePollResult(status="complete", value="token")

    with virtual_clock():
        result = await poll_oauth_device_code_flow(
            poll=poll, cancel=CancelToken(), interval_seconds=2, expires_in_seconds=30, wait_before_first_poll=True
        )

    assert result == "token"
    assert poll_times == [DEFAULT_START_MS + 2000]


@pytest.mark.tonio
async def test_increases_the_interval_by_five_seconds_after_slow_down_without_a_server_interval():
    poll_times: list[int] = []
    results = [
        OAuthDeviceCodePollResult(status="slow_down"),
        OAuthDeviceCodePollResult(status="complete", value="token"),
    ]

    async def poll() -> OAuthDeviceCodePollResult:
        poll_times.append(clock.now_ms())
        if not results:
            raise AssertionError("Unexpected extra poll")
        return results.pop(0)

    with virtual_clock():
        result = await poll_oauth_device_code_flow(
            poll=poll, cancel=CancelToken(), interval_seconds=2, expires_in_seconds=900
        )

    assert result == "token"
    # 2 s interval + the RFC 8628 5 s increment.
    assert poll_times == [DEFAULT_START_MS, DEFAULT_START_MS + 7000]


@pytest.mark.tonio
async def test_honors_a_server_provided_slow_down_interval():
    poll_times: list[int] = []
    results = [
        OAuthDeviceCodePollResult(status="slow_down", interval_seconds=30),
        OAuthDeviceCodePollResult(status="complete", value="token"),
    ]

    async def poll() -> OAuthDeviceCodePollResult:
        poll_times.append(clock.now_ms())
        if not results:
            raise AssertionError("Unexpected extra poll")
        return results.pop(0)

    with virtual_clock():
        result = await poll_oauth_device_code_flow(
            poll=poll, cancel=CancelToken(), interval_seconds=2, expires_in_seconds=900
        )

    assert result == "token"
    assert poll_times == [DEFAULT_START_MS, DEFAULT_START_MS + 30000]


@pytest.mark.tonio
async def test_cancels_an_in_flight_wait():
    """pi aborts between creating the promise and the first sleep; the token is
    cancelled from inside the first poll here, which lands in the same place."""
    cancel = CancelToken()

    async def poll() -> OAuthDeviceCodePollResult:
        cancel.cancel()
        return OAuthDeviceCodePollResult(status="pending")

    with virtual_clock(), pytest.raises(RuntimeError, match="Login cancelled"):
        await poll_oauth_device_code_flow(poll=poll, interval_seconds=5, expires_in_seconds=30, cancel=cancel)


@pytest.mark.tonio
async def test_times_out_and_reports_slow_down_when_one_was_seen():
    async def pending() -> OAuthDeviceCodePollResult:
        return OAuthDeviceCodePollResult(status="pending")

    with virtual_clock(), pytest.raises(RuntimeError, match="^Device flow timed out$"):
        await poll_oauth_device_code_flow(poll=pending, cancel=CancelToken(), interval_seconds=5, expires_in_seconds=12)

    async def slow_down() -> OAuthDeviceCodePollResult:
        return OAuthDeviceCodePollResult(status="slow_down")

    with virtual_clock(), pytest.raises(RuntimeError, match="one or more slow_down responses"):
        await poll_oauth_device_code_flow(
            poll=slow_down, cancel=CancelToken(), interval_seconds=5, expires_in_seconds=12
        )


@pytest.mark.tonio
async def test_a_failed_poll_raises_its_message():
    async def poll() -> OAuthDeviceCodePollResult:
        return OAuthDeviceCodePollResult(status="failed", message="denied")

    with virtual_clock(), pytest.raises(RuntimeError, match="^denied$"):
        await poll_oauth_device_code_flow(poll=poll, cancel=CancelToken(), interval_seconds=1, expires_in_seconds=30)
