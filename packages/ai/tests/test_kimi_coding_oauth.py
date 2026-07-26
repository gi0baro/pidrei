"""Mirror of pi's kimi-coding-oauth.test.ts."""

from typing import Any

import pytest

from pidrei_ai.auth.oauth.kimi_coding import kimi_coding_oauth
from pidrei_ai.auth.types import OAuthCredential
from pidrei_ai.utils import clock

from .oauth_helpers import (
    DEFAULT_START_MS,
    OAuthRequest,
    RecordingInteraction,
    json_response,
    process_env,
    stub_oauth_http,
    virtual_clock,
)


CLIENT_ID = "17e5f671-d194-4dfb-9706-5516cb48c098"
OAUTH_HOST = "https://auth.kimi.com"


def device_authorization_body(**overrides: Any) -> dict[str, Any]:
    return {
        "user_code": "ABCD-1234",
        "device_code": "device-code-123",
        "verification_uri": "https://www.kimi.com/code",
        "verification_uri_complete": "https://www.kimi.com/code?user_code=ABCD-1234",
        "interval": 5,
        "expires_in": 600,
        **overrides,
    }


@pytest.mark.tonio
async def test_logs_in_with_the_device_authorization_flow():
    poll_times: list[int] = []
    poll_responses = [
        json_response({"error": "authorization_pending"}, 400),
        json_response({"access_token": "access-token", "refresh_token": "refresh-token", "expires_in": 3600}),
    ]

    def handler(request: OAuthRequest):
        if request.url == f"{OAUTH_HOST}/api/oauth/device_authorization":
            assert request.method == "POST"
            assert request.headers["Content-Type"] == "application/x-www-form-urlencoded"
            assert request.headers["Accept"] == "application/json"
            assert request.form["client_id"] == CLIENT_ID
            return json_response(device_authorization_body())
        assert request.url == f"{OAUTH_HOST}/api/oauth/token"
        poll_times.append(clock.now_ms())
        assert request.form["grant_type"] == "urn:ietf:params:oauth:grant-type:device_code"
        assert request.form["client_id"] == CLIENT_ID
        assert request.form["device_code"] == "device-code-123"
        if not poll_responses:
            raise AssertionError("Unexpected extra token poll")
        return poll_responses.pop(0)

    interaction = RecordingInteraction()
    with virtual_clock(), stub_oauth_http(handler):
        credential = await kimi_coding_oauth.login(interaction)

    assert [
        (
            event.type,
            event.user_code,
            event.verification_uri,
            event.interval_seconds,
            event.expires_in_seconds,
        )
        for event in interaction.events
    ] == [("device_code", "ABCD-1234", "https://www.kimi.com/code?user_code=ABCD-1234", 5, 600)]

    # waitBeforeFirstPoll: the first poll happens after the 5 s interval.
    assert poll_times == [DEFAULT_START_MS + 5000, DEFAULT_START_MS + 10_000]
    assert credential == OAuthCredential(
        access="access-token",
        refresh="refresh-token",
        expires=DEFAULT_START_MS + 10_000 + 3600 * 1000,
    )


@pytest.mark.tonio
async def test_fails_when_the_device_code_expires():
    def handler(request: OAuthRequest):
        if request.url.endswith("/device_authorization"):
            return json_response(device_authorization_body())
        return json_response({"error": "expired_token"}, 400)

    with virtual_clock(), stub_oauth_http(handler), pytest.raises(RuntimeError, match="expired"):
        await kimi_coding_oauth.login(RecordingInteraction())


@pytest.mark.tonio
async def test_fails_when_the_user_denies_the_login():
    def handler(request: OAuthRequest):
        if request.url.endswith("/device_authorization"):
            return json_response(device_authorization_body())
        return json_response({"error": "access_denied"}, 400)

    with virtual_clock(), stub_oauth_http(handler), pytest.raises(RuntimeError, match="denied"):
        await kimi_coding_oauth.login(RecordingInteraction())


@pytest.mark.tonio
async def test_honors_the_kimi_code_oauth_host_override():
    urls: list[str] = []

    def handler(request: OAuthRequest):
        urls.append(request.url)
        if request.url == "https://auth.example.com/api/oauth/device_authorization":
            return json_response(device_authorization_body(interval=1))
        assert request.url == "https://auth.example.com/api/oauth/token"
        return json_response({"access_token": "a", "refresh_token": "r", "expires_in": 60})

    with (
        process_env(KIMI_CODE_OAUTH_HOST="https://auth.example.com/"),
        virtual_clock(),
        stub_oauth_http(handler),
    ):
        credential = await kimi_coding_oauth.login(RecordingInteraction())

    assert (credential.access, credential.refresh) == ("a", "r")
    assert urls == [
        "https://auth.example.com/api/oauth/device_authorization",
        "https://auth.example.com/api/oauth/token",
    ]


@pytest.mark.tonio
async def test_refreshes_tokens_and_returns_a_bearer_header_for_requests():
    def handler(request: OAuthRequest):
        assert request.url == f"{OAUTH_HOST}/api/oauth/token"
        assert request.form["grant_type"] == "refresh_token"
        assert request.form["refresh_token"] == "old-refresh"
        assert request.form["client_id"] == CLIENT_ID
        return json_response({"access_token": "new-access", "refresh_token": "new-refresh", "expires_in": 3600})

    with virtual_clock(), stub_oauth_http(handler):
        credential = await kimi_coding_oauth.refresh(
            OAuthCredential(access="old-access", refresh="old-refresh", expires=DEFAULT_START_MS), None
        )
        auth = await kimi_coding_oauth.to_auth(credential)

    assert credential == OAuthCredential(
        access="new-access", refresh="new-refresh", expires=DEFAULT_START_MS + 3600 * 1000
    )
    assert credential.expires >= DEFAULT_START_MS + 3600 * 1000
    assert auth.headers == {"Authorization": "Bearer new-access"}


@pytest.mark.tonio
async def test_retries_refresh_on_429_and_fails_unauthorized_on_invalid_grant():
    calls = 0

    def retry_then_succeed(_request: OAuthRequest):
        nonlocal calls
        calls += 1
        if calls == 1:
            return json_response({"error": "temporarily_unavailable"}, 429)
        return json_response({"access_token": "a", "refresh_token": "r", "expires_in": 60})

    with virtual_clock(), stub_oauth_http(retry_then_succeed):
        credential = await kimi_coding_oauth.refresh(OAuthCredential(access="old", refresh="old", expires=0), None)

    assert credential.access == "a"
    assert calls == 2

    # invalid_grant is not retried.
    def invalid_grant(_request: OAuthRequest):
        return json_response({"error": "invalid_grant"}, 400)

    with (
        virtual_clock(),
        stub_oauth_http(invalid_grant) as invalid_calls,
        pytest.raises(RuntimeError, match="unauthorized"),
    ):
        await kimi_coding_oauth.refresh(OAuthCredential(access="old", refresh="old", expires=0), None)

    assert len(invalid_calls) == 1
