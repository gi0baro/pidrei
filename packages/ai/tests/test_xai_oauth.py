"""Mirror of pi's xai-oauth.test.ts."""

from typing import Any

import pytest

from pidrei_ai.auth.oauth.xai import xai_oauth
from pidrei_ai.auth.types import AuthEvent, OAuthCredential
from pidrei_ai.utils import clock
from pidrei_ai.utils.cancel import CancelToken

from .oauth_helpers import (
    DEFAULT_START_MS,
    OAuthRequest,
    RecordingInteraction,
    json_response,
    stub_oauth_http,
    virtual_clock,
)


CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
DEVICE_CODE_URL = "https://auth.x.ai/oauth2/device/code"
TOKEN_URL = "https://auth.x.ai/oauth2/token"


def device_code_body(**overrides: Any) -> dict[str, Any]:
    return {
        "device_code": "device-code",
        "user_code": "ABCD-1234",
        "verification_uri": "https://accounts.x.ai/oauth2/device",
        "expires_in": 900,
        "interval": 5,
        **overrides,
    }


def token_body(**overrides: Any) -> dict[str, Any]:
    body = {
        "access_token": "access-token",
        "refresh_token": "refresh-token",
        "expires_in": 21_600,
        "token_type": "Bearer",
    }
    body.update(overrides)
    return {key: value for key, value in body.items() if value is not _ABSENT}


class _Absent:
    """`{ refresh_token: undefined }`: the key is not sent at all."""


_ABSENT = _Absent()


def device_code_events(interaction: RecordingInteraction) -> list[dict[str, Any]]:
    return [
        {
            "user_code": event.user_code,
            "verification_uri": event.verification_uri,
            "interval_seconds": event.interval_seconds,
            "expires_in_seconds": event.expires_in_seconds,
        }
        for event in interaction.events_of("device_code")
    ]


async def refresh_for_test(refresh_token: str) -> OAuthCredential:
    return await xai_oauth.refresh(OAuthCredential(access="old-access", refresh=refresh_token, expires=0), None)


@pytest.mark.tonio
async def test_uses_the_device_grant_delays_polling_and_handles_pending_and_slow_down():
    poll_times: list[int] = []
    token_replies = [
        json_response({"error": "authorization_pending"}, 400),
        json_response({"error": "slow_down", "interval": 10}, 400),
        json_response(token_body()),
    ]

    def handler(request: OAuthRequest):
        if request.url == DEVICE_CODE_URL:
            assert request.form["client_id"] == CLIENT_ID
            assert request.form["scope"] == "openid profile email offline_access grok-cli:access api:access"
            assert request.form["referrer"] == "pi"
            return json_response(device_code_body())
        assert request.url == TOKEN_URL
        poll_times.append(clock.now_ms())
        assert request.form["grant_type"] == "urn:ietf:params:oauth:grant-type:device_code"
        assert request.form["client_id"] == CLIENT_ID
        assert request.form["device_code"] == "device-code"
        if not token_replies:
            raise AssertionError("Unexpected token poll")
        return token_replies.pop(0)

    interaction = RecordingInteraction()
    with virtual_clock(), stub_oauth_http(handler):
        credential = await xai_oauth.login(interaction)

    assert device_code_events(interaction) == [
        {
            "user_code": "ABCD-1234",
            "verification_uri": "https://accounts.x.ai/oauth2/device",
            "interval_seconds": 5,
            "expires_in_seconds": 900,
        }
    ]
    # 5 s wait before the first poll, 5 s interval, then the server's 10 s slow_down.
    assert poll_times == [DEFAULT_START_MS + 5000, DEFAULT_START_MS + 10_000, DEFAULT_START_MS + 20_000]
    assert credential == OAuthCredential(
        access="access-token",
        refresh="refresh-token",
        expires=DEFAULT_START_MS + 20_000 + 21_600_000 - 300_000,
    )


@pytest.mark.tonio
async def test_falls_back_to_the_default_poll_interval_when_the_response_reports_interval_zero():
    poll_times: list[int] = []

    def handler(request: OAuthRequest):
        if request.url == DEVICE_CODE_URL:
            return json_response(device_code_body(interval=0))
        poll_times.append(clock.now_ms())
        return json_response(token_body())

    with virtual_clock(), stub_oauth_http(handler):
        await xai_oauth.login(RecordingInteraction())

    # RFC 8628 default interval is 5 seconds when the server does not require a wait.
    assert poll_times == [DEFAULT_START_MS + 5000]


@pytest.mark.tonio
async def test_prefers_verification_uri_complete_when_the_server_provides_it():
    def handler(request: OAuthRequest):
        if request.url == DEVICE_CODE_URL:
            return json_response(
                device_code_body(verification_uri_complete="https://accounts.x.ai/oauth2/device?user_code=ABCD-1234")
            )
        return json_response(token_body())

    interaction = RecordingInteraction()
    with virtual_clock(), stub_oauth_http(handler):
        await xai_oauth.login(interaction)

    assert device_code_events(interaction) == [
        {
            "user_code": "ABCD-1234",
            "verification_uri": "https://accounts.x.ai/oauth2/device?user_code=ABCD-1234",
            "interval_seconds": 5,
            "expires_in_seconds": 900,
        }
    ]


@pytest.mark.tonio
async def test_rejects_a_non_https_verification_uri_complete():
    def handler(_request: OAuthRequest):
        return json_response(
            device_code_body(verification_uri_complete="http://accounts.x.ai/oauth2/device?user_code=ABCD-1234")
        )

    with (
        virtual_clock(),
        stub_oauth_http(handler),
        pytest.raises(RuntimeError, match="Untrusted verification URI"),
    ):
        await xai_oauth.login(RecordingInteraction())


@pytest.mark.tonio
@pytest.mark.parametrize("verification_uri", ["http://accounts.x.ai/oauth2/device", "file:///etc/passwd", "not a url"])
async def test_rejects_a_non_https_verification_uri(verification_uri):
    def handler(_request: OAuthRequest):
        return json_response(device_code_body(verification_uri=verification_uri))

    with (
        virtual_clock(),
        stub_oauth_http(handler),
        pytest.raises(RuntimeError, match="Untrusted verification URI"),
    ):
        await xai_oauth.login(RecordingInteraction())


@pytest.mark.tonio
@pytest.mark.parametrize("error", ["access_denied", "authorization_denied"])
async def test_fails_when_device_authorization_is_denied(error):
    requests = 0

    def handler(_request: OAuthRequest):
        nonlocal requests
        requests += 1
        if requests == 1:
            return json_response(device_code_body(interval=1))
        return json_response({"error": error}, 400)

    with (
        virtual_clock(),
        stub_oauth_http(handler),
        pytest.raises(RuntimeError, match="xAI device authorization was denied"),
    ):
        await xai_oauth.login(RecordingInteraction())


@pytest.mark.tonio
async def test_cancels_while_waiting_for_the_first_token_poll():
    cancel = CancelToken()

    def handler(_request: OAuthRequest):
        return json_response(device_code_body())

    interaction = RecordingInteraction(cancel=cancel)
    original_notify = interaction.notify

    def notify(event: AuthEvent) -> None:
        original_notify(event)
        cancel.cancel()

    interaction.notify = notify  # type: ignore[method-assign]

    with virtual_clock(), stub_oauth_http(handler) as calls, pytest.raises(RuntimeError, match="Login cancelled"):
        await xai_oauth.login(interaction)

    assert len(calls) == 1


@pytest.mark.tonio
async def test_refreshes_tokens_and_preserves_an_unrotated_refresh_token():
    requests = 0

    def handler(request: OAuthRequest):
        nonlocal requests
        assert request.url == TOKEN_URL
        assert request.form["grant_type"] == "refresh_token"
        assert request.form["client_id"] == CLIENT_ID
        requests += 1
        if requests == 1:
            assert request.form["refresh_token"] == "old-refresh"
            return json_response(token_body(access_token="new-access", refresh_token="new-refresh"))
        assert request.form["refresh_token"] == "keep-refresh"
        return json_response(token_body(access_token="newer-access", refresh_token=_ABSENT))

    with virtual_clock(), stub_oauth_http(handler):
        rotated = await refresh_for_test("old-refresh")
        preserved = await refresh_for_test("keep-refresh")

    assert rotated.type == "oauth"
    assert rotated.refresh == "new-refresh"
    assert rotated.access == "new-access"
    assert preserved.refresh == "keep-refresh"
    assert preserved.access == "newer-access"
    assert xai_oauth.name == "xAI (Grok/X subscription)"
    assert (await xai_oauth.to_auth(preserved)).api_key == "newer-access"


@pytest.mark.tonio
async def test_assumes_a_one_hour_lifetime_when_expires_in_is_missing():
    def handler(_request: OAuthRequest):
        return json_response(token_body(expires_in=_ABSENT))

    with virtual_clock(), stub_oauth_http(handler):
        credential = await refresh_for_test("old-refresh")

    assert credential.expires == DEFAULT_START_MS + 3_600_000 - 300_000


@pytest.mark.tonio
async def test_rejects_token_responses_with_missing_fields():
    def handler(_request: OAuthRequest):
        return json_response(token_body(access_token=_ABSENT))

    with (
        virtual_clock(),
        stub_oauth_http(handler),
        pytest.raises(RuntimeError, match="Invalid xAI OAuth response field: access_token"),
    ):
        await refresh_for_test("old-refresh")


@pytest.mark.tonio
async def test_surfaces_the_upstream_error_code_and_description_on_refresh_failure():
    def handler(_request: OAuthRequest):
        return json_response({"error": "invalid_grant", "error_description": "refresh token revoked"}, 400)

    with (
        virtual_clock(),
        stub_oauth_http(handler),
        pytest.raises(
            RuntimeError,
            match=r"xAI OAuth token refresh failed \(HTTP 400\): invalid_grant: refresh token revoked",
        ),
    ):
        await refresh_for_test("old-refresh")
