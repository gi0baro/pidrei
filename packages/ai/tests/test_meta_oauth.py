"""Mirror of pi's meta-oauth.test.ts."""

from datetime import UTC, datetime

import pytest

from pidrei_ai.auth.oauth.meta import meta_oauth
from pidrei_ai.auth.types import AuthEvent, ModelAuth, OAuthCredential
from pidrei_ai.utils.cancel import CancelToken

from .oauth_helpers import OAuthRequest, RecordingInteraction, json_response, stub_oauth_http, virtual_clock


CLIENT_ID = "1031625952748946"
DEVICE_AUTHORIZATION_URL = "https://auth.meta.com/oidc/device/authorization/"
DEVICE_TOKEN_URL = "https://auth.meta.com/oidc/device/token/"
MINT_URL = "https://api.meta.ai/muse-code/key"
DAY_MS = 24 * 60 * 60 * 1000


def _epoch_ms(iso: str) -> int:
    return int(datetime.fromisoformat(iso).astimezone(UTC).timestamp() * 1000)


@pytest.mark.tonio
async def test_logs_in_with_the_device_flow_and_mints_a_model_api_key():
    start_time = _epoch_ms("2026-09-03T00:00:00Z")
    poll_responses = [
        json_response({"error": "authorization_pending"}, 400),
        json_response({"access_token": "identity-token", "token_type": "Bearer"}),
    ]

    def handler(request: OAuthRequest):
        if request.url == DEVICE_AUTHORIZATION_URL:
            assert request.method == "POST"
            assert request.form["client_id"] == CLIENT_ID
            return json_response(
                {
                    "device_code": "device-code-123",
                    "user_code": "ABCD-1234",
                    "verification_uri": "https://auth.meta.com/oauth/device/",
                    "verification_uri_complete": "https://auth.meta.com/oauth/device/?code=ABCD-1234",
                    "interval": 5,
                    "expires_in": 600,
                }
            )
        if request.url == DEVICE_TOKEN_URL:
            assert request.form["grant_type"] == "urn:ietf:params:oauth:grant-type:device_code"
            assert request.form["client_id"] == CLIENT_ID
            assert request.form["device_code"] == "device-code-123"
            if not poll_responses:
                raise AssertionError("Unexpected extra token poll")
            return poll_responses.pop(0)
        if request.url == MINT_URL:
            assert request.method == "POST"
            assert request.headers["Authorization"] == "Bearer identity-token"
            return json_response({"api_key": "LLM|minted-key"})
        raise AssertionError(f"Unexpected fetch URL: {request.url}")

    interaction = RecordingInteraction()
    with virtual_clock(start_time), stub_oauth_http(handler):
        credential = await meta_oauth.login(interaction)

    assert interaction.events[0] == AuthEvent(
        type="device_code",
        user_code="ABCD-1234",
        verification_uri="https://auth.meta.com/oauth/device/?code=ABCD-1234",
        interval_seconds=5,
        expires_in_seconds=600,
    )
    assert credential == OAuthCredential(
        refresh="identity-token", access="LLM|minted-key", expires=start_time + 10_000 + DAY_MS
    )


@pytest.mark.tonio
async def test_re_mints_the_api_key_from_the_stored_identity_token_on_refresh():
    now = _epoch_ms("2026-09-03T12:00:00Z")

    def handler(request: OAuthRequest):
        assert request.url == MINT_URL
        assert request.headers["Authorization"] == "Bearer identity-token"
        return json_response({"api_key": "LLM|fresh-key"})

    with virtual_clock(now), stub_oauth_http(handler):
        credential = await meta_oauth.refresh(
            OAuthCredential(refresh="identity-token", access="LLM|old-key", expires=1), CancelToken()
        )

    assert credential == OAuthCredential(refresh="identity-token", access="LLM|fresh-key", expires=now + DAY_MS)


@pytest.mark.tonio
async def test_reports_the_setup_url_when_meta_issues_no_key():
    def handler(_request: OAuthRequest):
        return json_response({"require_payment": True, "action_url": "https://dev.meta.ai/billing"})

    with (
        stub_oauth_http(handler),
        pytest.raises(RuntimeError, match="Complete setup at https://dev.meta.ai/billing"),
    ):
        await meta_oauth.refresh(OAuthCredential(refresh="identity-token", access="", expires=1), CancelToken())


@pytest.mark.tonio
async def test_uses_the_minted_key_as_the_request_api_key():
    auth = await meta_oauth.to_auth(OAuthCredential(refresh="identity-token", access="LLM|key", expires=1))
    assert auth == ModelAuth(api_key="LLM|key")
