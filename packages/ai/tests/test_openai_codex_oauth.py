"""Mirror of pi's openai-codex-oauth.test.ts.

pi's last case also asserts `console.error` was never called; there is no
equivalent write path here (the flow raises, it does not log), so that case keeps
only the message assertion.
"""

import base64
import json

import pytest

from pidrei_ai.auth.oauth.openai_codex import openai_codex_oauth
from pidrei_ai.auth.types import AuthPrompt, OAuthCredential
from pidrei_ai.utils import clock
from pidrei_ai.utils.cancel import CancelToken

from .oauth_helpers import (
    DEFAULT_START_MS,
    OAuthRequest,
    RecordingInteraction,
    json_response,
    stub_oauth_http,
    text_response,
    virtual_clock,
)


CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
USER_CODE_URL = "https://auth.openai.com/api/accounts/deviceauth/usercode"
DEVICE_TOKEN_URL = "https://auth.openai.com/api/accounts/deviceauth/token"
TOKEN_URL = "https://auth.openai.com/oauth/token"


def create_access_token(account_id: str) -> str:
    def segment(value: dict) -> str:
        return base64.b64encode(json.dumps(value, separators=(",", ":")).encode()).decode()

    header = segment({"alg": "none"})
    payload = segment({"https://api.openai.com/auth": {"chatgpt_account_id": account_id}})
    return f"{header}.{payload}.signature"


def device_auth_pending_response():
    return json_response(
        {
            "error": {
                "message": "Device authorization is pending. Please try again.",
                "type": "invalid_request_error",
                "param": None,
                "code": "deviceauth_authorization_pending",
            }
        },
        403,
    )


def select_device_code(prompt: AuthPrompt) -> str:
    if prompt.type != "select":
        raise AssertionError(f"Unexpected prompt: {prompt.type}")
    return "device_code"


def device_code_events(interaction: RecordingInteraction) -> list[tuple]:
    return [
        (event.user_code, event.verification_uri, event.interval_seconds, event.expires_in_seconds)
        for event in interaction.events_of("device_code")
    ]


@pytest.mark.tonio
async def test_logs_in_with_the_device_code_flow():
    access_token = create_access_token("account-123")
    poll_times: list[int] = []
    poll_responses = [
        device_auth_pending_response(),
        json_response(
            {
                "authorization_code": "oauth-code",
                "code_challenge": "device-code-challenge",
                "code_verifier": "device-code-verifier",
            }
        ),
    ]

    def handler(request: OAuthRequest):
        if request.url == USER_CODE_URL:
            assert request.method == "POST"
            assert request.headers["Content-Type"] == "application/json"
            assert request.json_body == {"client_id": CLIENT_ID}
            return json_response({"device_auth_id": "device-auth-id", "user_code": "ABCD-1234", "interval": "5"})

        if request.url == DEVICE_TOKEN_URL:
            poll_times.append(clock.now_ms())
            assert request.headers["Content-Type"] == "application/json"
            assert request.json_body == {"device_auth_id": "device-auth-id", "user_code": "ABCD-1234"}
            if not poll_responses:
                raise AssertionError("Unexpected extra device auth poll")
            return poll_responses.pop(0)

        assert request.url == TOKEN_URL
        assert request.headers["Content-Type"] == "application/x-www-form-urlencoded"
        assert request.form["grant_type"] == "authorization_code"
        assert request.form["client_id"] == CLIENT_ID
        assert request.form["code"] == "oauth-code"
        assert request.form["redirect_uri"] == "https://auth.openai.com/deviceauth/callback"
        assert request.form["code_verifier"] == "device-code-verifier"
        return json_response({"access_token": access_token, "refresh_token": "refresh-token", "expires_in": 3600})

    interaction = RecordingInteraction(prompt=select_device_code)
    with virtual_clock(), stub_oauth_http(handler):
        credential = await openai_codex_oauth.login(interaction)

    assert device_code_events(interaction) == [("ABCD-1234", "https://auth.openai.com/codex/device", 5, 900)]
    assert poll_times == [DEFAULT_START_MS, DEFAULT_START_MS + 5000]
    assert credential.access == access_token
    assert credential.refresh == "refresh-token"
    assert credential.expires == DEFAULT_START_MS + 5000 + 3600 * 1000
    assert credential.extra["accountId"] == "account-123"


@pytest.mark.tonio
async def test_offers_browser_login_first_and_uses_the_selected_device_code_flow():
    access_token = create_access_token("account-456")

    def handler(request: OAuthRequest):
        if request.url == USER_CODE_URL:
            assert request.json_body == {"client_id": CLIENT_ID}
            return json_response({"device_auth_id": "device-auth-id", "user_code": "WXYZ-7890", "interval": "5"})
        if request.url == DEVICE_TOKEN_URL:
            return json_response(
                {
                    "authorization_code": "oauth-code",
                    "code_challenge": "device-code-challenge",
                    "code_verifier": "device-code-verifier",
                }
            )
        assert request.url == TOKEN_URL
        return json_response({"access_token": access_token, "refresh_token": "refresh-token", "expires_in": 3600})

    interaction = RecordingInteraction(prompt=select_device_code)
    with virtual_clock(), stub_oauth_http(handler):
        credential = await openai_codex_oauth.login(interaction)

    assert credential.type == "oauth"
    assert credential.access == access_token
    assert credential.refresh == "refresh-token"
    assert credential.extra["accountId"] == "account-456"

    assert len(interaction.prompts) == 1
    prompt = interaction.prompts[0]
    assert prompt.type == "select"
    assert prompt.message == "Select OpenAI Codex login method:"
    assert [(option.id, option.label) for option in prompt.options or []] == [
        ("browser", "Browser login (default)"),
        ("device_code", "Device code login (headless)"),
    ]
    assert not interaction.events_of("auth_url"), "Browser login should not start"
    assert device_code_events(interaction) == [("WXYZ-7890", "https://auth.openai.com/codex/device", 5, 900)]


@pytest.mark.tonio
async def test_cancels_when_login_method_selection_is_cancelled():
    def cancelled_prompt(_prompt: AuthPrompt) -> str:
        raise RuntimeError("Login cancelled")

    with pytest.raises(RuntimeError, match="Login cancelled"):
        await openai_codex_oauth.login(RecordingInteraction(prompt=cancelled_prompt))


@pytest.mark.tonio
async def test_cancels_the_device_code_flow_while_waiting():
    """pi aborts after the first poll lands; cancelling from inside that poll's
    response is the same moment on a virtual clock."""
    cancel = CancelToken()
    poll_times: list[int] = []

    def handler(request: OAuthRequest):
        if request.url == USER_CODE_URL:
            assert request.json_body == {"client_id": CLIENT_ID}
            return json_response({"device_auth_id": "device-auth-id", "user_code": "ABCD-1234", "interval": "5"})
        assert request.url == DEVICE_TOKEN_URL
        poll_times.append(clock.now_ms())
        cancel.cancel()
        return device_auth_pending_response()

    interaction = RecordingInteraction(prompt=select_device_code, cancel=cancel)
    with virtual_clock(), stub_oauth_http(handler), pytest.raises(RuntimeError, match="Login cancelled"):
        await openai_codex_oauth.login(interaction)

    assert len(poll_times) == 1


@pytest.mark.tonio
async def test_times_out_the_device_code_flow_after_fifteen_minutes():
    poll_times: list[int] = []

    def handler(request: OAuthRequest):
        if request.url == USER_CODE_URL:
            return json_response({"device_auth_id": "device-auth-id", "user_code": "ABCD-1234", "interval": "60"})
        poll_times.append(clock.now_ms())
        return device_auth_pending_response()

    interaction = RecordingInteraction(prompt=select_device_code)
    with (
        virtual_clock(),
        stub_oauth_http(handler),
        pytest.raises(RuntimeError, match="^Device flow timed out$"),
    ):
        await openai_codex_oauth.login(interaction)

    # 15 minutes of 60 s polls, the first one immediate.
    assert poll_times == [DEFAULT_START_MS + 60_000 * index for index in range(15)]


@pytest.mark.tonio
async def test_treats_device_auth_403_and_404_responses_as_pending():
    access_token = create_access_token("account-403-404")
    poll_times: list[int] = []
    poll_responses = [
        json_response({"error": "access_denied", "error_description": "denied"}, 403),
        text_response("not ready", 404),
        json_response(
            {
                "authorization_code": "oauth-code",
                "code_challenge": "device-code-challenge",
                "code_verifier": "device-code-verifier",
            }
        ),
    ]

    def handler(request: OAuthRequest):
        if request.url == USER_CODE_URL:
            return json_response({"device_auth_id": "device-auth-id", "user_code": "ABCD-1234", "interval": "1"})
        if request.url == DEVICE_TOKEN_URL:
            poll_times.append(clock.now_ms())
            if not poll_responses:
                raise AssertionError("Unexpected extra device auth poll")
            return poll_responses.pop(0)
        return json_response({"access_token": access_token, "refresh_token": "refresh-token", "expires_in": 3600})

    interaction = RecordingInteraction(prompt=select_device_code)
    with virtual_clock(), stub_oauth_http(handler):
        credential = await openai_codex_oauth.login(interaction)

    assert credential.access == access_token
    assert credential.refresh == "refresh-token"
    assert credential.extra["accountId"] == "account-403-404"
    assert len(poll_times) == 3


@pytest.mark.tonio
async def test_includes_the_response_body_in_device_auth_poll_failures():
    def handler(request: OAuthRequest):
        if request.url == USER_CODE_URL:
            return json_response({"device_auth_id": "device-auth-id", "user_code": "ABCD-1234", "interval": "5"})
        return json_response({"error": "server_error", "error_description": "try again later"}, 500)

    interaction = RecordingInteraction(prompt=select_device_code)
    expected = (
        'OpenAI Codex device auth failed with status 500: {"error": "server_error", '
        '"error_description": "try again later"}'
    )
    with virtual_clock(), stub_oauth_http(handler), pytest.raises(RuntimeError) as error:
        await openai_codex_oauth.login(interaction)

    assert str(error.value) == expected


@pytest.mark.tonio
async def test_reports_token_refresh_failures_through_the_raised_error():
    def handler(_request: OAuthRequest):
        return json_response(
            {
                "error": {
                    "message": "Could not validate your token. Please try signing in again.",
                    "type": "invalid_request_error",
                }
            },
            401,
        )

    with (
        virtual_clock(),
        stub_oauth_http(handler),
        pytest.raises(RuntimeError, match=r"OpenAI Codex token refresh failed \(401\).*Could not validate your token"),
    ):
        await openai_codex_oauth.refresh(
            OAuthCredential(access="invalid-access-token", refresh="invalid-refresh-token", expires=0), None
        )
