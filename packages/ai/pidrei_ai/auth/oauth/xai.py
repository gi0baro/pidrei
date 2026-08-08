"""Port of pi's xAI device-code flow (packages/ai/src/auth/oauth/xai.ts)."""

import math
from dataclasses import dataclass
from typing import Any

from pidrei_ai.auth.oauth import http as oauth_http
from pidrei_ai.auth.oauth.device_code import OAuthDeviceCodePollResult, poll_oauth_device_code_flow
from pidrei_ai.auth.oauth.urls import https_url
from pidrei_ai.auth.types import AuthEvent, ModelAuth, OAuthAuth, OAuthCredential, ProviderAuthInteraction
from pidrei_ai.utils import clock
from pidrei_ai.utils.cancel import AbortError, CancelToken


XAI_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
XAI_SCOPE = "openid profile email offline_access grok-cli:access api:access"
XAI_DEVICE_CODE_URL = "https://auth.x.ai/oauth2/device/code"
XAI_TOKEN_URL = "https://auth.x.ai/oauth2/token"  # noqa: S105 - an endpoint, not a secret
# Refresh slightly before the reported expiry to avoid using a token that dies mid-request.
REFRESH_SKEW_MS = 5 * 60 * 1000
DEFAULT_TOKEN_LIFETIME_SECONDS = 3600


_FORM_HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/x-www-form-urlencoded",
}


@dataclass(slots=True)
class _XaiDeviceCode:
    device_code: str
    user_code: str
    verification_uri: str
    expires_in_seconds: float
    verification_uri_complete: str | None = None
    interval_seconds: float | None = None


def _required_string(body: dict[str, Any], field: str) -> str:
    value = body.get(field)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Invalid xAI OAuth response field: {field}")
    return value


def _positive_number(body: dict[str, Any], field: str) -> float:
    value = body.get(field)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"Invalid xAI OAuth response field: {field}")
    return value


def _validate_verification_uri(raw: str) -> str:
    """The verification URI is opened in the user's browser; force it to be an https URL
    so a malicious response cannot make `open` launch something else."""
    url = https_url(raw)
    if url is None:
        raise RuntimeError("Untrusted verification URI in xAI OAuth response")
    return url


async def _post_form(url: str, fields: dict[str, str], cancel: CancelToken) -> oauth_http.OAuthHttpResponse:
    try:
        return await oauth_http.request(url, form=fields, headers=_FORM_HEADERS, cancel=cancel)
    except AbortError:
        raise RuntimeError("Login cancelled") from None


def _response_body(response: oauth_http.OAuthHttpResponse, cancel: CancelToken) -> dict[str, Any]:
    try:
        parsed = response.json()
    except ValueError:
        if cancel.cancelled:
            raise RuntimeError("Login cancelled") from None
        raise RuntimeError(f"xAI OAuth returned invalid JSON (HTTP {response.status})") from None
    return parsed if isinstance(parsed, dict) else {}


def _request_failure(action: str, status: int, body: dict[str, Any]) -> RuntimeError:
    error = body.get("error") if isinstance(body.get("error"), str) else None
    description = body.get("error_description") if isinstance(body.get("error_description"), str) else None
    detail = ": ".join(part for part in (error, description) if part)
    return RuntimeError(f"xAI OAuth {action} failed (HTTP {status}){f': {detail}' if detail else ''}")


def _parse_device_code(body: dict[str, Any]) -> _XaiDeviceCode:
    # RFC 8628 allows interval 0 (no minimum wait); fall back to the poller's
    # default instead of failing on non-positive or malformed values.
    interval = body.get("interval")
    interval_seconds = (
        interval
        if not isinstance(interval, bool)
        and isinstance(interval, int | float)
        and math.isfinite(interval)
        and interval > 0
        else None
    )
    raw_complete = body.get("verification_uri_complete")
    verification_uri_complete = (
        _validate_verification_uri(raw_complete) if isinstance(raw_complete, str) and raw_complete else None
    )
    return _XaiDeviceCode(
        device_code=_required_string(body, "device_code"),
        user_code=_required_string(body, "user_code"),
        verification_uri=_validate_verification_uri(_required_string(body, "verification_uri")),
        verification_uri_complete=verification_uri_complete,
        interval_seconds=interval_seconds,
        expires_in_seconds=_positive_number(body, "expires_in"),
    )


def _credentials_from_token_response(
    body: dict[str, Any], previous_refresh_token: str | None = None
) -> OAuthCredential:
    access = _required_string(body, "access_token")
    # xAI may omit refresh_token on refresh when the token is not rotated.
    refresh = (
        previous_refresh_token
        if "refresh_token" not in body and previous_refresh_token
        else _required_string(body, "refresh_token")
    )
    expires_in_seconds = (
        DEFAULT_TOKEN_LIFETIME_SECONDS if "expires_in" not in body else _positive_number(body, "expires_in")
    )
    return OAuthCredential(
        access=access,
        refresh=refresh,
        expires=int(clock.now_ms() + expires_in_seconds * 1000 - REFRESH_SKEW_MS),
    )


async def _request_device_code(cancel: CancelToken) -> _XaiDeviceCode:
    response = await _post_form(
        XAI_DEVICE_CODE_URL,
        {"client_id": XAI_CLIENT_ID, "scope": XAI_SCOPE, "referrer": "pi"},
        cancel,
    )
    body = _response_body(response, cancel)
    if not response.ok:
        raise _request_failure("device authorization", response.status, body)
    return _parse_device_code(body)


async def _poll_for_tokens(device: _XaiDeviceCode, cancel: CancelToken) -> OAuthCredential:
    async def poll() -> OAuthDeviceCodePollResult:
        response = await _post_form(
            XAI_TOKEN_URL,
            {
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "client_id": XAI_CLIENT_ID,
                "device_code": device.device_code,
            },
            cancel,
        )
        body = _response_body(response, cancel)

        if response.ok:
            return OAuthDeviceCodePollResult(status="complete", value=_credentials_from_token_response(body))

        error = body.get("error")
        if error == "authorization_pending":
            return OAuthDeviceCodePollResult(status="pending")
        if error == "slow_down":
            interval = body.get("interval")
            return OAuthDeviceCodePollResult(
                status="slow_down",
                interval_seconds=interval
                if isinstance(interval, int | float) and not isinstance(interval, bool)
                else None,
            )
        if error in ("access_denied", "authorization_denied"):
            return OAuthDeviceCodePollResult(status="failed", message="xAI device authorization was denied")
        if error == "expired_token":
            return OAuthDeviceCodePollResult(status="failed", message="xAI device code expired")
        return OAuthDeviceCodePollResult(
            status="failed", message=str(_request_failure("device token polling", response.status, body))
        )

    return await poll_oauth_device_code_flow(
        poll=poll,
        interval_seconds=device.interval_seconds,
        expires_in_seconds=device.expires_in_seconds,
        wait_before_first_poll=True,
        cancel=cancel,
    )


async def _login_xai(interaction: ProviderAuthInteraction) -> OAuthCredential:
    device = await _request_device_code(interaction.cancel)
    interaction.notify(
        AuthEvent(
            type="device_code",
            user_code=device.user_code,
            verification_uri=device.verification_uri_complete or device.verification_uri,
            interval_seconds=device.interval_seconds,
            expires_in_seconds=device.expires_in_seconds,
        )
    )
    return await _poll_for_tokens(device, interaction.cancel)


async def _refresh_xai_token(refresh_token: str, cancel: CancelToken) -> OAuthCredential:
    response = await _post_form(
        XAI_TOKEN_URL,
        {"grant_type": "refresh_token", "client_id": XAI_CLIENT_ID, "refresh_token": refresh_token},
        cancel,
    )
    body = _response_body(response, cancel)
    if not response.ok:
        raise _request_failure("token refresh", response.status, body)
    return _credentials_from_token_response(body, refresh_token)


async def _refresh(credential: OAuthCredential, cancel: CancelToken) -> OAuthCredential:
    return await _refresh_xai_token(credential.refresh, cancel)


async def _to_auth(credential: OAuthCredential) -> ModelAuth:
    return ModelAuth(api_key=credential.access)


xai_oauth = OAuthAuth(
    name="xAI (Grok/X subscription)",
    is_subscription=True,
    login_label="Sign in with SuperGrok or X Premium",
    login=_login_xai,
    refresh=_refresh,
    to_auth=_to_auth,
)
