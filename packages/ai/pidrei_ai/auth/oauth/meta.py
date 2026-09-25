"""Port of pi's Meta Model API flow (packages/ai/src/auth/oauth/meta.ts).

RFC 8628 device authorization grant against https://auth.meta.com (JSON
responses). Meta splits identity from API access: the resulting identity token
is not accepted for inference, so it is exchanged for a Model API key via the
Muse Code key-mint endpoint (minted keys live about a day). The identity token
is stored as `refresh` and the minted key as `access`, so the standard OAuth
scheduler re-mints the key when it expires with no bespoke renewal machinery.
The identity token itself is not renewable (auth.meta.com answers
grant_type=refresh_token with 404 and issues no refresh_token), so a 401/403
from mint means the session is dead and the user must sign in again.
"""

import json as json_module
import math
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any

from pidrei_ai.auth.oauth import http as oauth_http
from pidrei_ai.auth.oauth.device_code import OAuthDeviceCodePollResult, poll_oauth_device_code_flow
from pidrei_ai.auth.oauth.urls import http_or_https_url
from pidrei_ai.auth.types import AuthEvent, ModelAuth, OAuthAuth, OAuthCredential, ProviderAuthInteraction
from pidrei_ai.utils import clock
from pidrei_ai.utils.cancel import CancelToken


# Muse Code CLI client id.
CLIENT_ID = "1031625952748946"
AUTH_HOST = "https://auth.meta.com"
DEVICE_AUTHORIZATION_URL = f"{AUTH_HOST}/oidc/device/authorization/"
DEVICE_TOKEN_URL = f"{AUTH_HOST}/oidc/device/token/"
API_KEY_MINT_URL = "https://api.meta.ai/muse-code/key"
API_KEY_LIFETIME_MS = 24 * 60 * 60 * 1000
REQUEST_TIMEOUT_MS = 30 * 1000

_FORM_HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded",
    "Accept": "application/json",
}


@dataclass(slots=True)
class _DeviceAuthorization:
    device_code: str
    user_code: str
    verification_uri: str
    interval_seconds: float | None = None
    expires_in_seconds: float | None = None


def _error_detail(body: dict[str, Any] | None) -> str:
    for key in ("error_description", "detail", "message", "error"):
        value = body.get(key) if body else None
        if isinstance(value, str) and value.strip():
            return f": {value.strip()}"
    return ""


def _positive_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _stringify(value: Any) -> str:
    """`JSON.stringify`: no spaces between tokens."""
    return json_module.dumps(value, separators=(",", ":"))


def _post_form(url: str, fields: dict[str, str], cancel: CancelToken) -> Awaitable[oauth_http.OAuthHttpResponse]:
    return oauth_http.request(url, form=fields, headers=_FORM_HEADERS, timeout_ms=REQUEST_TIMEOUT_MS, cancel=cancel)


async def _start_device_authorization(cancel: CancelToken) -> _DeviceAuthorization:
    response = await _post_form(DEVICE_AUTHORIZATION_URL, {"client_id": CLIENT_ID}, cancel)
    body = response.json_object()
    if not response.ok:
        raise RuntimeError(f"Meta device authorization failed with status {response.status}{_error_detail(body)}")
    fields = body or {}
    device_code = fields.get("device_code")
    user_code = fields.get("user_code")
    # The verification URI is opened in the user's browser; only http(s) URLs are trusted.
    verification_uri = http_or_https_url(fields.get("verification_uri_complete")) or http_or_https_url(
        fields.get("verification_uri")
    )
    if (
        not isinstance(device_code, str)
        or not device_code
        or not isinstance(user_code, str)
        or not user_code
        or verification_uri is None
    ):
        raise RuntimeError(f"Invalid Meta device authorization response: {_stringify(body)}")
    return _DeviceAuthorization(
        device_code=device_code,
        user_code=user_code,
        verification_uri=verification_uri,
        interval_seconds=_positive_number(fields.get("interval")),
        expires_in_seconds=_positive_number(fields.get("expires_in")),
    )


def _poll_for_identity_token(device: _DeviceAuthorization, cancel: CancelToken) -> Awaitable[str]:
    async def poll() -> OAuthDeviceCodePollResult:
        response = await _post_form(
            DEVICE_TOKEN_URL,
            {
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "device_code": device.device_code,
                "client_id": CLIENT_ID,
            },
            cancel,
        )
        body = response.json_object()
        access_token = body.get("access_token") if body else None
        if response.ok and isinstance(access_token, str) and access_token:
            return OAuthDeviceCodePollResult(status="complete", value=access_token)
        match body.get("error") if body else None:
            case "authorization_pending":
                return OAuthDeviceCodePollResult(status="pending")
            case "slow_down":
                return OAuthDeviceCodePollResult(
                    status="slow_down", interval_seconds=_positive_number(body.get("interval") if body else None)
                )
            case "access_denied":
                return OAuthDeviceCodePollResult(status="failed", message="Meta login was denied.")
            case "expired_token":
                return OAuthDeviceCodePollResult(
                    status="failed", message="Meta device authorization expired. Please restart login."
                )
            case _:
                return OAuthDeviceCodePollResult(
                    status="failed",
                    message=f"Meta device token request failed with status {response.status}{_error_detail(body)}",
                )

    return poll_oauth_device_code_flow(
        poll=poll,
        interval_seconds=device.interval_seconds,
        expires_in_seconds=device.expires_in_seconds,
        wait_before_first_poll=True,
        cancel=cancel,
    )


async def _mint_api_key(identity_token: str, cancel: CancelToken) -> OAuthCredential:
    """Exchange an identity token for a Model API key. Keys are valid for about a day."""
    response = await oauth_http.request(
        API_KEY_MINT_URL,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {identity_token}",
            "Content-Type": "application/json",
            "x-api-version": "1.0.0",
        },
        json_body={},
        timeout_ms=REQUEST_TIMEOUT_MS,
        cancel=cancel,
    )
    body = response.json_object()
    if response.status in (401, 403):
        # Identity token is not renewable (see module docstring); only a fresh device flow helps.
        raise RuntimeError(
            f"Meta session expired (status {response.status}). Run `/login meta` to sign in again.{_error_detail(body)}"
        )
    if not response.ok:
        raise RuntimeError(f"Meta API key mint failed with status {response.status}{_error_detail(body)}")
    api_key = body.get("api_key") if body else None
    if not isinstance(api_key, str) or not api_key:
        action_url = http_or_https_url(body.get("action_url") if body else None)
        raise RuntimeError(f"Meta did not issue an API key.{f' Complete setup at {action_url}' if action_url else ''}")
    return OAuthCredential(refresh=identity_token, access=api_key, expires=clock.now_ms() + API_KEY_LIFETIME_MS)


async def _login_meta(interaction: ProviderAuthInteraction) -> OAuthCredential:
    cancel = interaction.cancel
    try:
        device = await _start_device_authorization(cancel)
        interaction.notify(
            AuthEvent(
                type="device_code",
                user_code=device.user_code,
                verification_uri=device.verification_uri,
                interval_seconds=device.interval_seconds,
                expires_in_seconds=device.expires_in_seconds,
            )
        )
        identity_token = await _poll_for_identity_token(device, cancel)
        interaction.notify(AuthEvent(type="progress", message="Enabling Meta Model API access..."))
        return await _mint_api_key(identity_token, cancel)
    except Exception:
        # An in-flight request raises AbortError on cancel; the login UI matches on this message.
        if cancel.cancelled:
            raise RuntimeError("Login cancelled") from None
        raise


def _refresh(credential: OAuthCredential, cancel: CancelToken) -> Awaitable[OAuthCredential]:
    return _mint_api_key(credential.refresh, cancel)


async def _to_auth(credential: OAuthCredential) -> ModelAuth:
    return ModelAuth(api_key=credential.access)


meta_oauth = OAuthAuth(
    name="Meta (Muse subscription)",
    is_subscription=True,
    login_label="Sign in with Meta",
    login=_login_meta,
    refresh=_refresh,
    to_auth=_to_auth,
)
