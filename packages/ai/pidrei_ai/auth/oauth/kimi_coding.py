"""Port of pi's Kimi Code flow (packages/ai/src/auth/oauth/kimi-coding.ts).

RFC 8628 device authorization grant against https://auth.kimi.com with JSON
responses. The access token authenticates requests to
https://api.kimi.com/coding as an `Authorization: Bearer` header.
"""

import json as json_module
import math
import re
from dataclasses import dataclass
from typing import Any

from pidrei_ai.auth.oauth import http as oauth_http
from pidrei_ai.auth.oauth.device_code import OAuthDeviceCodePollResult, poll_oauth_device_code_flow
from pidrei_ai.auth.oauth.urls import http_or_https_url
from pidrei_ai.auth.types import AuthEvent, AuthInteraction, ModelAuth, OAuthAuth, OAuthCredential
from pidrei_ai.utils import clock
from pidrei_ai.utils.cancel import CancelToken
from pidrei_ai.utils.provider_env import get_provider_env_value


CLIENT_ID = "17e5f671-d194-4dfb-9706-5516cb48c098"
DEFAULT_OAUTH_HOST = "https://auth.kimi.com"
DEVICE_CODE_TIMEOUT_SECONDS = 15 * 60
DEFAULT_POLL_INTERVAL_SECONDS = 5
REQUEST_TIMEOUT_MS = 30 * 1000
REFRESH_MAX_RETRIES = 3

_FORM_HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded",
    "Accept": "application/json",
}


@dataclass(slots=True)
class _DeviceAuthorization:
    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str
    interval_seconds: float
    expires_in_seconds: float


@dataclass(slots=True)
class _TokenResponse:
    access: str
    refresh: str
    expires: int


def _get_oauth_host() -> str:
    override = get_provider_env_value("KIMI_CODE_OAUTH_HOST") or get_provider_env_value("KIMI_OAUTH_HOST")
    return re.sub(r"/+$", "", override or DEFAULT_OAUTH_HOST)


def _positive_seconds(value: Any, fallback: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return fallback
    return value if math.isfinite(value) and value > 0 else fallback


def _stringify(value: Any) -> str:
    """`JSON.stringify`: no spaces between tokens."""
    return json_module.dumps(value, separators=(",", ":"))


async def _post_form(
    url: str, fields: dict[str, str], cancel: CancelToken | None = None
) -> oauth_http.OAuthHttpResponse:
    return await oauth_http.request(
        url, form=fields, headers=_FORM_HEADERS, timeout_ms=REQUEST_TIMEOUT_MS, cancel=cancel
    )


async def _start_device_authorization(oauth_host: str, cancel: CancelToken | None = None) -> _DeviceAuthorization:
    response = await _post_form(f"{oauth_host}/api/oauth/device_authorization", {"client_id": CLIENT_ID}, cancel)

    if not response.ok:
        text = response.text
        raise RuntimeError(
            f"Kimi Code device authorization failed with status {response.status}{f': {text}' if text else ''}"
        )

    body = response.json_object()
    device_code = body.get("device_code") if body else None
    user_code = body.get("user_code") if body else None
    verification_uri = body.get("verification_uri") if body else None
    verification_uri_complete = body.get("verification_uri_complete") if body else None
    if (
        not isinstance(device_code, str)
        or not isinstance(user_code, str)
        or not isinstance(verification_uri, str)
        or not isinstance(verification_uri_complete, str)
        # The verification URI is opened in the user's browser; only http(s) URLs are trusted.
        or http_or_https_url(verification_uri_complete) is None
        or http_or_https_url(verification_uri) is None
    ):
        raise RuntimeError(f"Invalid Kimi Code device authorization response: {_stringify(body)}")

    interval = body.get("interval") if body else None
    expires_in = body.get("expires_in") if body else None
    return _DeviceAuthorization(
        device_code=device_code,
        user_code=user_code,
        verification_uri=verification_uri,
        verification_uri_complete=verification_uri_complete,
        interval_seconds=_positive_seconds(interval, DEFAULT_POLL_INTERVAL_SECONDS),
        expires_in_seconds=_positive_seconds(expires_in, DEVICE_CODE_TIMEOUT_SECONDS),
    )


def _parse_token_response(body: dict[str, Any] | None, operation: str) -> _TokenResponse:
    access_token = body.get("access_token") if body else None
    refresh_token = body.get("refresh_token") if body else None
    expires_in = body.get("expires_in") if body else None
    if (
        not isinstance(access_token, str)
        or not access_token
        or not isinstance(refresh_token, str)
        or not refresh_token
        or isinstance(expires_in, bool)
        or not isinstance(expires_in, int | float)
        or not math.isfinite(expires_in)
        or expires_in <= 0
    ):
        raise RuntimeError(f"Kimi Code token {operation} response missing fields: {_stringify(body)}")
    return _TokenResponse(
        access=access_token,
        refresh=refresh_token,
        expires=int(clock.now_ms() + expires_in * 1000),
    )


async def _poll_for_token(
    oauth_host: str, device: _DeviceAuthorization, cancel: CancelToken | None = None
) -> _TokenResponse:
    async def poll() -> OAuthDeviceCodePollResult:
        response = await _post_form(
            f"{oauth_host}/api/oauth/token",
            {
                "client_id": CLIENT_ID,
                "device_code": device.device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
            cancel,
        )

        if response.status >= 500:
            text = response.text
            return OAuthDeviceCodePollResult(
                status="failed",
                message=(
                    f"Kimi Code device token request failed with status {response.status}{f': {text}' if text else ''}"
                ),
            )

        body = response.json_object()
        if response.ok and isinstance(body.get("access_token") if body else None, str):
            try:
                return OAuthDeviceCodePollResult(status="complete", value=_parse_token_response(body, "poll"))
            except Exception as error:
                return OAuthDeviceCodePollResult(status="failed", message=str(error))

        error_code = body.get("error") if body else None
        error_description = body.get("error_description") if body else None
        description = f": {error_description}" if isinstance(error_description, str) else ""
        if error_code == "authorization_pending":
            return OAuthDeviceCodePollResult(status="pending")
        if error_code == "slow_down":
            interval = body.get("interval") if body else None
            return OAuthDeviceCodePollResult(
                status="slow_down",
                interval_seconds=(
                    interval
                    if isinstance(interval, int | float) and not isinstance(interval, bool) and interval > 0
                    else None
                ),
            )
        if error_code == "expired_token":
            return OAuthDeviceCodePollResult(
                status="failed", message="Kimi Code device authorization expired. Please restart login."
            )
        if error_code == "access_denied":
            return OAuthDeviceCodePollResult(status="failed", message="Kimi Code login was denied.")
        return OAuthDeviceCodePollResult(
            status="failed",
            message=(
                f"Kimi Code device token request failed (status {response.status})"
                f"{f': {error_code}{description}' if isinstance(error_code, str) else ''}"
            ),
        )

    return await poll_oauth_device_code_flow(
        poll=poll,
        interval_seconds=device.interval_seconds,
        expires_in_seconds=device.expires_in_seconds,
        wait_before_first_poll=True,
        cancel=cancel,
    )


def _is_retryable_refresh_failure(status: int) -> bool:
    return status == 429 or status >= 500


async def _refresh_token(
    oauth_host: str, refresh_token_value: str, cancel: CancelToken | None = None
) -> _TokenResponse:
    last_error: Exception | None = None
    for attempt in range(REFRESH_MAX_RETRIES + 1):
        if attempt > 0:
            await clock.sleep_ms(1000 * 2 ** (attempt - 1))
        if cancel is not None and cancel.cancelled:
            raise RuntimeError("Kimi Code token refresh aborted")

        try:
            response = await _post_form(
                f"{oauth_host}/api/oauth/token",
                {
                    "client_id": CLIENT_ID,
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token_value,
                },
                cancel,
            )
        except Exception as error:
            last_error = error
            continue

        body = response.json_object()
        if response.ok:
            return _parse_token_response(body, "refresh")

        # Unauthorized: the stored credential is dead; Models clears it and prompts re-login.
        if response.status in (401, 403) or (body or {}).get("error") == "invalid_grant":
            error_description = (body or {}).get("error_description")
            description = f": {error_description}" if isinstance(error_description, str) else ""
            raise RuntimeError(f"Kimi Code token refresh unauthorized (status {response.status}){description}")

        if _is_retryable_refresh_failure(response.status) and attempt < REFRESH_MAX_RETRIES:
            last_error = RuntimeError(f"Kimi Code token refresh failed with status {response.status}")
            continue

        text = _stringify(body)
        raise RuntimeError(f"Kimi Code token refresh failed with status {response.status}{f': {text}' if text else ''}")

    raise last_error if last_error is not None else RuntimeError("Kimi Code token refresh failed")


async def _login_kimi_coding(interaction: AuthInteraction) -> OAuthCredential:
    oauth_host = _get_oauth_host()
    device = await _start_device_authorization(oauth_host, interaction.cancel)
    interaction.notify(
        AuthEvent(
            type="device_code",
            user_code=device.user_code,
            verification_uri=device.verification_uri_complete,
            interval_seconds=device.interval_seconds,
            expires_in_seconds=device.expires_in_seconds,
        )
    )
    token = await _poll_for_token(oauth_host, device, interaction.cancel)
    return OAuthCredential(access=token.access, refresh=token.refresh, expires=token.expires)


async def _refresh(credential: OAuthCredential, cancel: CancelToken | None) -> OAuthCredential:
    token = await _refresh_token(_get_oauth_host(), credential.refresh, cancel)
    return OAuthCredential(access=token.access, refresh=token.refresh, expires=token.expires)


async def _to_auth(credential: OAuthCredential) -> ModelAuth:
    return ModelAuth(headers={"Authorization": f"Bearer {credential.access}"})


kimi_coding_oauth = OAuthAuth(
    name="Kimi Code (subscription)",
    login_label="Sign in with Kimi Code",
    login=_login_kimi_coding,
    refresh=_refresh,
    to_auth=_to_auth,
)
