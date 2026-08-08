"""Port of pi's OpenAI Codex flow (packages/ai/src/auth/oauth/openai-codex.ts).

ChatGPT Plus/Pro login, offering browser and device-code methods. The credential
carries the `chatgpt_account_id` claim decoded out of the access token; it lands
in `OAuthCredential.extra` under pi's own key, which is what auth.json stores.

pi's deferred `node:crypto`/`node:http` imports and the "only available in
Node.js environments" guards they exist for have no counterpart here.
"""

import base64
import json
import math
import secrets
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

import tonio.colored as tonio

from pidrei_ai.auth.oauth import http as oauth_http
from pidrei_ai.auth.oauth.callback_server import (
    CallbackRequest,
    CallbackResponse,
    CallbackServer,
    OneShotValue,
    start_callback_server,
)
from pidrei_ai.auth.oauth.device_code import OAuthDeviceCodePollResult, poll_oauth_device_code_flow
from pidrei_ai.auth.oauth.oauth_page import oauth_error_html, oauth_success_html
from pidrei_ai.auth.oauth.pkce import generate_pkce
from pidrei_ai.auth.types import (
    AuthEvent,
    AuthPrompt,
    AuthPromptOption,
    ModelAuth,
    OAuthAuth,
    OAuthCredential,
    ProviderAuthInteraction,
)
from pidrei_ai.utils import clock
from pidrei_ai.utils.cancel import AbortError, CancelToken
from pidrei_ai.utils.provider_env import get_provider_env_value


CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTH_BASE_URL = "https://auth.openai.com"
AUTHORIZE_URL = f"{AUTH_BASE_URL}/oauth/authorize"
TOKEN_URL = f"{AUTH_BASE_URL}/oauth/token"
REDIRECT_URI = "http://localhost:1455/auth/callback"
DEVICE_USER_CODE_URL = f"{AUTH_BASE_URL}/api/accounts/deviceauth/usercode"
DEVICE_TOKEN_URL = f"{AUTH_BASE_URL}/api/accounts/deviceauth/token"
DEVICE_VERIFICATION_URI = f"{AUTH_BASE_URL}/codex/device"
DEVICE_REDIRECT_URI = f"{AUTH_BASE_URL}/deviceauth/callback"
DEVICE_CODE_TIMEOUT_SECONDS = 15 * 60
OPENAI_CODEX_BROWSER_LOGIN_METHOD = "browser"
OPENAI_CODEX_DEVICE_CODE_LOGIN_METHOD = "device_code"
SCOPE = "openid profile email offline_access"
JWT_CLAIM_PATH = "https://api.openai.com/auth"
CALLBACK_PORT = 1455
CALLBACK_PATH = "/auth/callback"


@dataclass(slots=True)
class _OAuthToken:
    access: str
    refresh: str
    expires: int


@dataclass(slots=True)
class _DeviceAuthInfo:
    device_auth_id: str
    user_code: str
    interval_seconds: float


@dataclass(slots=True)
class _DeviceTokenSuccess:
    authorization_code: str
    code_verifier: str


def _get_callback_host() -> str:
    return get_provider_env_value("PIDREI_OAUTH_CALLBACK_HOST") or "127.0.0.1"


def _create_state() -> str:
    return secrets.token_hex(16)


def _parse_authorization_input(value: str) -> tuple[str | None, str | None]:
    value = value.strip()
    if not value:
        return None, None

    split = urlsplit(value)
    if split.scheme and split.netloc:
        query = parse_qs(split.query)
        return (query.get("code") or [None])[0], (query.get("state") or [None])[0]

    if "#" in value:
        code, state = value.split("#")[:2]
        return code, state

    if "code=" in value:
        query = parse_qs(value)
        return (query.get("code") or [None])[0], (query.get("state") or [None])[0]

    return value, None


def _decode_jwt(token: str) -> dict[str, Any] | None:
    """pi's `decodeJwt`. `atob` only accepts standard base64; decoding base64url
    too costs nothing and is what a JWT is actually specified to use."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        payload = parts[1] or ""
        decoded = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
        parsed = json.loads(decoded)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


async def _request_with_login_cancellation(
    url: str,
    *,
    headers: dict[str, str],
    json_body: Any = None,
    form: dict[str, str] | None = None,
    cancel: CancelToken | None = None,
) -> oauth_http.OAuthHttpResponse:
    try:
        return await oauth_http.request(url, headers=headers, json_body=json_body, form=form, cancel=cancel)
    except AbortError:
        raise RuntimeError("Login cancelled") from None


def _read_token_response(response: oauth_http.OAuthHttpResponse, operation: str) -> _OAuthToken:
    if not response.ok:
        text = response.text
        try:
            status_text = HTTPStatus(response.status).phrase
        except ValueError:
            status_text = ""
        raise RuntimeError(f"OpenAI Codex token {operation} failed ({response.status}): {text or status_text}")

    body = response.json_object()
    access_token = body.get("access_token") if body else None
    refresh_token = body.get("refresh_token") if body else None
    expires_in = body.get("expires_in") if body else None
    if not access_token or not refresh_token or isinstance(expires_in, bool) or not isinstance(expires_in, int | float):
        raise RuntimeError(
            f"OpenAI Codex token {operation} response missing fields: {json.dumps(body, separators=(',', ':'))}"
        )

    return _OAuthToken(
        access=access_token,
        refresh=refresh_token,
        expires=int(clock.now_ms() + expires_in * 1000),
    )


async def _exchange_authorization_code(
    code: str,
    verifier: str,
    redirect_uri: str,
    cancel: CancelToken,
) -> _OAuthToken:
    response = await _request_with_login_cancellation(
        TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        form={
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": redirect_uri,
        },
        cancel=cancel,
    )
    return _read_token_response(response, "exchange")


async def _refresh_access_token(refresh_token: str, cancel: CancelToken) -> _OAuthToken:
    try:
        response = await oauth_http.request(
            TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            form={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CLIENT_ID,
            },
            cancel=cancel,
        )
    except Exception as error:
        raise RuntimeError(f"OpenAI Codex token refresh error: {error}") from error

    return _read_token_response(response, "refresh")


async def _start_device_auth(cancel: CancelToken) -> _DeviceAuthInfo:
    response = await _request_with_login_cancellation(
        DEVICE_USER_CODE_URL,
        headers={"Content-Type": "application/json"},
        json_body={"client_id": CLIENT_ID},
        cancel=cancel,
    )

    if not response.ok:
        if response.status == 404:
            raise RuntimeError(
                "OpenAI Codex device code login is not enabled for this server. "
                "Use browser login or verify the server URL."
            )
        response_body = response.text
        raise RuntimeError(
            f"OpenAI Codex device code request failed with status {response.status}"
            f"{f': {response_body}' if response_body else ''}"
        )

    body = response.json_object()
    raw_interval = body.get("interval") if body else None
    interval_seconds = _number(raw_interval.strip()) if isinstance(raw_interval, str) else raw_interval
    device_auth_id = body.get("device_auth_id") if body else None
    user_code = body.get("user_code") if body else None
    if (
        not device_auth_id
        or not user_code
        or isinstance(interval_seconds, bool)
        or not isinstance(interval_seconds, int | float)
        or not math.isfinite(interval_seconds)
        or interval_seconds < 0
    ):
        raise RuntimeError(f"Invalid OpenAI Codex device code response: {json.dumps(body, separators=(',', ':'))}")

    return _DeviceAuthInfo(device_auth_id=device_auth_id, user_code=user_code, interval_seconds=interval_seconds)


def _number(value: str) -> float:
    """`Number(value)`: NaN for anything unparseable, which the caller rejects."""
    try:
        return float(value)
    except ValueError:
        return math.nan


async def _poll_device_auth(device: _DeviceAuthInfo, cancel: CancelToken) -> _DeviceTokenSuccess:
    async def poll() -> OAuthDeviceCodePollResult:
        response = await _request_with_login_cancellation(
            DEVICE_TOKEN_URL,
            headers={"Content-Type": "application/json"},
            json_body={"device_auth_id": device.device_auth_id, "user_code": device.user_code},
            cancel=cancel,
        )

        if response.ok:
            body = response.json_object()
            authorization_code = body.get("authorization_code") if body else None
            code_verifier = body.get("code_verifier") if body else None
            if not authorization_code or not code_verifier:
                return OAuthDeviceCodePollResult(
                    status="failed",
                    message=(
                        f"Invalid OpenAI Codex device auth token response: {json.dumps(body, separators=(',', ':'))}"
                    ),
                )
            return OAuthDeviceCodePollResult(
                status="complete",
                value=_DeviceTokenSuccess(authorization_code=authorization_code, code_verifier=code_verifier),
            )

        if response.status in (403, 404):
            return OAuthDeviceCodePollResult(status="pending")

        response_body = response.text
        error_code: Any = None
        try:
            parsed = json.loads(response_body)
            error = parsed.get("error") if isinstance(parsed, dict) else None
            error_code = error.get("code") if isinstance(error, dict) else error
        except ValueError:
            pass

        if error_code == "deviceauth_authorization_pending":
            return OAuthDeviceCodePollResult(status="pending")
        if error_code == "slow_down":
            return OAuthDeviceCodePollResult(status="slow_down")

        return OAuthDeviceCodePollResult(
            status="failed",
            message=(
                f"OpenAI Codex device auth failed with status {response.status}"
                f"{f': {response_body}' if response_body else ''}"
            ),
        )

    return await poll_oauth_device_code_flow(
        poll=poll,
        interval_seconds=device.interval_seconds,
        expires_in_seconds=DEVICE_CODE_TIMEOUT_SECONDS,
        cancel=cancel,
    )


@dataclass(slots=True)
class _AuthorizationFlow:
    verifier: str
    state: str
    url: str


def _create_authorization_flow(originator: str = "pi") -> _AuthorizationFlow:
    pkce = generate_pkce()
    state = _create_state()
    url = f"{AUTHORIZE_URL}?" + urlencode(
        {
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": REDIRECT_URI,
            "scope": SCOPE,
            "code_challenge": pkce.challenge,
            "code_challenge_method": "S256",
            "state": state,
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "originator": originator,
        }
    )
    return _AuthorizationFlow(verifier=pkce.verifier, state=state, url=url)


class _LocalOAuthServer:
    """pi's `OAuthServerInfo`. When the port is taken pi resolves a stub whose
    `waitForCode` yields null immediately, so login falls through to the manual
    paste prompt instead of failing."""

    __slots__ = ("_code", "_server")

    def __init__(self, server: CallbackServer | None, code: OneShotValue):
        self._server = server
        self._code = code

    def close(self) -> None:
        if self._server is not None:
            self._server.close()

    def cancel_wait(self) -> None:
        self._code.settle(None)

    async def wait_for_code(self) -> dict[str, str] | None:
        return await self._code.wait()


async def _start_local_oauth_server(state: str) -> _LocalOAuthServer:
    code = OneShotValue()

    async def handle(request: CallbackRequest) -> CallbackResponse:
        if request.path != CALLBACK_PATH:
            return CallbackResponse(404, oauth_error_html("Callback route not found."))
        if request.get("state") != state:
            return CallbackResponse(400, oauth_error_html("State mismatch."))
        callback_code = request.get("code")
        if not callback_code:
            return CallbackResponse(400, oauth_error_html("Missing authorization code."))
        code.settle({"code": callback_code})
        return CallbackResponse(200, oauth_success_html("OpenAI authentication completed. You can close this window."))

    try:
        server = await start_callback_server(host=_get_callback_host(), port=CALLBACK_PORT, handle=handle)
    except OSError:
        code.settle(None)
        return _LocalOAuthServer(None, code)
    return _LocalOAuthServer(server, code)


def _get_account_id(access_token: str) -> str | None:
    payload = _decode_jwt(access_token)
    auth = payload.get(JWT_CLAIM_PATH) if payload else None
    account_id = auth.get("chatgpt_account_id") if isinstance(auth, dict) else None
    return account_id if isinstance(account_id, str) and account_id else None


def _credentials_from_token(token: _OAuthToken) -> OAuthCredential:
    account_id = _get_account_id(token.access)
    if not account_id:
        raise RuntimeError("Failed to extract accountId from token")

    return OAuthCredential(
        access=token.access,
        refresh=token.refresh,
        expires=token.expires,
        extra={"accountId": account_id},
    )


async def _exchange_authorization_code_for_credentials(
    code: str, verifier: str, redirect_uri: str, cancel: CancelToken
) -> OAuthCredential:
    return _credentials_from_token(await _exchange_authorization_code(code, verifier, redirect_uri, cancel))


async def _login_device_code(interaction: ProviderAuthInteraction) -> OAuthCredential:
    device = await _start_device_auth(interaction.cancel)
    interaction.notify(
        AuthEvent(
            type="device_code",
            user_code=device.user_code,
            verification_uri=DEVICE_VERIFICATION_URI,
            interval_seconds=device.interval_seconds,
            expires_in_seconds=DEVICE_CODE_TIMEOUT_SECONDS,
        )
    )
    code = await _poll_device_auth(device, interaction.cancel)
    return await _exchange_authorization_code_for_credentials(
        code.authorization_code, code.code_verifier, DEVICE_REDIRECT_URI, interaction.cancel
    )


async def _login_browser(interaction: ProviderAuthInteraction) -> OAuthCredential:
    flow = _create_authorization_flow()
    server = await _start_local_oauth_server(flow.state)
    manual_abort = CancelToken()
    unsubscribe_abort = interaction.cancel.on_cancel(lambda _reason: server.cancel_wait())
    manual: dict[str, Any] = {}
    prompt_done = tonio.Event()

    interaction.notify(
        AuthEvent(
            type="auth_url",
            url=flow.url,
            instructions="A browser window should open. Complete login to finish.",
        )
    )

    async def run_manual_prompt() -> None:
        try:
            manual["input"] = await interaction.prompt(
                AuthPrompt(
                    type="manual_code",
                    message="Complete login in your browser, or paste the authorization code / redirect URL here:",
                    placeholder=REDIRECT_URI,
                    cancel=manual_abort,
                )
            )
        except Exception as error:
            manual["error"] = error
        prompt_done.set()
        server.cancel_wait()

    def read_manual_input() -> str | None:
        parsed_code, parsed_state = _parse_authorization_input(manual["input"])
        if parsed_state and parsed_state != flow.state:
            raise RuntimeError("State mismatch")
        return parsed_code

    code: str | None = None
    try:
        tonio.spawn.without_tracking(run_manual_prompt())

        result = await server.wait_for_code()
        if manual.get("error") is not None:
            raise manual["error"]
        if result is not None and result.get("code"):
            code = result["code"]
        elif manual.get("input"):
            code = read_manual_input()

        if not code:
            await prompt_done.wait()
            if manual.get("error") is not None:
                raise manual["error"]
            if manual.get("input"):
                code = read_manual_input()

        if not code:
            raise RuntimeError("Missing authorization code")
        return await _exchange_authorization_code_for_credentials(code, flow.verifier, REDIRECT_URI, interaction.cancel)
    finally:
        unsubscribe_abort()
        manual_abort.cancel()
        server.close()


async def _login(interaction: ProviderAuthInteraction) -> OAuthCredential:
    method = await interaction.prompt(
        AuthPrompt(
            type="select",
            message="Select OpenAI Codex login method:",
            options=[
                AuthPromptOption(id=OPENAI_CODEX_BROWSER_LOGIN_METHOD, label="Browser login (default)"),
                AuthPromptOption(id=OPENAI_CODEX_DEVICE_CODE_LOGIN_METHOD, label="Device code login (headless)"),
            ],
        )
    )

    if method == OPENAI_CODEX_DEVICE_CODE_LOGIN_METHOD:
        return await _login_device_code(interaction)
    if method != OPENAI_CODEX_BROWSER_LOGIN_METHOD:
        raise RuntimeError(f"Unknown OpenAI Codex login method: {method}")

    return await _login_browser(interaction)


async def _refresh(credential: OAuthCredential, cancel: CancelToken) -> OAuthCredential:
    return _credentials_from_token(await _refresh_access_token(credential.refresh, cancel))


async def _to_auth(credential: OAuthCredential) -> ModelAuth:
    return ModelAuth(api_key=credential.access)


openai_codex_oauth = OAuthAuth(
    name="OpenAI (ChatGPT Plus/Pro)",
    is_subscription=True,
    login=_login,
    refresh=_refresh,
    to_auth=_to_auth,
)
