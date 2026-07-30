"""Port of pi's OpenRouter PKCE flow (packages/ai/src/auth/oauth/openrouter.ts).

OpenRouter exchanges an authorization code for a permanent, user-controlled API
key rather than an expiring access/refresh token pair. The callback is handled by
a one-shot loopback server on an ephemeral port, raced against a manual prompt so
remote/headless sessions can paste the redirect URL when the browser cannot reach
the loopback server.
"""

import uuid
from collections.abc import Callable
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import tonio.colored as tonio

from pidrei_ai.auth.oauth import http as oauth_http
from pidrei_ai.auth.oauth.callback_server import (
    CallbackRequest,
    CallbackResponse,
    OneShotValue,
    start_callback_server,
)
from pidrei_ai.auth.oauth.oauth_page import oauth_error_html, oauth_success_html
from pidrei_ai.auth.oauth.pkce import generate_pkce
from pidrei_ai.auth.types import AuthEvent, AuthInteraction, AuthPrompt, ModelAuth, OAuthAuth, OAuthCredential
from pidrei_ai.utils import http
from pidrei_ai.utils.cancel import AbortError, CancelToken
from pidrei_ai.utils.provider_env import get_provider_env_value


AUTHORIZE_URL = "https://openrouter.ai/auth"
TOKEN_URL = "https://openrouter.ai/api/v1/auth/keys"  # noqa: S105 - an endpoint, not a secret
LOGIN_TIMEOUT_MS = 5 * 60 * 1000
TOKEN_EXCHANGE_TIMEOUT_MS = 30_000
# `Number.MAX_SAFE_INTEGER`: the credential never expires.
MAX_SAFE_INTEGER = 9007199254740991


def _get_callback_host() -> str:
    return get_provider_env_value("PIDREI_OAUTH_CALLBACK_HOST") or "127.0.0.1"


def _parse_authorization_input(input_value: str) -> str | None:
    """pi's `parseAuthorizationInput`: a redirect URL, a query string carrying
    `code=`, or the bare authorization code."""
    value = input_value.strip()
    if not value:
        return None

    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc:
        codes = parse_qs(parsed.query).get("code")
        return codes[0] if codes else None

    if "code=" in value:
        # URLSearchParams tolerates one leading "?".
        codes = parse_qs(value.removeprefix("?")).get("code")
        return codes[0] if codes else None

    return value


def _error_detail(body: dict[str, Any]) -> str | None:
    if isinstance(body.get("error_description"), str):
        return body["error_description"]
    if isinstance(body.get("message"), str):
        return body["message"]
    if isinstance(body.get("error"), str):
        return body["error"]
    error = body.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    return None


async def _exchange_authorization_code(code: str, verifier: str, cancel: CancelToken | None = None) -> OAuthCredential:
    if cancel is not None and cancel.cancelled:
        raise RuntimeError("Login cancelled")

    body: dict[str, Any] = {}
    try:
        response = await oauth_http.request(
            TOKEN_URL,
            headers={"accept": "application/json", "content-type": "application/json"},
            json_body={"code": code, "code_verifier": verifier, "code_challenge_method": "S256"},
            timeout_ms=TOKEN_EXCHANGE_TIMEOUT_MS,
            cancel=cancel,
        )
    except AbortError:
        raise RuntimeError("Login cancelled") from None
    except http.RequestTimeout:
        raise RuntimeError("OpenRouter OAuth token exchange timed out") from None

    parsed = response.json_object()
    if parsed is not None:
        body = parsed
    elif response.ok:
        raise RuntimeError("OpenRouter OAuth returned invalid JSON")

    if not response.ok:
        detail = _error_detail(body)
        raise RuntimeError(
            f"OpenRouter OAuth key exchange failed (HTTP {response.status}){f': {detail}' if detail else ''}"
        )

    if not isinstance(body.get("key"), str) or not body["key"]:
        raise RuntimeError('OpenRouter OAuth response carries no "key"')

    return OAuthCredential(access=body["key"], refresh="", expires=MAX_SAFE_INTEGER)


class _OpenRouterCallbackServer:
    """pi's `OpenRouterCallbackServer`, following the codex `OAuthServerInfo`
    shape: `wait_for_credential` resolves None once `cancel_wait` hands the
    login over to manual entry, and `close` is pure cleanup that never settles
    the wait."""

    __slots__ = ("_cancel_wait", "_close", "_result", "callback_url")

    def __init__(
        self, callback_url: str, result: OneShotValue, close: Callable[[], None], cancel_wait: Callable[[], None]
    ):
        self.callback_url = callback_url
        self._result = result
        self._close = close
        self._cancel_wait = cancel_wait

    async def wait_for_credential(self) -> OAuthCredential | None:
        outcome, value = await self._result.wait()
        if outcome == "error":
            raise value
        return value

    def cancel_wait(self) -> None:
        self._cancel_wait()

    def close(self) -> None:
        self._close()


async def _start_callback_server(
    callback_path: str, verifier: str, cancel: CancelToken | None = None
) -> _OpenRouterCallbackServer:
    if cancel is not None and cancel.cancelled:
        raise RuntimeError("Login cancelled")
    callback_host = _get_callback_host()
    result = OneShotValue()
    claimed = False
    closed = False
    unsubscribe: Any = None

    def close() -> None:
        nonlocal closed
        if closed:
            return
        closed = True
        if unsubscribe is not None:
            unsubscribe()
        server.close()

    def finish(outcome: str, value: Any) -> None:
        if result.settled:
            return
        close()
        result.settle((outcome, value))

    def cancel_wait() -> None:
        # A claimed callback is already exchanging its code; let that exchange settle the login.
        if not claimed:
            finish("ok", None)

    async def handle(request: CallbackRequest) -> CallbackResponse:
        nonlocal claimed
        if request.method != "GET" or request.path != callback_path:
            return CallbackResponse(404, oauth_error_html("OAuth callback route not found."))
        if claimed or result.settled:
            return CallbackResponse(409, oauth_error_html("This OAuth callback has already been used."))

        oauth_error = request.get("error")
        if oauth_error:
            description = request.get("error_description") or oauth_error
            finish("error", RuntimeError(f"OpenRouter authorization failed: {description}"))
            return CallbackResponse(400, oauth_error_html("OpenRouter authorization was denied.", description))

        code = request.get("code")
        if not code:
            return CallbackResponse(400, oauth_error_html("OpenRouter returned no authorization code."))
        claimed = True

        try:
            credential = await _exchange_authorization_code(code, verifier, cancel)
        except Exception as error:
            message = str(error) or "Unknown token exchange error"
            finish("error", error)
            return CallbackResponse(502, oauth_error_html("OpenRouter key exchange failed.", message))
        finish("ok", credential)
        return CallbackResponse(200, oauth_success_html("Signed in to OpenRouter. You may now close this page."))

    server = await start_callback_server(host=callback_host, port=0, handle=handle)

    if cancel is not None:
        unsubscribe = cancel.on_cancel(lambda _reason: finish("error", RuntimeError("Login cancelled")))
        if result.settled:
            raise RuntimeError("Login cancelled")

    async def _login_deadline() -> None:
        if not await result.wait_for(LOGIN_TIMEOUT_MS / 1000):
            finish("error", RuntimeError("OpenRouter OAuth login timed out"))

    tonio.spawn.without_tracking(_login_deadline())

    return _OpenRouterCallbackServer(
        callback_url=f"http://{callback_host}:{server.port}{callback_path}",
        result=result,
        close=close,
        cancel_wait=cancel_wait,
    )


async def _login_openrouter(interaction: AuthInteraction) -> OAuthCredential:
    pkce = generate_pkce()
    callback_path = f"/oauth/callback/{uuid.uuid4()}"
    callback = await _start_callback_server(callback_path, pkce.verifier, interaction.cancel)
    manual_abort = CancelToken()
    manual: dict[str, Any] = {}
    prompt_done = tonio.Event()

    async def run_manual_prompt() -> None:
        try:
            manual["input"] = await interaction.prompt(
                AuthPrompt(
                    type="manual_code",
                    message="Complete sign-in in your browser, or paste the authorization code / redirect URL here:",
                    placeholder=callback.callback_url,
                    cancel=manual_abort,
                )
            )
        except Exception as error:
            manual["error"] = error
        prompt_done.set()
        callback.cancel_wait()

    try:
        authorize_url = f"{AUTHORIZE_URL}?" + urlencode(
            {
                "callback_url": callback.callback_url,
                "code_challenge": pkce.challenge,
                "code_challenge_method": "S256",
            }
        )

        interaction.notify(
            AuthEvent(
                type="progress",
                message=f"Listening for OpenRouter OAuth callback on {callback.callback_url}",
            )
        )
        interaction.notify(
            AuthEvent(
                type="auth_url",
                url=authorize_url,
                instructions=(
                    "Complete sign-in in your browser. "
                    "If the browser is on another machine, paste the final redirect URL here."
                ),
            )
        )

        tonio.spawn.without_tracking(run_manual_prompt())

        credential = await callback.wait_for_credential()
        if manual.get("error") is not None:
            raise manual["error"]
        if credential is not None:
            return credential

        await prompt_done.wait()
        if manual.get("error") is not None:
            raise manual["error"]
        code = _parse_authorization_input(manual["input"]) if manual.get("input") else None
        if not code:
            raise RuntimeError("Missing authorization code")
        interaction.notify(AuthEvent(type="progress", message="Exchanging authorization code for an API key..."))
        return await _exchange_authorization_code(code, pkce.verifier, interaction.cancel)
    finally:
        manual_abort.cancel()
        callback.close()


async def _refresh(credential: OAuthCredential, _cancel: CancelToken | None) -> OAuthCredential:
    return credential


async def _to_auth(credential: OAuthCredential) -> ModelAuth:
    return ModelAuth(api_key=credential.access)


openrouter_oauth = OAuthAuth(
    name="OpenRouter OAuth",
    login_label="Sign in with OpenRouter",
    login=_login_openrouter,
    refresh=_refresh,
    to_auth=_to_auth,
)
