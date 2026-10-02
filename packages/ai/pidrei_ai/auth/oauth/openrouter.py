"""Port of pi's OpenRouter PKCE flow (packages/ai/src/auth/oauth/openrouter.ts).

OpenRouter exchanges an authorization code for a permanent, user-controlled API
key rather than an expiring access/refresh token pair. The callback is handled by
a one-shot loopback server on an ephemeral port, raced against a manual prompt so
remote/headless sessions can paste the redirect URL when the browser cannot reach
the loopback server.
"""

import uuid
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

from pidrei_ai.auth.oauth import http as oauth_http
from pidrei_ai.auth.oauth.callback_server import start_oauth_callback_server, wait_for_callback_or_manual_input
from pidrei_ai.auth.types import (
    AuthEvent,
    LoginOptions,
    ModelAuth,
    OAuthAuth,
    OAuthCredential,
    ProviderAuthInteraction,
)
from pidrei_ai.utils.provider_env import get_provider_env_value
from pidrei_http import http
from pidrei_http.pkce import generate_pkce
from pidrei_utils.cancel import AbortError, CancelToken


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


async def _exchange_authorization_code(code: str, verifier: str, cancel: CancelToken) -> OAuthCredential:
    if cancel.cancelled:
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


async def _login_openrouter(
    interaction: ProviderAuthInteraction, options: LoginOptions | None = None
) -> OAuthCredential:
    pkce = generate_pkce()
    # OpenRouter sends no `state`; the random path keeps stray requests from completing the sign-in.
    callback = await start_oauth_callback_server(
        provider_name="OpenRouter",
        host=_get_callback_host(),
        port=0,
        path=f"/oauth/callback/{uuid.uuid4()}",
        complete=lambda code: _exchange_authorization_code(code, pkce.verifier, interaction.cancel),
        cancel=interaction.cancel,
        timeout_ms=LOGIN_TIMEOUT_MS,
    )

    try:
        authorize_url = f"{AUTHORIZE_URL}?" + urlencode(
            {
                "callback_url": callback.redirect_uri,
                "code_challenge": pkce.challenge,
                "code_challenge_method": "S256",
            }
        )

        interaction.notify(
            AuthEvent(
                type="progress",
                message=f"Listening for OpenRouter OAuth callback on {callback.redirect_uri}",
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

        result = await wait_for_callback_or_manual_input(
            interaction,
            callback,
            message="Complete sign-in in your browser, or paste the authorization code / redirect URL here:",
            placeholder=callback.redirect_uri,
        )
        if result.type == "callback":
            return result.value
        code = _parse_authorization_input(result.input)
        if not code:
            raise RuntimeError("Missing authorization code")
        interaction.notify(AuthEvent(type="progress", message="Exchanging authorization code for an API key..."))
        return await _exchange_authorization_code(code, pkce.verifier, interaction.cancel)
    finally:
        callback.close()


async def _refresh(credential: OAuthCredential, _cancel: CancelToken) -> OAuthCredential:
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
