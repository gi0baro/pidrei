"""Port of pi's Anthropic OAuth flow (packages/ai/src/auth/oauth/anthropic.ts).

Claude Pro/Max login: PKCE against claude.ai, with a choice of two methods.
Browser login runs a loopback callback server on a fixed port racing a manual
paste prompt, so a browser on another machine still works; when the port is
taken, only the paste prompt is used. Copy code login (headless) redirects to
Anthropic's code page and asks for the code it shows.

pi's Node-only guard (`getNodeApis` refusing to run outside Node/Bun) has no
counterpart: there is no browser build to protect here.
"""

import base64
import json
import traceback
from collections.abc import Awaitable
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

from pidrei_ai.auth.oauth import http as oauth_http
from pidrei_ai.auth.oauth.callback_server import (
    OAuthCallbackServer,
    keep_code,
    start_oauth_callback_server,
    wait_for_callback_or_manual_input,
)
from pidrei_ai.auth.oauth.pkce import generate_pkce
from pidrei_ai.auth.types import (
    AuthEvent,
    AuthPrompt,
    AuthPromptOption,
    LoginOptions,
    ModelAuth,
    OAuthAuth,
    OAuthCredential,
    ProviderAuthInteraction,
)
from pidrei_ai.utils import clock
from pidrei_ai.utils.cancel import CancelToken
from pidrei_ai.utils.provider_env import get_provider_env_value


def _decode(value: str) -> str:
    return base64.b64decode(value).decode("utf-8")


CLIENT_ID = _decode("OWQxYzI1MGEtZTYxYi00NGQ5LTg4ZWQtNTk0NGQxOTYyZjVl")
AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
TOKEN_URL = "https://platform.claude.com/v1/oauth/token"  # noqa: S105 - an endpoint, not a secret
CALLBACK_HOST = get_provider_env_value("PIDREI_OAUTH_CALLBACK_HOST") or "127.0.0.1"
CALLBACK_PORT = 53692
CALLBACK_PATH = "/callback"
REDIRECT_URI = f"http://localhost:{CALLBACK_PORT}{CALLBACK_PATH}"
COPY_CODE_REDIRECT_URI = "https://platform.claude.com/oauth/code/callback"
ANTHROPIC_BROWSER_LOGIN_METHOD = "browser"
ANTHROPIC_COPY_CODE_LOGIN_METHOD = "copy_code"
SCOPES = "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload"
TOKEN_TIMEOUT_MS = 30_000


def _parse_authorization_input(value: str) -> tuple[str | None, str | None]:
    """pi's `parseAuthorizationInput`, as `(code, state)`."""
    value = value.strip()
    if not value:
        return None, None

    split = urlsplit(value)
    if split.scheme and split.netloc:
        query = parse_qs(split.query)
        return (query.get("code") or [None])[0], (query.get("state") or [None])[0]

    if "#" in value:
        # `String.split("#", 2)` truncates; it does not keep the tail.
        code, state = value.split("#")[:2]
        return code, state

    if "code=" in value:
        query = parse_qs(value)
        return (query.get("code") or [None])[0], (query.get("state") or [None])[0]

    return value, None


def _format_error_details(error: BaseException | Any) -> str:
    """pi's `formatErrorDetails`. Node's `code` has no Python counterpart; `errno`
    does, on OSError."""
    if isinstance(error, BaseException):
        details = [f"{type(error).__name__}: {error}"]
        errno = getattr(error, "errno", None)
        if errno is not None:
            details.append(f"errno={errno}")
        if error.__cause__ is not None:
            details.append(f"cause={_format_error_details(error.__cause__)}")
        stack = "".join(traceback.format_exception(type(error), error, error.__traceback__)).rstrip()
        if stack:
            details.append(f"stack={stack}")
        return "; ".join(details)
    return str(error)


async def _post_json(url: str, body: dict[str, Any], cancel: CancelToken) -> str:
    # pi: `AbortSignal.any([signal, AbortSignal.timeout(30s)])` — the timeout
    # half lives in the transport's timeout_ms.
    response = await oauth_http.request(
        url,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        json_body=body,
        timeout_ms=TOKEN_TIMEOUT_MS,
        cancel=cancel,
    )
    response_body = response.text
    if not response.ok:
        raise RuntimeError(f"HTTP request failed. status={response.status}; url={url}; body={response_body}")
    return response_body


async def _exchange_authorization_code(
    code: str, state: str, verifier: str, redirect_uri: str, cancel: CancelToken
) -> OAuthCredential:
    try:
        response_body = await _post_json(
            TOKEN_URL,
            {
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "code": code,
                "state": state,
                "redirect_uri": redirect_uri,
                "code_verifier": verifier,
            },
            cancel,
        )
    except Exception as error:
        raise RuntimeError(
            f"Token exchange request failed. url={TOKEN_URL}; redirect_uri={redirect_uri}; "
            f"response_type=authorization_code; details={_format_error_details(error)}"
        ) from error

    try:
        token_data = json.loads(response_body)
    except ValueError as error:
        raise RuntimeError(
            f"Token exchange returned invalid JSON. url={TOKEN_URL}; body={response_body}; "
            f"details={_format_error_details(error)}"
        ) from error

    return OAuthCredential(
        refresh=token_data["refresh_token"],
        access=token_data["access_token"],
        expires=int(clock.now_ms() + token_data["expires_in"] * 1000 - 5 * 60 * 1000),
    )


async def _login_anthropic(interaction: ProviderAuthInteraction) -> OAuthCredential:
    pkce = generate_pkce()
    verifier, challenge = pkce.verifier, pkce.challenge
    callback: OAuthCallbackServer[str] | None
    try:
        callback = await start_oauth_callback_server(
            provider_name="Anthropic",
            host=CALLBACK_HOST,
            port=CALLBACK_PORT,
            path=CALLBACK_PATH,
            state=verifier,
            complete=keep_code,
            cancel=interaction.cancel,
        )
    except Exception:
        callback = None

    try:
        auth_params = urlencode(
            {
                "code": "true",
                "client_id": CLIENT_ID,
                "response_type": "code",
                "redirect_uri": REDIRECT_URI,
                "scope": SCOPES,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "state": verifier,
            }
        )
        interaction.notify(
            AuthEvent(
                type="auth_url",
                url=f"{AUTHORIZE_URL}?{auth_params}",
                instructions=(
                    "Complete login in your browser. If the browser is on another machine, "
                    "paste the final redirect URL here."
                ),
            )
        )

        result = await wait_for_callback_or_manual_input(
            interaction,
            callback,
            message="Complete login in your browser, or paste the authorization code / redirect URL here:",
            placeholder=REDIRECT_URI,
        )
        code: str | None
        state = verifier
        if result.type == "callback":
            code = result.value
        else:
            code, parsed_state = _parse_authorization_input(result.input)
            if parsed_state and parsed_state != verifier:
                raise RuntimeError("OAuth state mismatch")
            # pi's `parsed.state ?? verifier`: an empty state is kept.
            state = parsed_state if parsed_state is not None else verifier

        if not code:
            raise RuntimeError("Missing authorization code")
        interaction.notify(AuthEvent(type="progress", message="Exchanging authorization code for tokens..."))
        return await _exchange_authorization_code(code, state, verifier, REDIRECT_URI, interaction.cancel)
    finally:
        if callback is not None:
            callback.close()


async def _login_anthropic_copy_code(interaction: ProviderAuthInteraction) -> OAuthCredential:
    pkce = generate_pkce()
    verifier, challenge = pkce.verifier, pkce.challenge
    auth_params = urlencode(
        {
            "code": "true",
            "client_id": CLIENT_ID,
            "response_type": "code",
            "redirect_uri": COPY_CODE_REDIRECT_URI,
            "scope": SCOPES,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": verifier,
        }
    )
    interaction.notify(
        AuthEvent(
            type="auth_url",
            url=f"{AUTHORIZE_URL}?{auth_params}",
            instructions="Complete login in your browser, then copy the code Anthropic shows and paste it here.",
        )
    )

    value = await interaction.prompt(
        AuthPrompt(
            type="manual_code",
            message="Paste the code Anthropic shows after you sign in:",
            placeholder="code#state",
            cancel=interaction.cancel,
        )
    )
    code, parsed_state = _parse_authorization_input(value)
    if parsed_state and parsed_state != verifier:
        raise RuntimeError("OAuth state mismatch")
    if not code:
        raise RuntimeError("Missing authorization code")
    interaction.notify(AuthEvent(type="progress", message="Exchanging authorization code for tokens..."))
    return await _exchange_authorization_code(
        code,
        # pi's `parsed.state ?? verifier`: an empty state is kept.
        parsed_state if parsed_state is not None else verifier,
        verifier,
        COPY_CODE_REDIRECT_URI,
        interaction.cancel,
    )


async def _login(interaction: ProviderAuthInteraction, options: LoginOptions | None = None) -> OAuthCredential:
    method = await interaction.prompt(
        AuthPrompt(
            type="select",
            message="Select Anthropic login method:",
            options=[
                AuthPromptOption(id=ANTHROPIC_BROWSER_LOGIN_METHOD, label="Browser login (default)"),
                AuthPromptOption(id=ANTHROPIC_COPY_CODE_LOGIN_METHOD, label="Copy code login (headless)"),
            ],
        )
    )

    if method == ANTHROPIC_COPY_CODE_LOGIN_METHOD:
        return await _login_anthropic_copy_code(interaction)
    if method != ANTHROPIC_BROWSER_LOGIN_METHOD:
        raise RuntimeError(f"Unknown Anthropic login method: {method}")

    return await _login_anthropic(interaction)


async def _refresh_anthropic_token(refresh_token: str, cancel: CancelToken) -> OAuthCredential:
    try:
        response_body = await _post_json(
            TOKEN_URL,
            {"grant_type": "refresh_token", "client_id": CLIENT_ID, "refresh_token": refresh_token},
            cancel,
        )
    except Exception as error:
        raise RuntimeError(
            f"Anthropic token refresh request failed. url={TOKEN_URL}; details={_format_error_details(error)}"
        ) from error

    try:
        data = json.loads(response_body)
    except ValueError as error:
        raise RuntimeError(
            f"Anthropic token refresh returned invalid JSON. url={TOKEN_URL}; body={response_body}; "
            f"details={_format_error_details(error)}"
        ) from error

    return OAuthCredential(
        refresh=data["refresh_token"],
        access=data["access_token"],
        expires=int(clock.now_ms() + data["expires_in"] * 1000 - 5 * 60 * 1000),
    )


def _refresh(credential: OAuthCredential, cancel: CancelToken) -> Awaitable[OAuthCredential]:
    return _refresh_anthropic_token(credential.refresh, cancel)


async def _to_auth(credential: OAuthCredential) -> ModelAuth:
    return ModelAuth(api_key=credential.access)


anthropic_oauth = OAuthAuth(
    name="Anthropic (Claude Pro/Max)",
    is_subscription=True,
    login=_login,
    refresh=_refresh,
    to_auth=_to_auth,
)
