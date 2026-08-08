"""Port of pi's Anthropic OAuth flow (packages/ai/src/auth/oauth/anthropic.ts).

Claude Pro/Max login: PKCE against claude.ai, with a loopback callback server on
a fixed port racing a manual paste prompt, so a browser on another machine still
works.

pi's Node-only guard (`getNodeApis` refusing to run outside Node/Bun) has no
counterpart: there is no browser build to protect here.
"""

import base64
import json
import traceback
from dataclasses import dataclass
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
from pidrei_ai.auth.oauth.oauth_page import oauth_error_html, oauth_success_html
from pidrei_ai.auth.oauth.pkce import generate_pkce
from pidrei_ai.auth.types import (
    AuthEvent,
    AuthPrompt,
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
SCOPES = "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload"
TOKEN_TIMEOUT_MS = 30_000


@dataclass(slots=True)
class _CallbackServerInfo:
    server: CallbackServer
    redirect_uri: str
    code: OneShotValue

    def cancel_wait(self) -> None:
        self.code.settle(None)

    async def wait_for_code(self) -> dict[str, str] | None:
        return await self.code.wait()


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


async def _start_callback_server(expected_state: str) -> _CallbackServerInfo:
    code = OneShotValue()

    async def handle(request: CallbackRequest) -> CallbackResponse:
        if request.path != CALLBACK_PATH:
            return CallbackResponse(404, oauth_error_html("Callback route not found."))

        error = request.get("error")
        if error:
            return CallbackResponse(
                400, oauth_error_html("Anthropic authentication did not complete.", f"Error: {error}")
            )

        callback_code = request.get("code")
        state = request.get("state")
        if not callback_code or not state:
            return CallbackResponse(400, oauth_error_html("Missing code or state parameter."))

        if state != expected_state:
            return CallbackResponse(400, oauth_error_html("State mismatch."))

        code.settle({"code": callback_code, "state": state})
        return CallbackResponse(
            200, oauth_success_html("Anthropic authentication completed. You can close this window.")
        )

    server = await start_callback_server(host=CALLBACK_HOST, port=CALLBACK_PORT, handle=handle)
    return _CallbackServerInfo(server=server, redirect_uri=REDIRECT_URI, code=code)


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
    server = await _start_callback_server(verifier)
    manual_abort = CancelToken()
    unsubscribe_abort = interaction.cancel.on_cancel(lambda _reason: server.cancel_wait())
    manual: dict[str, Any] = {}
    prompt_done = tonio.Event()

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

    def read_manual_input() -> tuple[str | None, str | None]:
        parsed_code, parsed_state = _parse_authorization_input(manual["input"])
        if parsed_state and parsed_state != verifier:
            raise RuntimeError("OAuth state mismatch")
        return parsed_code, parsed_state or verifier

    code: str | None = None
    state: str | None = None
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

        tonio.spawn.without_tracking(run_manual_prompt())

        result = await server.wait_for_code()
        if manual.get("error") is not None:
            raise manual["error"]
        if result is not None and result.get("code"):
            code, state = result["code"], result.get("state")
        elif manual.get("input"):
            code, state = read_manual_input()

        if not code:
            await prompt_done.wait()
            if manual.get("error") is not None:
                raise manual["error"]
            if manual.get("input"):
                code, state = read_manual_input()

        if not code:
            raise RuntimeError("Missing authorization code")
        if not state:
            raise RuntimeError("Missing OAuth state")
        interaction.notify(AuthEvent(type="progress", message="Exchanging authorization code for tokens..."))
        return await _exchange_authorization_code(code, state, verifier, REDIRECT_URI, interaction.cancel)
    finally:
        unsubscribe_abort()
        manual_abort.cancel()
        server.server.close()


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


async def _refresh(credential: OAuthCredential, cancel: CancelToken) -> OAuthCredential:
    return await _refresh_anthropic_token(credential.refresh, cancel)


async def _to_auth(credential: OAuthCredential) -> ModelAuth:
    return ModelAuth(api_key=credential.access)


anthropic_oauth = OAuthAuth(
    name="Anthropic (Claude Pro/Max)",
    is_subscription=True,
    login=_login_anthropic,
    refresh=_refresh,
    to_auth=_to_auth,
)
