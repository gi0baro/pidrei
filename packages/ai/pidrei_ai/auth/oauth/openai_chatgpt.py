"""Port of pi's Sign in with ChatGPT flow (packages/ai/src/auth/oauth/openai-chatgpt.ts).

OpenAI Responses API token sharing: this public-client flow uses no client
secret and sends the resulting user access token directly to api.openai.com.
The issued client ID and the granted scopes land in `OAuthCredential.extra`
under pi's keys (`clientId`, `scopes`), which is what auth.json stores.

pi keeps its own `createServer` here rather than the shared callback server
(fixed port 1455, a callback that carries `client_id`); this module keeps its
own handler on `callback_server`'s lower layer for the same reason.

The agent name hint is pidrei's, not pi's "Pi": OpenAI shows it to the user as
the connected app (see `utils/user_agent.py`).
"""

import math
import re
import secrets
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Literal
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
from pidrei_ai.auth.oauth.pkce import generate_pkce
from pidrei_ai.auth.types import (
    AuthEvent,
    AuthPrompt,
    LoginOptions,
    ModelAuth,
    OAuthAuth,
    OAuthCredential,
    ProviderAuthInteraction,
)
from pidrei_ai.utils import clock
from pidrei_ai.utils.cancel import CancelToken, combine_cancel_tokens
from pidrei_ai.utils.oauth_page import oauth_error_html, oauth_success_html
from pidrei_ai.utils.provider_env import get_provider_env_value
from pidrei_ai.utils.user_agent import CLIENT_NAME


# every login registers a new client with this ID; OpenAI returns the issued client ID in the callback
DYNAMIC_CLIENT_ID = "dynamic_agent_client"
AGENT_NAME_HINT = CLIENT_NAME
UUID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)
AUTHORIZE_URL = "https://auth.openai.com/api/accounts/authorize"
TOKEN_URL = "https://auth.openai.com/api/accounts/oauth/token"  # noqa: S105 - an endpoint, not a secret
RESOURCE = "https://api.openai.com/v1"
CALLBACK_HOST = get_provider_env_value("PIDREI_OAUTH_CALLBACK_HOST") or "127.0.0.1"
CALLBACK_PORT = 1455
CALLBACK_PATH = "/auth/callback"
REDIRECT_URI = f"http://127.0.0.1:{CALLBACK_PORT}{CALLBACK_PATH}"
DIRECT_TOKEN_SCOPE = "chatgpt.tokens.use.direct"  # noqa: S105 - a scope name, not a secret
SCOPE = f"openid profile email offline_access resource.invoke {DIRECT_TOKEN_SCOPE}"
# Refresh this long before the real expiry so a request never starts with a token about to expire.
EXPIRY_MARGIN_MS = 3 * 60 * 1000


@dataclass(slots=True, frozen=True)
class _AuthorizationResult:
    code: str
    client_id: str


def _random_value() -> str:
    """`randomBytes(32).toString("base64url")`."""
    return secrets.token_urlsafe(32)


def _authorization_result_from_callback(query: dict[str, str], expected_state: str) -> _AuthorizationResult:
    code = query.get("code")
    if not code:
        raise RuntimeError("Missing authorization code")
    state = query.get("state")
    if not state:
        raise RuntimeError("Missing OAuth state")
    if state != expected_state:
        raise RuntimeError("OAuth state mismatch")
    client_id = (query.get("client_id") or "").strip()
    if not client_id:
        raise RuntimeError("OpenAI OAuth registration callback did not contain an issued client ID")
    return _AuthorizationResult(code=code, client_id=client_id)


def _authorization_result_from_manual_input(input_value: str, expected_state: str) -> _AuthorizationResult:
    # `new URL(input)` throws on anything that is not an absolute URL.
    try:
        url = urlsplit(input_value.strip())
        origin = (url.scheme.lower(), url.hostname, url.port)
    except ValueError:
        raise RuntimeError("Paste the full callback URL from the browser") from None
    if not url.scheme:
        raise RuntimeError("Paste the full callback URL from the browser")
    if origin != ("http", "127.0.0.1", CALLBACK_PORT) or url.path != CALLBACK_PATH:
        raise RuntimeError(f"The pasted callback URL must start with {REDIRECT_URI}")
    query = {name: values[0] for name, values in parse_qs(url.query, keep_blank_values=True).items()}
    error = query.get("error")
    if error:
        raise RuntimeError(f"ChatGPT authorization failed: {error}")
    return _authorization_result_from_callback(query, expected_state)


async def _start_callback_server(expected_state: str, result: OneShotValue) -> CallbackServer:
    """pi's `startCallbackServer`: its `result` promise is `result` here, shared
    with the manual prompt as pi's `Promise.race` over both."""

    async def handle(request: CallbackRequest) -> CallbackResponse:
        try:
            if request.path != CALLBACK_PATH:
                return CallbackResponse(404, oauth_error_html("Callback route not found."))

            error = request.get("error")
            if error:
                result.settle(("error", RuntimeError(f"ChatGPT authorization failed: {error}")))
                return CallbackResponse(400, oauth_error_html("ChatGPT was not connected.", f"Error: {error}"))

            try:
                authorization_result = _authorization_result_from_callback(request.query, expected_state)
            except Exception as failure:
                return CallbackResponse(400, oauth_error_html(str(failure)))

            result.settle(("ok", authorization_result))
            return CallbackResponse(
                200, oauth_success_html("ChatGPT authentication completed. You can close this window.")
            )
        except Exception:
            return CallbackResponse(500, oauth_error_html("Internal error while processing the callback."))

    return await start_callback_server(host=CALLBACK_HOST, port=CALLBACK_PORT, handle=handle)


async def _request_token(form: dict[str, str], cancel: CancelToken | None) -> dict[str, Any]:
    response = await oauth_http.request(
        TOKEN_URL,
        headers={"accept": "application/json", "content-type": "application/x-www-form-urlencoded"},
        form=form,
        cancel=cancel,
    )
    if not response.ok:
        try:
            status_text = HTTPStatus(response.status).phrase
        except ValueError:
            status_text = ""
        raise RuntimeError(f"OpenAI OAuth token request failed ({response.status}): {response.text or status_text}")
    data = response.json()
    if isinstance(data, dict):
        return data
    raise RuntimeError("OpenAI OAuth token response must be an object")


def _require_token_string(value: Any, field: Literal["access_token", "refresh_token", "scope"]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"OpenAI OAuth token response has invalid {field}")
    return value


def _credential_from_token_response(token: dict[str, Any], client_id: str) -> OAuthCredential:
    access = _require_token_string(token.get("access_token"), "access_token")
    refresh = _require_token_string(token.get("refresh_token"), "refresh_token")
    scope = _require_token_string(token.get("scope"), "scope")
    expires_in = token.get("expires_in")
    if (
        isinstance(expires_in, bool)
        or not isinstance(expires_in, int | float)
        or not math.isfinite(expires_in)
        or expires_in <= 0
    ):
        raise RuntimeError("OpenAI OAuth token response has invalid expires_in")
    scopes = scope.split()
    if DIRECT_TOKEN_SCOPE not in scopes:
        raise RuntimeError(f"OpenAI OAuth grant did not include {DIRECT_TOKEN_SCOPE}")
    return OAuthCredential(
        access=access,
        refresh=refresh,
        expires=int(clock.now_ms() + expires_in * 1000 - EXPIRY_MARGIN_MS),
        extra={"clientId": client_id, "scopes": scopes},
    )


async def _exchange_authorization_code(
    code: str, verifier: str, client_id: str, cancel: CancelToken
) -> OAuthCredential:
    token = await _request_token(
        {
            "grant_type": "authorization_code",
            "client_id": client_id,
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": REDIRECT_URI,
            "resource": RESOURCE,
        },
        cancel,
    )
    # pidrei does not use the ID token to identify the user or read profile data.
    # Keep the presence check as part of the token-response contract.
    id_token = token.get("id_token")
    if not isinstance(id_token, str) or not id_token.strip():
        raise RuntimeError("OpenAI OAuth token response did not contain an ID token")
    return _credential_from_token_response(token, client_id)


async def _refresh_access_token(credential: OAuthCredential, cancel: CancelToken | None) -> OAuthCredential:
    client_id = credential.extra.get("clientId")
    if not isinstance(client_id, str) or not client_id.strip():
        raise RuntimeError("Stored OpenAI OAuth credential does not contain an issued client ID; reconnect ChatGPT")
    token = await _request_token(
        {
            "grant_type": "refresh_token",
            "client_id": client_id,
            "refresh_token": credential.refresh,
            "resource": RESOURCE,
        },
        cancel,
    )
    return _credential_from_token_response(token, client_id)


def _agent_host_id(device_id: str | None) -> str:
    """OpenAI identifies each installation ("agent host") by a stable URI such as `urn:uuid:<uuid>`."""
    if not device_id or not UUID_PATTERN.match(device_id):
        raise RuntimeError("Sign in with ChatGPT requires a device ID (UUID) for this installation")
    return f"urn:uuid:{device_id.lower()}"


async def _login_openai_chatgpt(
    interaction: ProviderAuthInteraction, options: LoginOptions | None = None
) -> OAuthCredential:
    host_id = _agent_host_id(
        options.get_device_id() if options is not None and options.get_device_id is not None else None
    )
    pkce = generate_pkce()
    state = _random_value()
    nonce = _random_value()
    result = OneShotValue()
    callback: CallbackServer | None = None
    try:
        callback = await _start_callback_server(state, result)
    except Exception as error:
        interaction.notify(
            AuthEvent(
                type="info",
                message=f"Could not listen on {REDIRECT_URI}; paste the final redirect URL to continue. {error}",
            )
        )

    authorization_url = f"{AUTHORIZE_URL}?" + urlencode(
        {
            "client_id": DYNAMIC_CLIENT_ID,
            "agent_name_hint": AGENT_NAME_HINT,
            "ext_agent_host_id": host_id,
            "response_type": "code",
            "redirect_uri": REDIRECT_URI,
            "resource": RESOURCE,
            "scope": SCOPE,
            "state": state,
            "code_challenge": pkce.challenge,
            "code_challenge_method": "S256",
            "nonce": nonce,
        }
    )
    interaction.notify(
        AuthEvent(
            type="auth_url",
            url=authorization_url,
            instructions=(
                "Complete sign-in in your browser. If the callback does not complete, paste the final redirect URL here."
            ),
        )
    )

    manual_abort = CancelToken()
    manual_cancel = combine_cancel_tokens(manual_abort, interaction.cancel)

    async def run_manual_prompt() -> None:
        try:
            input_value = await interaction.prompt(
                AuthPrompt(
                    type="manual_code",
                    message="Complete login in your browser, or paste the final redirect URL here:",
                    placeholder=REDIRECT_URI,
                    cancel=manual_cancel.token,
                )
            )
            result.settle(("ok", _authorization_result_from_manual_input(input_value, state)))
        except Exception as error:
            result.settle(("error", error))

    tonio.spawn.without_tracking(run_manual_prompt())

    try:
        # Without a callback server only the manual prompt settles `result`: pi's
        # `callback ? Promise.race([callback.result, manualCode]) : manualCode`.
        outcome, value = await result.wait()
        if outcome == "error":
            raise value
        interaction.notify(AuthEvent(type="progress", message="Exchanging authorization code for tokens..."))
        return await _exchange_authorization_code(value.code, pkce.verifier, value.client_id, interaction.cancel)
    except Exception:
        if interaction.cancel.cancelled:
            raise RuntimeError("Login cancelled") from None
        raise
    finally:
        manual_abort.cancel()
        manual_cancel.cleanup()
        if callback is not None:
            callback.close()
            # close() only stops accepting new connections. Browsers open spare connections ahead of
            # time, and one that has not sent a request yet stays open and attached to this server.
            # A later login in the same process starts a new server with a new state, but the browser
            # may send that login's callback over the spare connection. This server would then handle
            # it and reject it with "OAuth state mismatch", and the new login would never see it.
            callback.close_all_connections()


async def _to_auth(credential: OAuthCredential) -> ModelAuth:
    return ModelAuth(api_key=credential.access)


openai_chatgpt_oauth = OAuthAuth(
    name="OpenAI (ChatGPT subscription)",
    is_subscription=True,
    login_label="Sign in with ChatGPT",
    login=_login_openai_chatgpt,
    refresh=_refresh_access_token,
    to_auth=_to_auth,
)
