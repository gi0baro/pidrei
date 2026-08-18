"""Custom Provider Example

Registers a custom provider with:
- A custom API identifier ("custom-anthropic-api") dispatched through the
  provider's api dict
- OAuth support for /login (a hand-written PKCE flow)
- API key support via environment variable
- Two model definitions

pi's version of this example hand-rolls the streaming client on top of the
Anthropic SDK. pidrei ships that wire implementation as
`pidrei_ai.api.anthropic_messages`, so this port delegates streaming to it —
including the Claude Code identity handling it applies when the resolved key
is an OAuth access token.

Usage:
    # With OAuth (run /login custom-anthropic first)
    pidrei -e ./examples/extensions/custom_provider_anthropic

    # With API key
    CUSTOM_ANTHROPIC_API_KEY=sk-ant-... pidrei -e ./examples/extensions/custom_provider_anthropic

Then use /model to select custom-anthropic/claude-sonnet-4-5
"""

import base64
from urllib.parse import urlencode

from pidrei_ai.api.anthropic_messages_lazy import anthropic_messages_api
from pidrei_ai.auth.oauth import http as oauth_http
from pidrei_ai.auth.oauth.pkce import generate_pkce
from pidrei_ai.auth.types import (
    ApiKeyAuth,
    ApiKeyCredential,
    AuthContext,
    AuthEvent,
    AuthPrompt,
    AuthResult,
    ModelAuth,
    OAuthAuth,
    OAuthCredential,
    ProviderAuth,
    ProviderAuthInteraction,
)
from pidrei_ai.registry import create_provider
from pidrei_ai.types import Model, ModelCost
from pidrei_ai.utils import clock
from pidrei_ai.utils.cancel import CancelToken


PROVIDER_ID = "custom-anthropic"
API_ID = "custom-anthropic-api"
BASE_URL = "https://api.anthropic.com"
API_KEY_ENV = "CUSTOM_ANTHROPIC_API_KEY"

# =============================================================================
# OAuth implementation
#
# A deliberately minimal manual-paste flow, as in pi's example. pidrei's
# built-in Anthropic flow (pidrei_ai/auth/oauth/anthropic.py) is the full
# version, with a loopback callback server racing the paste prompt.
# =============================================================================

CLIENT_ID = base64.b64decode("OWQxYzI1MGEtZTYxYi00NGQ5LTg4ZWQtNTk0NGQxOTYyZjVl").decode("utf-8")
AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"  # noqa: S105 - an endpoint, not a secret
REDIRECT_URI = "https://console.anthropic.com/oauth/code/callback"
SCOPES = "org:create_api_key user:profile user:inference"


def _credential_from_token_response(data: dict) -> OAuthCredential:
    return OAuthCredential(
        refresh=data["refresh_token"],
        access=data["access_token"],
        expires=int(clock.now_ms() + data["expires_in"] * 1000 - 5 * 60 * 1000),
    )


async def _post_token(body: dict, cancel: CancelToken) -> dict:
    response = await oauth_http.request(
        TOKEN_URL,
        headers={"Content-Type": "application/json"},
        json_body=body,
        cancel=cancel,
    )
    if not response.ok:
        raise RuntimeError(f"Token request failed: {response.text}")
    return response.json()


async def _login_oauth(interaction: ProviderAuthInteraction) -> OAuthCredential:
    pkce = generate_pkce()

    auth_params = urlencode(
        {
            "code": "true",
            "client_id": CLIENT_ID,
            "response_type": "code",
            "redirect_uri": REDIRECT_URI,
            "scope": SCOPES,
            "code_challenge": pkce.challenge,
            "code_challenge_method": "S256",
            "state": pkce.verifier,
        }
    )
    interaction.notify(AuthEvent(type="auth_url", url=f"{AUTHORIZE_URL}?{auth_params}"))

    auth_code = await interaction.prompt(AuthPrompt(type="text", message="Paste the authorization code:"))
    code, _, state = auth_code.strip().partition("#")

    data = await _post_token(
        {
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "code": code,
            "state": state,
            "redirect_uri": REDIRECT_URI,
            "code_verifier": pkce.verifier,
        },
        interaction.cancel,
    )
    return _credential_from_token_response(data)


async def _refresh_oauth(credential: OAuthCredential, cancel: CancelToken) -> OAuthCredential:
    data = await _post_token(
        {
            "grant_type": "refresh_token",
            "client_id": CLIENT_ID,
            "refresh_token": credential.refresh,
        },
        cancel,
    )
    return _credential_from_token_response(data)


async def _to_auth(credential: OAuthCredential) -> ModelAuth:
    # The wire implementation detects sk-ant-oat tokens and switches to Bearer
    # auth with Claude Code identity headers on its own.
    return ModelAuth(api_key=credential.access)


# =============================================================================
# API key auth (environment variable, with a /login prompt fallback)
# =============================================================================


def _api_key_auth() -> ApiKeyAuth:
    async def login(interaction: ProviderAuthInteraction) -> ApiKeyCredential:
        interaction.cancel.raise_if_cancelled()
        key = await interaction.prompt(AuthPrompt(type="secret", message="Enter Anthropic API key"))
        interaction.cancel.raise_if_cancelled()
        return ApiKeyCredential(key=key)

    async def resolve(ctx: AuthContext, credential: ApiKeyCredential | None, cancel: CancelToken) -> AuthResult | None:
        cancel.raise_if_cancelled()
        if credential is not None and credential.key:
            return AuthResult(auth=ModelAuth(api_key=credential.key), env=credential.env, source="stored credential")
        api_key = await ctx.env(API_KEY_ENV)
        cancel.raise_if_cancelled()
        if api_key:
            return AuthResult(auth=ModelAuth(api_key=api_key), source=API_KEY_ENV)
        return None

    return ApiKeyAuth(name="Custom Anthropic API key", resolve=resolve, login=login)


# =============================================================================
# Models
# =============================================================================


def _model(id: str, name: str, cost: ModelCost, max_tokens: int) -> Model:
    return Model(
        id=id,
        name=name,
        api=API_ID,
        provider=PROVIDER_ID,
        base_url=BASE_URL,
        reasoning=True,
        input=["text", "image"],
        cost=cost,
        context_window=200_000,
        max_tokens=max_tokens,
    )


MODELS = [
    _model(
        "claude-opus-4-5",
        "Claude Opus 4.5 (Custom)",
        ModelCost(input=5, output=25, cache_read=0.5, cache_write=6.25),
        64_000,
    ),
    _model(
        "claude-sonnet-4-5",
        "Claude Sonnet 4.5 (Custom)",
        ModelCost(input=3, output=15, cache_read=0.3, cache_write=3.75),
        64_000,
    ),
]


# =============================================================================
# Extension Entry Point
# =============================================================================


def extension(pi):
    provider = create_provider(
        id=PROVIDER_ID,
        name="Custom Anthropic",
        base_url=BASE_URL,
        auth=ProviderAuth(
            api_key=_api_key_auth(),
            oauth=OAuthAuth(
                name="Custom Anthropic (Claude Pro/Max)",
                is_subscription=True,
                login=_login_oauth,
                refresh=_refresh_oauth,
                to_auth=_to_auth,
            ),
        ),
        models=list(MODELS),
        # An api dict dispatches on each model's `api` string; every model here
        # carries the custom identifier, so all of them stream through the
        # Anthropic Messages wire implementation.
        api={API_ID: anthropic_messages_api()},
    )
    pi.register_provider(provider)
