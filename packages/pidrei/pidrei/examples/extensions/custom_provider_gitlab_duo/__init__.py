"""GitLab Duo Provider Extension

Provides access to GitLab Duo AI models (Claude and GPT) through GitLab's AI
Gateway. Every request first exchanges the GitLab token for a short-lived
gateway token, then delegates to pidrei_ai's built-in Anthropic and OpenAI
wire implementations.

pi routes both backends through one custom `streamSimple`; here each backend
gets its own api identifier and the provider's api dict dispatches on it — a
model's `api` string picks the wrapper that wraps the right wire
implementation.

Usage:
    pidrei -e ./examples/extensions/custom_provider_gitlab_duo
    # Then /login gitlab-duo, or set GITLAB_TOKEN=glpat-...
"""

import uuid
from dataclasses import dataclass, replace
from urllib.parse import parse_qs, urlencode, urlsplit

from pidrei_ai.api.anthropic_messages_lazy import anthropic_messages_api
from pidrei_ai.api.lazy import _cancel_of, call_stream_into, lazy_stream
from pidrei_ai.api.openai_responses_lazy import openai_responses_api
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
from pidrei_ai.registry import Provider, create_provider
from pidrei_ai.types import (
    AnthropicMessagesCompat,
    Context,
    Model,
    ModelCost,
    SimpleStreamOptions,
    StreamOptions,
)
from pidrei_ai.utils import clock
from pidrei_ai.utils.cancel import CancelToken
from pidrei_ai.utils.event_stream import AssistantMessageEventStream


# =============================================================================
# Constants
# =============================================================================

PROVIDER_ID = "gitlab-duo"
ANTHROPIC_API_ID = "gitlab-duo-anthropic"
OPENAI_API_ID = "gitlab-duo-openai"

GITLAB_COM_URL = "https://gitlab.com"
AI_GATEWAY_URL = "https://cloud.gitlab.com"
ANTHROPIC_PROXY_URL = f"{AI_GATEWAY_URL}/ai/v1/proxy/anthropic/"
OPENAI_PROXY_URL = f"{AI_GATEWAY_URL}/ai/v1/proxy/openai/v1"

BUNDLED_CLIENT_ID = "da4edff2e6ebd2bc3208611e2768bc1c1dd7be791dc5ff26ca34ca9ee44f7d4b"
OAUTH_SCOPES = ["api"]
REDIRECT_URI = "http://127.0.0.1:8080/callback"
DIRECT_ACCESS_TTL_MS = 25 * 60 * 1000
GITLAB_TOKEN_ENV = "GITLAB_TOKEN"  # noqa: S105 - the variable's name, not a secret


# =============================================================================
# Models — exported for use by test.py
# =============================================================================


def _claude(id: str, name: str, cost: ModelCost, context_window: int, max_tokens: int) -> Model:
    return Model(
        id=id,
        name=name,
        api=ANTHROPIC_API_ID,
        provider=PROVIDER_ID,
        base_url=ANTHROPIC_PROXY_URL,
        reasoning=True,
        thinking_level_map={"xhigh": "max"},
        input=["text", "image"],
        cost=cost,
        context_window=context_window,
        max_tokens=max_tokens,
        # The gateway serves models whose ids do not always reveal their
        # adaptive-thinking support; force it, as pi's example does per request.
        compat=AnthropicMessagesCompat(force_adaptive_thinking=True),
    )


def _gpt(id: str, name: str, cost: ModelCost, context_window: int, max_tokens: int) -> Model:
    return Model(
        id=id,
        name=name,
        api=OPENAI_API_ID,
        provider=PROVIDER_ID,
        base_url=OPENAI_PROXY_URL,
        reasoning=True,
        input=["text", "image"],
        cost=cost,
        context_window=context_window,
        max_tokens=max_tokens,
    )


MODELS = [
    # Anthropic
    _claude(
        "claude-opus-4-8",
        "Claude Opus 4.8",
        ModelCost(input=5, output=25, cache_read=0.5, cache_write=6.25),
        1_000_000,
        128_000,
    ),
    _claude(
        "claude-sonnet-4-6",
        "Claude Sonnet 4.6",
        ModelCost(input=3, output=15, cache_read=0.3, cache_write=3.75),
        1_000_000,
        64_000,
    ),
    _claude(
        "claude-opus-4-5-20251101",
        "Claude Opus 4.5",
        ModelCost(input=15, output=75, cache_read=1.5, cache_write=18.75),
        200_000,
        32_000,
    ),
    _claude(
        "claude-sonnet-4-5-20250929",
        "Claude Sonnet 4.5",
        ModelCost(input=3, output=15, cache_read=0.3, cache_write=3.75),
        200_000,
        16_384,
    ),
    _claude(
        "claude-haiku-4-5-20251001",
        "Claude Haiku 4.5",
        ModelCost(input=1, output=5, cache_read=0.1, cache_write=1.25),
        200_000,
        8_192,
    ),
    # OpenAI (all use the Responses API)
    _gpt(
        "gpt-5.5-2026-04-23",
        "GPT-5.5",
        ModelCost(input=5, output=30, cache_read=0.5, cache_write=0),
        272_000,
        128_000,
    ),
    _gpt(
        "gpt-5.1-2025-11-13",
        "GPT-5.1",
        ModelCost(input=2.5, output=10, cache_read=0, cache_write=0),
        128_000,
        16_384,
    ),
    _gpt(
        "gpt-5-mini-2025-08-07",
        "GPT-5 Mini",
        ModelCost(input=0.15, output=0.6, cache_read=0, cache_write=0),
        128_000,
        16_384,
    ),
    _gpt(
        "gpt-5-codex",
        "GPT-5 Codex",
        ModelCost(input=2.5, output=10, cache_read=0, cache_write=0),
        128_000,
        16_384,
    ),
]


# =============================================================================
# Direct Access Token Cache
# =============================================================================


@dataclass(slots=True)
class _DirectAccessToken:
    token: str
    headers: dict[str, str]
    expires_at: int


_cached_direct_access: _DirectAccessToken | None = None


async def _get_direct_access_token(gitlab_access_token: str, cancel: CancelToken | None) -> _DirectAccessToken:
    global _cached_direct_access
    now = clock.now_ms()
    if _cached_direct_access is not None and _cached_direct_access.expires_at > now:
        return _cached_direct_access

    response = await oauth_http.request(
        f"{GITLAB_COM_URL}/api/v4/ai/third_party_agents/direct_access",
        headers={"Authorization": f"Bearer {gitlab_access_token}", "Content-Type": "application/json"},
        json_body={"feature_flags": {"DuoAgentPlatformNext": True}},
        cancel=cancel,
    )
    if not response.ok:
        if response.status == 403:
            raise RuntimeError(
                f"GitLab Duo access denied. Ensure GitLab Duo is enabled for your account. Error: {response.text}"
            )
        raise RuntimeError(f"Failed to get direct access token: {response.status} {response.text}")

    data = response.json()
    _cached_direct_access = _DirectAccessToken(
        token=data["token"],
        headers=dict(data["headers"]),
        expires_at=now + DIRECT_ACCESS_TTL_MS,
    )
    return _cached_direct_access


def _invalidate_direct_access_token() -> None:
    global _cached_direct_access
    _cached_direct_access = None


# =============================================================================
# OAuth
# =============================================================================


def _credential_from_token_response(data: dict) -> OAuthCredential:
    _invalidate_direct_access_token()
    return OAuthCredential(
        refresh=data["refresh_token"],
        access=data["access_token"],
        expires=(data["created_at"] + data["expires_in"]) * 1000 - 5 * 60 * 1000,
    )


async def _post_token_form(form: dict[str, str], cancel: CancelToken) -> dict:
    response = await oauth_http.request(
        f"{GITLAB_COM_URL}/oauth/token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        form=form,
        cancel=cancel,
    )
    if not response.ok:
        raise RuntimeError(f"Token request failed: {response.text}")
    return response.json()


async def _login_gitlab(interaction: ProviderAuthInteraction) -> OAuthCredential:
    pkce = generate_pkce()
    auth_params = urlencode(
        {
            "client_id": BUNDLED_CLIENT_ID,
            "redirect_uri": REDIRECT_URI,
            "response_type": "code",
            "scope": " ".join(OAUTH_SCOPES),
            "code_challenge": pkce.challenge,
            "code_challenge_method": "S256",
            "state": str(uuid.uuid4()),
        }
    )
    interaction.notify(AuthEvent(type="auth_url", url=f"{GITLAB_COM_URL}/oauth/authorize?{auth_params}"))

    callback_url = await interaction.prompt(AuthPrompt(type="text", message="Paste the callback URL:"))
    query = parse_qs(urlsplit(callback_url.strip()).query)
    code = (query.get("code") or [None])[0]
    if not code:
        raise RuntimeError("No authorization code found in callback URL")

    data = await _post_token_form(
        {
            "client_id": BUNDLED_CLIENT_ID,
            "grant_type": "authorization_code",
            "code": code,
            "code_verifier": pkce.verifier,
            "redirect_uri": REDIRECT_URI,
        },
        interaction.cancel,
    )
    return _credential_from_token_response(data)


async def _refresh_gitlab_token(credential: OAuthCredential, cancel: CancelToken) -> OAuthCredential:
    data = await _post_token_form(
        {
            "client_id": BUNDLED_CLIENT_ID,
            "grant_type": "refresh_token",
            "refresh_token": credential.refresh,
        },
        cancel,
    )
    return _credential_from_token_response(data)


async def _to_auth(credential: OAuthCredential) -> ModelAuth:
    return ModelAuth(api_key=credential.access)


# =============================================================================
# API key auth (GITLAB_TOKEN personal access token)
# =============================================================================


def _api_key_auth() -> ApiKeyAuth:
    async def login(interaction: ProviderAuthInteraction) -> ApiKeyCredential:
        interaction.cancel.raise_if_cancelled()
        key = await interaction.prompt(AuthPrompt(type="secret", message="Enter GitLab personal access token"))
        interaction.cancel.raise_if_cancelled()
        return ApiKeyCredential(key=key)

    async def resolve(ctx: AuthContext, credential: ApiKeyCredential | None, cancel: CancelToken) -> AuthResult | None:
        cancel.raise_if_cancelled()
        if credential is not None and credential.key:
            return AuthResult(auth=ModelAuth(api_key=credential.key), env=credential.env, source="stored credential")
        token = await ctx.env(GITLAB_TOKEN_ENV)
        cancel.raise_if_cancelled()
        if token:
            return AuthResult(auth=ModelAuth(api_key=token), source=GITLAB_TOKEN_ENV)
        return None

    return ApiKeyAuth(name="GitLab personal access token", resolve=resolve, login=login)


# =============================================================================
# Gateway wrapper
# =============================================================================


class _GitLabDuoApi:
    """Wraps a wire implementation behind the AI Gateway token exchange.

    The resolved provider auth arrives as `options.api_key` (the GitLab OAuth
    access token or personal access token); each request trades it for a
    short-lived gateway token and forwards to the wrapped implementation with
    the gateway's headers.
    """

    def __init__(self, inner):
        self._inner = inner

    def stream(self, model: Model, context: Context, options: StreamOptions | None = None):
        return self._delegate("stream", model, context, options)

    def stream_simple(self, model: Model, context: Context, options: SimpleStreamOptions | None = None):
        return self._delegate("stream_simple", model, context, options)

    def _delegate(self, method: str, model: Model, context: Context, options) -> AssistantMessageEventStream:
        async def _setup(stream):
            opts = options if options is not None else SimpleStreamOptions()
            gitlab_access_token = opts.api_key
            if not gitlab_access_token:
                raise RuntimeError(f"No GitLab access token. Run /login {PROVIDER_ID} or set {GITLAB_TOKEN_ENV}")

            direct_access = await _get_direct_access_token(gitlab_access_token, opts.cancel)
            headers = {**direct_access.headers, "Authorization": f"Bearer {direct_access.token}"}
            request_options = replace(opts, api_key="gitlab-duo", headers=headers)
            return call_stream_into(getattr(self._inner, method), model, context, request_options, into=stream)

        # lazy_stream turns setup failures (token exchange included) into
        # error events instead of raising out of the stream call.
        return lazy_stream(model, _setup, _cancel_of(options))


# =============================================================================
# Extension Entry Point
# =============================================================================


def gitlab_duo_provider() -> Provider:
    return create_provider(
        id=PROVIDER_ID,
        name="GitLab Duo",
        base_url=AI_GATEWAY_URL,
        auth=ProviderAuth(
            api_key=_api_key_auth(),
            oauth=OAuthAuth(
                name="GitLab Duo",
                login=_login_gitlab,
                refresh=_refresh_gitlab_token,
                to_auth=_to_auth,
            ),
        ),
        models=list(MODELS),
        api={
            ANTHROPIC_API_ID: _GitLabDuoApi(anthropic_messages_api()),
            OPENAI_API_ID: _GitLabDuoApi(openai_responses_api()),
        },
    )


def extension(pi):
    pi.register_provider(gitlab_duo_provider())
