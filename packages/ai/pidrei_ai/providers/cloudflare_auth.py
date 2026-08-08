"""Port of pi's cloudflare-auth.ts."""

from pidrei_ai.auth.types import (
    ApiKeyAuth,
    ApiKeyCredential,
    AuthContext,
    AuthInteraction,
    AuthPrompt,
    AuthResult,
    ModelAuth,
)
from pidrei_ai.types import ProviderEnv
from pidrei_ai.utils.cancel import CancelToken


CLOUDFLARE_API_KEY = "CLOUDFLARE_API_KEY"
CLOUDFLARE_ACCOUNT_ID = "CLOUDFLARE_ACCOUNT_ID"
CLOUDFLARE_GATEWAY_ID = "CLOUDFLARE_GATEWAY_ID"


async def _resolve_value(
    name: str, ctx: AuthContext, credential: ApiKeyCredential | None, cancel: CancelToken
) -> str | None:
    # Per-field merge: prefer the credential value, fall back to ambient env. A
    # credential carrying only the API key must still pick up the account /
    # gateway id from the environment.
    from_credential = None
    if credential is not None:
        from_credential = credential.key if name == CLOUDFLARE_API_KEY else (credential.env or {}).get(name)
    if from_credential is not None:
        return from_credential
    cancel.raise_if_cancelled()
    value = await ctx.env(name)
    cancel.raise_if_cancelled()
    return value


async def _resolve_cloudflare_env(
    kind: str, ctx: AuthContext, credential: ApiKeyCredential | None, cancel: CancelToken
) -> tuple[str, ProviderEnv, str] | None:
    """Returns `(api key, provider env, source)`."""
    api_key = await _resolve_value(CLOUDFLARE_API_KEY, ctx, credential, cancel)
    account_id = await _resolve_value(CLOUDFLARE_ACCOUNT_ID, ctx, credential, cancel)
    gateway_id = await _resolve_value(CLOUDFLARE_GATEWAY_ID, ctx, credential, cancel) if kind == "ai-gateway" else None

    if not api_key or not account_id or (kind == "ai-gateway" and not gateway_id):
        return None

    env: ProviderEnv = {CLOUDFLARE_ACCOUNT_ID: account_id}
    if gateway_id:
        env[CLOUDFLARE_GATEWAY_ID] = gateway_id
    return api_key, env, "stored credential" if credential is not None else CLOUDFLARE_API_KEY


async def _workers_ai_login(interaction: AuthInteraction) -> ApiKeyCredential:
    key = await interaction.prompt(AuthPrompt(type="secret", message="Enter Cloudflare API key"))
    account_id = await interaction.prompt(AuthPrompt(type="text", message="Enter Cloudflare account ID"))
    return ApiKeyCredential(key=key, env={CLOUDFLARE_ACCOUNT_ID: account_id})


async def _workers_ai_resolve(
    ctx: AuthContext, credential: ApiKeyCredential | None, cancel: CancelToken
) -> AuthResult | None:
    resolved = await _resolve_cloudflare_env("workers-ai", ctx, credential, cancel)
    if resolved is None:
        return None
    api_key, env, source = resolved
    return AuthResult(auth=ModelAuth(api_key=api_key), env=env, source=source)


def cloudflare_workers_ai_auth() -> ApiKeyAuth:
    return ApiKeyAuth(name="Cloudflare API key", login=_workers_ai_login, resolve=_workers_ai_resolve)


async def _ai_gateway_login(interaction: AuthInteraction) -> ApiKeyCredential:
    key = await interaction.prompt(AuthPrompt(type="secret", message="Enter Cloudflare API key"))
    account_id = await interaction.prompt(AuthPrompt(type="text", message="Enter Cloudflare account ID"))
    gateway_id = await interaction.prompt(AuthPrompt(type="text", message="Enter Cloudflare AI Gateway ID"))
    return ApiKeyCredential(key=key, env={CLOUDFLARE_ACCOUNT_ID: account_id, CLOUDFLARE_GATEWAY_ID: gateway_id})


async def _ai_gateway_resolve(
    ctx: AuthContext, credential: ApiKeyCredential | None, cancel: CancelToken
) -> AuthResult | None:
    resolved = await _resolve_cloudflare_env("ai-gateway", ctx, credential, cancel)
    if resolved is None:
        return None
    api_key, env, source = resolved
    return AuthResult(
        # The gateway carries its own authorization header; the upstream
        # provider's own auth headers are suppressed with an explicit None.
        auth=ModelAuth(
            headers={
                "cf-aig-authorization": f"Bearer {api_key}",
                "Authorization": None,
                "x-api-key": None,
            }
        ),
        env=env,
        source=source,
    )


def cloudflare_ai_gateway_auth() -> ApiKeyAuth:
    return ApiKeyAuth(name="Cloudflare API key", login=_ai_gateway_login, resolve=_ai_gateway_resolve)
