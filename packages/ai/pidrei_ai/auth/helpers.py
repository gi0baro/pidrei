"""Port of pi's auth helpers (packages/ai/src/auth/helpers.ts)."""

from collections.abc import Awaitable, Callable

from pidrei_ai.auth.types import (
    ApiKeyAuth,
    ApiKeyCredential,
    AuthContext,
    AuthInteraction,
    AuthPrompt,
    AuthResult,
    ModelAuth,
    OAuthAuth,
    OAuthCredential,
)
from pidrei_ai.utils.cancel import CancelToken


def env_api_key_auth(name: str, env_vars: list[str]) -> ApiKeyAuth:
    """Standard api-key auth: a stored credential key wins, otherwise the first
    set env var resolves. Includes a `login` that prompts for the key.
    Providers with non-standard resolution write their own `ApiKeyAuth`.
    """

    async def login(interaction: AuthInteraction) -> ApiKeyCredential:
        key = await interaction.prompt(AuthPrompt(type="secret", message=f"Enter {name}"))
        return ApiKeyCredential(key=key)

    async def resolve(ctx: AuthContext, credential: ApiKeyCredential | None) -> AuthResult | None:
        if credential is not None and credential.key:
            return AuthResult(auth=ModelAuth(api_key=credential.key), env=credential.env, source="stored credential")
        for env_var in env_vars:
            value = await ctx.env(env_var)
            if value:
                return AuthResult(auth=ModelAuth(api_key=value), source=env_var)
        return None

    return ApiKeyAuth(name=name, resolve=resolve, login=login)


def lazy_oauth(
    *,
    name: str,
    load: Callable[[], Awaitable[OAuthAuth]],
    login_label: str | None = None,
) -> OAuthAuth:
    """Wraps a lazily imported `OAuthAuth` so provider definitions can advertise
    OAuth without importing the flow implementation; it loads on first
    `login`/`refresh`/`to_auth` call.
    """
    loaded: list[OAuthAuth | None] = [None]

    async def _loaded() -> OAuthAuth:
        if loaded[0] is None:
            loaded[0] = await load()
        return loaded[0]

    async def login(interaction: AuthInteraction) -> OAuthCredential:
        return await (await _loaded()).login(interaction)

    async def refresh(credential: OAuthCredential, cancel: CancelToken | None) -> OAuthCredential:
        return await (await _loaded()).refresh(credential, cancel)

    async def to_auth(credential: OAuthCredential) -> ModelAuth:
        return await (await _loaded()).to_auth(credential)

    return OAuthAuth(name=name, login=login, refresh=refresh, to_auth=to_auth, login_label=login_label)
