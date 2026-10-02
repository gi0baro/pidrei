"""Port of pi's auth helpers (packages/ai/src/auth/helpers.ts)."""

from collections.abc import Awaitable, Callable

from tonio.colored import sync

from pidrei_ai.auth.types import (
    ApiKeyAuth,
    ApiKeyCredential,
    AuthContext,
    AuthPrompt,
    AuthResult,
    LoginOptions,
    ModelAuth,
    OAuthAuth,
    OAuthCredential,
    ProviderAuthInteraction,
)
from pidrei_utils.cancel import CancelToken


def env_api_key_auth(name: str, env_vars: list[str]) -> ApiKeyAuth:
    """Standard api-key auth: a stored credential key wins, otherwise the first
    set env var resolves. Includes a `login` that prompts for the key.
    Providers with non-standard resolution write their own `ApiKeyAuth`.
    """

    async def login(interaction: ProviderAuthInteraction) -> ApiKeyCredential:
        interaction.cancel.raise_if_cancelled()
        key = await interaction.prompt(AuthPrompt(type="secret", message=f"Enter {name}"))
        interaction.cancel.raise_if_cancelled()
        return ApiKeyCredential(key=key)

    async def resolve(ctx: AuthContext, credential: ApiKeyCredential | None, cancel: CancelToken) -> AuthResult | None:
        cancel.raise_if_cancelled()
        if credential is not None and credential.key:
            return AuthResult(auth=ModelAuth(api_key=credential.key), env=credential.env, source="stored credential")
        for env_var in env_vars:
            value = await ctx.env(env_var)
            cancel.raise_if_cancelled()
            if value:
                return AuthResult(auth=ModelAuth(api_key=value), source=env_var)
        return None

    return ApiKeyAuth(name=name, resolve=resolve, login=login)


def lazy_oauth(
    *,
    name: str,
    load: Callable[[], Awaitable[OAuthAuth]],
    is_subscription: bool | None = None,
    login_label: str | None = None,
) -> OAuthAuth:
    """Wraps a lazily imported `OAuthAuth` so provider definitions can advertise
    OAuth without importing the flow implementation; it loads on first
    `login`/`refresh`/`to_auth` call.

    pi memoizes the load's promise, so concurrent first calls share one load.
    Here the load runs under a lock: callers that arrive meanwhile wait for it
    and use the flow it published. A load that fails publishes nothing, and
    the next call loads again.
    """
    loaded: OAuthAuth | None = None
    load_lock = sync.Lock()

    async def _loaded() -> OAuthAuth:
        nonlocal loaded
        flow = loaded
        if flow is not None:
            return flow
        async with load_lock:
            # The caller that held the lock may have loaded it.
            if loaded is None:
                loaded = await load()
            return loaded

    async def login(interaction: ProviderAuthInteraction, options: LoginOptions | None = None) -> OAuthCredential:
        return await (await _loaded()).login(interaction, options)

    async def refresh(credential: OAuthCredential, cancel: CancelToken) -> OAuthCredential:
        return await (await _loaded()).refresh(credential, cancel)

    async def to_auth(credential: OAuthCredential) -> ModelAuth:
        return await (await _loaded()).to_auth(credential)

    return OAuthAuth(
        name=name,
        login=login,
        refresh=refresh,
        to_auth=to_auth,
        is_subscription=is_subscription,
        login_label=login_label,
    )
