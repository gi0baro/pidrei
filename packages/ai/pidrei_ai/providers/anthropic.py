"""Port of pi's anthropic provider factory (packages/ai/src/providers/anthropic.ts)."""

from pidrei_ai.api.anthropic_messages_lazy import anthropic_messages_api
from pidrei_ai.auth.helpers import lazy_oauth
from pidrei_ai.auth.oauth.load import load_anthropic_oauth
from pidrei_ai.auth.types import (
    ApiKeyAuth,
    ApiKeyCredential,
    AuthContext,
    AuthPrompt,
    AuthResult,
    ModelAuth,
    ProviderAuth,
    ProviderAuthInteraction,
)
from pidrei_ai.env_api_keys import (
    ANTHROPIC_API_KEY_ENV,
    ANTHROPIC_AUTH_TOKEN_ENV,
    ANTHROPIC_OAUTH_TOKEN_ENV,
)
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import Provider, create_provider
from pidrei_ai.utils.cancel import CancelToken


def _anthropic_api_key_auth() -> ApiKeyAuth:
    async def login(interaction: ProviderAuthInteraction) -> ApiKeyCredential:
        interaction.cancel.raise_if_cancelled()
        key = await interaction.prompt(AuthPrompt(type="secret", message="Enter Anthropic API key"))
        interaction.cancel.raise_if_cancelled()
        return ApiKeyCredential(key=key)

    async def resolve(ctx: AuthContext, credential: ApiKeyCredential | None, cancel: CancelToken) -> AuthResult | None:
        cancel.raise_if_cancelled()
        if credential is not None and credential.key:
            return AuthResult(auth=ModelAuth(api_key=credential.key), env=credential.env, source="stored credential")

        auth_token = await ctx.env(ANTHROPIC_AUTH_TOKEN_ENV)
        cancel.raise_if_cancelled()
        if auth_token:
            return AuthResult(
                auth=ModelAuth(headers={"Authorization": f"Bearer {auth_token}"}),
                source=ANTHROPIC_AUTH_TOKEN_ENV,
            )

        for env_var in (ANTHROPIC_OAUTH_TOKEN_ENV, ANTHROPIC_API_KEY_ENV):
            api_key = await ctx.env(env_var)
            cancel.raise_if_cancelled()
            if api_key:
                return AuthResult(auth=ModelAuth(api_key=api_key), source=env_var)
        return None

    return ApiKeyAuth(name="Anthropic API key", resolve=resolve, login=login)


def anthropic_provider() -> Provider:
    return create_provider(
        id="anthropic",
        name="Anthropic",
        base_url="https://api.anthropic.com",
        auth=ProviderAuth(
            api_key=_anthropic_api_key_auth(),
            oauth=lazy_oauth(
                name="Anthropic (Claude Pro/Max)",
                is_subscription=True,
                load=load_anthropic_oauth,
            ),
        ),
        models=list(MODELS.get("anthropic", [])),
        api=anthropic_messages_api(),
    )
