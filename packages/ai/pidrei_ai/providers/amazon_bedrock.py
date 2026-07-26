"""Port of pi's amazon-bedrock provider factory (packages/ai/src/providers/amazon-bedrock.ts)."""

from pidrei_ai.api.bedrock_converse_stream_lazy import bedrock_converse_stream_api
from pidrei_ai.auth.types import (
    ApiKeyAuth,
    ApiKeyCredential,
    AuthContext,
    AuthEvent,
    AuthInfoLink,
    AuthInteraction,
    AuthPrompt,
    AuthPromptOption,
    AuthResult,
    ModelAuth,
    ProviderAuth,
)
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import Provider, create_provider


async def _login(interaction: AuthInteraction) -> ApiKeyCredential:
    method = await interaction.prompt(
        AuthPrompt(
            type="select",
            message="Select Amazon Bedrock authentication method:",
            options=[
                AuthPromptOption(id="bearer-token", label="Bearer token"),
                AuthPromptOption(id="aws-profile", label="AWS profile"),
                AuthPromptOption(id="credential-chain", label="Existing AWS credential chain"),
            ],
        )
    )
    if method == "bearer-token":
        return ApiKeyCredential(
            key=await interaction.prompt(AuthPrompt(type="secret", message="Enter Amazon Bedrock bearer token"))
        )
    interaction.notify(
        AuthEvent(
            type="info",
            message="Amazon Bedrock supports AWS profiles, IAM credentials, and role-based credentials.",
            links=[
                AuthInfoLink(
                    label="AWS credential provider chain",
                    url="https://docs.aws.amazon.com/sdkref/latest/guide/standardized-credentials.html",
                )
            ],
        )
    )
    if method == "aws-profile":
        return ApiKeyCredential(
            env={"AWS_PROFILE": await interaction.prompt(AuthPrompt(type="text", message="Enter AWS profile name"))}
        )
    if method != "credential-chain":
        raise RuntimeError(f"Unknown Amazon Bedrock auth method: {method}")
    await interaction.prompt(AuthPrompt(type="text", message="Configure AWS credentials, then press Enter to continue"))
    return ApiKeyCredential()


async def _resolve(ctx: AuthContext, credential: ApiKeyCredential | None) -> AuthResult | None:
    credential_env = (credential.env if credential is not None else None) or {}
    if credential is not None and credential.key:
        return AuthResult(auth=ModelAuth(api_key=credential.key), env=credential.env, source="stored credential")
    if await ctx.env("AWS_BEARER_TOKEN_BEDROCK"):
        return AuthResult(auth=ModelAuth(), source="AWS_BEARER_TOKEN_BEDROCK")
    if credential_env.get("AWS_PROFILE") or await ctx.env("AWS_PROFILE"):
        return AuthResult(
            auth=ModelAuth(),
            env=credential.env if credential is not None else None,
            source="stored credential" if credential_env.get("AWS_PROFILE") else "AWS_PROFILE",
        )
    if await ctx.env("AWS_ACCESS_KEY_ID") and await ctx.env("AWS_SECRET_ACCESS_KEY"):
        return AuthResult(auth=ModelAuth(), source="AWS access keys")
    if await ctx.env("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"):
        return AuthResult(auth=ModelAuth(), source="ECS task role")
    if await ctx.env("AWS_CONTAINER_CREDENTIALS_FULL_URI"):
        return AuthResult(auth=ModelAuth(), source="ECS task role")
    if await ctx.env("AWS_WEB_IDENTITY_TOKEN_FILE"):
        return AuthResult(auth=ModelAuth(), source="web identity token")
    return None


def _bedrock_auth() -> ApiKeyAuth:
    """Bedrock accepts a bearer token or the AWS SDK's default credential chain.

    The login flow can store a token/profile choice; `resolve` also detects
    ambient AWS credentials without copying them into pidrei's credential store.
    """
    return ApiKeyAuth(name="AWS credentials or bearer token", login=_login, resolve=_resolve)


def amazon_bedrock_provider() -> Provider:
    return create_provider(
        id="amazon-bedrock",
        name="Amazon Bedrock",
        auth=ProviderAuth(api_key=_bedrock_auth()),
        models=list(MODELS.get("amazon-bedrock", [])),
        api=bedrock_converse_stream_api(),
    )
