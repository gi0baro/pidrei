"""Port of pi's google-vertex provider factory (packages/ai/src/providers/google-vertex.ts)."""

from pidrei_ai.api.google_vertex_lazy import google_vertex_api
from pidrei_ai.auth.types import (
    ApiKeyAuth,
    ApiKeyCredential,
    AuthContext,
    AuthEvent,
    AuthInfoLink,
    AuthPrompt,
    AuthPromptOption,
    AuthResult,
    ModelAuth,
    ProviderAuth,
    ProviderAuthInteraction,
)
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import Provider, create_provider
from pidrei_ai.utils.cancel import CancelToken


VERTEX_ADC_PATH = "~/.config/gcloud/application_default_credentials.json"


async def _login(interaction: ProviderAuthInteraction) -> ApiKeyCredential:
    interaction.cancel.raise_if_cancelled()
    method = await interaction.prompt(
        AuthPrompt(
            type="select",
            message="Select Google Vertex AI authentication method:",
            options=[
                AuthPromptOption(id="api-key", label="Google Cloud API key"),
                AuthPromptOption(id="adc", label="Application Default Credentials"),
                AuthPromptOption(id="service-account", label="Service account credentials file"),
            ],
        )
    )
    interaction.cancel.raise_if_cancelled()
    if method == "api-key":
        return ApiKeyCredential(
            key=await interaction.prompt(AuthPrompt(type="secret", message="Enter Google Cloud API key"))
        )
    if method not in ("adc", "service-account"):
        raise RuntimeError(f"Unknown Google Vertex AI auth method: {method}")
    interaction.notify(
        AuthEvent(
            type="info",
            message=(
                "Run `gcloud auth application-default login`, then provide the project and location."
                if method == "adc"
                else "Provide a service account credentials file, project, and location."
            ),
            links=[
                AuthInfoLink(
                    label="Application Default Credentials",
                    url="https://cloud.google.com/docs/authentication/provide-credentials-adc",
                )
            ],
        )
    )
    project = await interaction.prompt(AuthPrompt(type="text", message="Enter Google Cloud project ID"))
    location = await interaction.prompt(AuthPrompt(type="text", message="Enter Google Cloud location"))
    credentials_path = (
        await interaction.prompt(AuthPrompt(type="text", message="Enter service account credentials file path"))
        if method == "service-account"
        else None
    )
    return ApiKeyCredential(
        env={
            "GOOGLE_CLOUD_PROJECT": project,
            "GOOGLE_CLOUD_LOCATION": location,
            **({"GOOGLE_APPLICATION_CREDENTIALS": credentials_path} if credentials_path else {}),
        }
    )


async def _resolve(ctx: AuthContext, credential: ApiKeyCredential | None, cancel: CancelToken) -> AuthResult | None:
    async def env(name: str) -> str | None:
        cancel.raise_if_cancelled()
        value = await ctx.env(name)
        cancel.raise_if_cancelled()
        return value

    stored_key = credential.key if credential is not None else None
    key = stored_key or await env("GOOGLE_CLOUD_API_KEY")
    if key:
        return AuthResult(
            auth=ModelAuth(api_key=key),
            source="stored credential" if stored_key else "GOOGLE_CLOUD_API_KEY",
        )

    credential_env = (credential.env if credential is not None else None) or {}
    adc_path = credential_env.get("GOOGLE_APPLICATION_CREDENTIALS") or await env("GOOGLE_APPLICATION_CREDENTIALS")
    cancel.raise_if_cancelled()
    has_credentials = await ctx.file_exists(adc_path or VERTEX_ADC_PATH)
    cancel.raise_if_cancelled()
    project = (
        credential_env.get("GOOGLE_CLOUD_PROJECT") or await env("GOOGLE_CLOUD_PROJECT") or await env("GCLOUD_PROJECT")
    )
    location = credential_env.get("GOOGLE_CLOUD_LOCATION") or await env("GOOGLE_CLOUD_LOCATION")
    if has_credentials and project and location:
        return AuthResult(
            auth=ModelAuth(),
            env=credential.env if credential is not None else None,
            source="stored credential" if credential is not None else "gcloud application default credentials",
        )
    return None


def _vertex_auth() -> ApiKeyAuth:
    """Vertex accepts an explicit API key or Application Default Credentials
    (`gcloud auth application-default login`). ADC additionally requires project
    and location env vars, which the implementation reads itself.
    """
    return ApiKeyAuth(name="Google Cloud credentials", login=_login, resolve=_resolve)


def google_vertex_provider() -> Provider:
    return create_provider(
        id="google-vertex",
        name="Google Vertex AI",
        auth=ProviderAuth(api_key=_vertex_auth()),
        models=list(MODELS.get("google-vertex", [])),
        api=google_vertex_api(),
    )
