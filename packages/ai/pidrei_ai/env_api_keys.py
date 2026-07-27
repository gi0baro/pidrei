"""Port of pi's env API key resolution (packages/ai/src/env-api-keys.ts).

Only reports actual API key variables; ambient credential sources (AWS
profiles/IAM, Google ADC) surface as the `<authenticated>` marker via
`get_env_api_key` for the providers that support them.
"""

from pathlib import Path

from tonio.colored import fs

from pidrei_ai.types import ProviderEnv
from pidrei_ai.utils.provider_env import get_provider_env_value


AMBIENT_AUTH_MARKER = "<authenticated>"

_ENV_MAP: dict[str, str] = {
    "ant-ling": "ANT_LING_API_KEY",
    "qwen-token-plan": "QWEN_TOKEN_PLAN_API_KEY",
    "qwen-token-plan-cn": "QWEN_TOKEN_PLAN_CN_API_KEY",
    "openai": "OPENAI_API_KEY",
    "azure-openai-responses": "AZURE_OPENAI_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "google": "GEMINI_API_KEY",
    "google-vertex": "GOOGLE_CLOUD_API_KEY",
    "groq": "GROQ_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "xai": "XAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "vercel-ai-gateway": "AI_GATEWAY_API_KEY",
    "zai": "ZAI_API_KEY",
    "zai-coding-cn": "ZAI_CODING_CN_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "minimax": "MINIMAX_API_KEY",
    "minimax-cn": "MINIMAX_CN_API_KEY",
    "moonshotai": "MOONSHOT_API_KEY",
    "moonshotai-cn": "MOONSHOT_API_KEY",
    "huggingface": "HF_TOKEN",
    "fireworks": "FIREWORKS_API_KEY",
    "together": "TOGETHER_API_KEY",
    "opencode": "OPENCODE_API_KEY",
    "opencode-go": "OPENCODE_API_KEY",
    "kimi-coding": "KIMI_API_KEY",
    "cloudflare-workers-ai": "CLOUDFLARE_API_KEY",
    "cloudflare-ai-gateway": "CLOUDFLARE_API_KEY",
    "xiaomi": "XIAOMI_API_KEY",
    "xiaomi-token-plan-cn": "XIAOMI_TOKEN_PLAN_CN_API_KEY",
    "xiaomi-token-plan-ams": "XIAOMI_TOKEN_PLAN_AMS_API_KEY",
    "xiaomi-token-plan-sgp": "XIAOMI_TOKEN_PLAN_SGP_API_KEY",
}

_cached_vertex_adc_credentials_exists: bool | None = None


async def _has_vertex_adc_credentials(env: ProviderEnv | None) -> bool:
    global _cached_vertex_adc_credentials_exists

    explicit_credentials_path = (env or {}).get("GOOGLE_APPLICATION_CREDENTIALS")
    if explicit_credentials_path:
        return await fs.Path(explicit_credentials_path).exists()

    if _cached_vertex_adc_credentials_exists is None:
        gac_path = get_provider_env_value("GOOGLE_APPLICATION_CREDENTIALS", env)
        if gac_path:
            _cached_vertex_adc_credentials_exists = await fs.Path(gac_path).exists()
        else:
            default_path = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
            _cached_vertex_adc_credentials_exists = await fs.Path(default_path).exists()
    return _cached_vertex_adc_credentials_exists


def _get_api_key_env_vars(provider: str) -> list[str] | None:
    if provider == "github-copilot":
        return ["COPILOT_GITHUB_TOKEN"]

    # ANTHROPIC_OAUTH_TOKEN takes precedence over ANTHROPIC_API_KEY
    if provider == "anthropic":
        return ["ANTHROPIC_OAUTH_TOKEN", "ANTHROPIC_API_KEY"]

    env_var = _ENV_MAP.get(provider)
    return [env_var] if env_var else None


def find_env_keys(provider: str, env: ProviderEnv | None = None) -> list[str] | None:
    """Find configured environment variables that can provide an API key."""
    env_vars = _get_api_key_env_vars(provider)
    if not env_vars:
        return None

    found = [env_var for env_var in env_vars if get_provider_env_value(env_var, env)]
    return found if found else None


async def get_env_api_key(provider: str, env: ProviderEnv | None = None) -> str | None:
    """Get an API key for a provider from known environment variables.

    Will not return API keys for providers that require OAuth tokens.

    Async only because of the `google-vertex` branch, which has to check for an
    ADC credentials file; `find_env_keys` stays sync since it reads env alone.
    """
    env_keys = find_env_keys(provider, env)
    if env_keys:
        return get_provider_env_value(env_keys[0], env)

    # Vertex AI supports either an explicit API key or Application Default
    # Credentials (configured via `gcloud auth application-default login`).
    if provider == "google-vertex":
        has_credentials = await _has_vertex_adc_credentials(env)
        has_project = bool(
            get_provider_env_value("GOOGLE_CLOUD_PROJECT", env) or get_provider_env_value("GCLOUD_PROJECT", env)
        )
        has_location = bool(get_provider_env_value("GOOGLE_CLOUD_LOCATION", env))
        if has_credentials and has_project and has_location:
            return AMBIENT_AUTH_MARKER

    # Bedrock supports multiple credential sources: named profiles, IAM keys,
    # bearer tokens, ECS task roles, and IRSA web identity tokens.
    if provider == "amazon-bedrock" and (
        get_provider_env_value("AWS_PROFILE", env)
        or (get_provider_env_value("AWS_ACCESS_KEY_ID", env) and get_provider_env_value("AWS_SECRET_ACCESS_KEY", env))
        or get_provider_env_value("AWS_BEARER_TOKEN_BEDROCK", env)
        or get_provider_env_value("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", env)
        or get_provider_env_value("AWS_CONTAINER_CREDENTIALS_FULL_URI", env)
        or get_provider_env_value("AWS_WEB_IDENTITY_TOKEN_FILE", env)
    ):
        return AMBIENT_AUTH_MARKER

    return None
