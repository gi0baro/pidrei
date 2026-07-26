"""Port of pi's deepseek provider factory (packages/ai/src/providers/deepseek.ts)."""

from pidrei_ai.api.openai_completions_lazy import openai_completions_api
from pidrei_ai.auth.helpers import env_api_key_auth
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import Provider, create_provider


def deepseek_provider() -> Provider:
    return create_provider(
        id="deepseek",
        name="DeepSeek",
        base_url="https://api.deepseek.com",
        auth=ProviderAuth(api_key=env_api_key_auth("DeepSeek API key", ["DEEPSEEK_API_KEY"])),
        models=list(MODELS.get("deepseek", [])),
        api=openai_completions_api(),
    )
