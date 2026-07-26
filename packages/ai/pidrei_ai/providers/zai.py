"""Port of pi's zai provider factory (packages/ai/src/providers/zai.ts)."""

from pidrei_ai.api.openai_completions_lazy import openai_completions_api
from pidrei_ai.auth.helpers import env_api_key_auth
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import Provider, create_provider


def zai_provider() -> Provider:
    return create_provider(
        id="zai",
        name="Z.AI",
        base_url="https://api.z.ai/api/coding/paas/v4",
        auth=ProviderAuth(api_key=env_api_key_auth("Z.AI API key", ["ZAI_API_KEY"])),
        models=list(MODELS.get("zai", [])),
        api=openai_completions_api(),
    )
