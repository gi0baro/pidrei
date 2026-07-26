"""Port of pi's xiaomi provider factory (packages/ai/src/providers/xiaomi.ts)."""

from pidrei_ai.api.openai_completions_lazy import openai_completions_api
from pidrei_ai.auth.helpers import env_api_key_auth
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import Provider, create_provider


def xiaomi_provider() -> Provider:
    return create_provider(
        id="xiaomi",
        name="Xiaomi",
        base_url="https://api.xiaomimimo.com/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("Xiaomi API key", ["XIAOMI_API_KEY"])),
        models=list(MODELS.get("xiaomi", [])),
        api=openai_completions_api(),
    )
