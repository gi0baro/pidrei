"""Port of pi's moonshotai provider factory (packages/ai/src/providers/moonshotai.ts)."""

from pidrei_ai.api.openai_completions_lazy import openai_completions_api
from pidrei_ai.auth.helpers import env_api_key_auth
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import Provider, create_provider


def moonshotai_provider() -> Provider:
    return create_provider(
        id="moonshotai",
        name="Moonshot AI",
        base_url="https://api.moonshot.ai/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("Moonshot AI API key", ["MOONSHOT_API_KEY"])),
        models=list(MODELS.get("moonshotai", [])),
        api=openai_completions_api(),
    )
