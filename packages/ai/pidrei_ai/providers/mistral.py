"""Port of pi's mistral provider factory (packages/ai/src/providers/mistral.ts)."""

from pidrei_ai.api.mistral_conversations_lazy import mistral_conversations_api
from pidrei_ai.auth.helpers import env_api_key_auth
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import Provider, create_provider


def mistral_provider() -> Provider:
    return create_provider(
        id="mistral",
        name="Mistral",
        base_url="https://api.mistral.ai",
        auth=ProviderAuth(api_key=env_api_key_auth("Mistral API key", ["MISTRAL_API_KEY"])),
        models=list(MODELS.get("mistral", [])),
        api=mistral_conversations_api(),
    )
