"""Port of pi's cerebras provider factory (packages/ai/src/providers/cerebras.ts)."""

from pidrei_ai.api.openai_completions_lazy import openai_completions_api
from pidrei_ai.auth.helpers import env_api_key_auth
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import Provider, create_provider


def cerebras_provider() -> Provider:
    return create_provider(
        id="cerebras",
        name="Cerebras",
        base_url="https://api.cerebras.ai/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("Cerebras API key", ["CEREBRAS_API_KEY"])),
        models=list(MODELS.get("cerebras", [])),
        api=openai_completions_api(),
    )
