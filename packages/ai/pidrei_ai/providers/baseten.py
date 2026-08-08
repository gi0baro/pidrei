"""Port of pi's Baseten provider factory (packages/ai/src/providers/baseten.ts)."""

from pidrei_ai.api.openai_completions_lazy import openai_completions_api
from pidrei_ai.auth.helpers import env_api_key_auth
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import Provider, create_provider


def baseten_provider() -> Provider:
    return create_provider(
        id="baseten",
        name="Baseten",
        base_url="https://inference.baseten.co/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("Baseten API key", ["BASETEN_API_KEY"])),
        models=list(MODELS.get("baseten", [])),
        api=openai_completions_api(),
    )
