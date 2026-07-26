"""Port of pi's huggingface provider factory (packages/ai/src/providers/huggingface.ts)."""

from pidrei_ai.api.openai_completions_lazy import openai_completions_api
from pidrei_ai.auth.helpers import env_api_key_auth
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import Provider, create_provider


def huggingface_provider() -> Provider:
    return create_provider(
        id="huggingface",
        name="Hugging Face",
        base_url="https://router.huggingface.co/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("Hugging Face token", ["HF_TOKEN"])),
        models=list(MODELS.get("huggingface", [])),
        api=openai_completions_api(),
    )
