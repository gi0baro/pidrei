"""Port of pi's nvidia provider factory (packages/ai/src/providers/nvidia.ts)."""

from pidrei_ai.api.openai_completions_lazy import openai_completions_api
from pidrei_ai.auth.helpers import env_api_key_auth
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import Provider, create_provider


def nvidia_provider() -> Provider:
    return create_provider(
        id="nvidia",
        name="NVIDIA",
        base_url="https://integrate.api.nvidia.com/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("NVIDIA API key", ["NVIDIA_API_KEY"])),
        models=list(MODELS.get("nvidia", [])),
        api=openai_completions_api(),
    )
