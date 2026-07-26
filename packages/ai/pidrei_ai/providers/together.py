"""Port of pi's together provider factory (packages/ai/src/providers/together.ts)."""

from pidrei_ai.api.openai_completions_lazy import openai_completions_api
from pidrei_ai.auth.helpers import env_api_key_auth
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import Provider, create_provider


def together_provider() -> Provider:
    return create_provider(
        id="together",
        name="Together",
        base_url="https://api.together.ai/v1",
        auth=ProviderAuth(api_key=env_api_key_auth("Together API key", ["TOGETHER_API_KEY"])),
        models=list(MODELS.get("together", [])),
        api=openai_completions_api(),
    )
