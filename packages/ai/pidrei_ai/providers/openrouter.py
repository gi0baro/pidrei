"""Port of pi's openrouter provider factory (packages/ai/src/providers/openrouter.ts)."""

from pidrei_ai.api.openai_completions_lazy import openai_completions_api
from pidrei_ai.auth.helpers import env_api_key_auth
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import Provider, create_provider


def openrouter_provider() -> Provider:
    return create_provider(
        id="openrouter",
        name="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        auth=ProviderAuth(
            api_key=env_api_key_auth("OpenRouter API key", ["OPENROUTER_API_KEY"]),
            # OAuth login flow "OpenRouter OAuth" lands in Phase 5b.
            oauth=None,
        ),
        models=list(MODELS.get("openrouter", [])),
        api=openai_completions_api(),
    )
