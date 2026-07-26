"""Port of pi's fireworks provider factory (packages/ai/src/providers/fireworks.ts).

`api` dispatches on `model.api`: anthropic-messages, openai-completions.
"""

from pidrei_ai.api.anthropic_messages_lazy import anthropic_messages_api
from pidrei_ai.api.openai_completions_lazy import openai_completions_api
from pidrei_ai.auth.helpers import env_api_key_auth
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import Provider, create_provider


def fireworks_provider() -> Provider:
    return create_provider(
        id="fireworks",
        name="Fireworks",
        base_url="https://api.fireworks.ai/inference",
        auth=ProviderAuth(api_key=env_api_key_auth("Fireworks API key", ["FIREWORKS_API_KEY"])),
        models=list(MODELS.get("fireworks", [])),
        api={
            "anthropic-messages": anthropic_messages_api(),
            "openai-completions": openai_completions_api(),
        },
    )
