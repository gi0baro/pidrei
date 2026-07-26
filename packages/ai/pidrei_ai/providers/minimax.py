"""Port of pi's minimax provider factory (packages/ai/src/providers/minimax.ts)."""

from pidrei_ai.api.anthropic_messages_lazy import anthropic_messages_api
from pidrei_ai.auth.helpers import env_api_key_auth
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import Provider, create_provider


def minimax_provider() -> Provider:
    return create_provider(
        id="minimax",
        name="MiniMax",
        base_url="https://api.minimax.io/anthropic",
        auth=ProviderAuth(api_key=env_api_key_auth("MiniMax API key", ["MINIMAX_API_KEY"])),
        models=list(MODELS.get("minimax", [])),
        api=anthropic_messages_api(),
    )
