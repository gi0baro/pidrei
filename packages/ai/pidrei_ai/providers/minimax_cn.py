"""Port of pi's minimax-cn provider factory (packages/ai/src/providers/minimax-cn.ts)."""

from pidrei_ai.api.anthropic_messages_lazy import anthropic_messages_api
from pidrei_ai.auth.helpers import env_api_key_auth
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import Provider, create_provider


def minimax_cn_provider() -> Provider:
    return create_provider(
        id="minimax-cn",
        name="MiniMax CN",
        base_url="https://api.minimaxi.com/anthropic",
        auth=ProviderAuth(api_key=env_api_key_auth("MiniMax CN API key", ["MINIMAX_CN_API_KEY"])),
        models=list(MODELS.get("minimax-cn", [])),
        api=anthropic_messages_api(),
    )
