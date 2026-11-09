"""MiniMax China (pi: catalog/providers/minimax-cn.ts)."""

from catalog.define import ChatModel, ProviderContext, ProviderDefinition
from catalog.providers.minimax import minimax_models


def _models(ctx: ProviderContext) -> list[ChatModel]:
    return minimax_models(ctx, "minimax-cn", "https://api.minimaxi.com/anthropic")


PROVIDER = ProviderDefinition(
    id="minimax-cn",
    models=_models,
)
