"""Xiaomi MiMo Token Plan China (pi: catalog/providers/xiaomi-token-plan-cn.ts)."""

from catalog.define import ChatModel, ProviderContext, ProviderDefinition
from catalog.providers.xiaomi import XIAOMI_COMPLETIONS_COMPAT, xiaomi_models


def _models(ctx: ProviderContext) -> list[ChatModel]:
    return xiaomi_models(ctx, "https://token-plan-cn.xiaomimimo.com/v1")


PROVIDER = ProviderDefinition(
    id="xiaomi-token-plan-cn",
    models=_models,
    compat={"openai-completions": XIAOMI_COMPLETIONS_COMPAT},
)
