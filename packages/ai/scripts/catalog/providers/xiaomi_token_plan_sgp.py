"""Xiaomi MiMo Token Plan Singapore (pi: catalog/providers/xiaomi-token-plan-sgp.ts)."""

from catalog.define import ChatModel, ProviderContext, ProviderDefinition
from catalog.providers.xiaomi import XIAOMI_COMPLETIONS_COMPAT, xiaomi_models


def _models(ctx: ProviderContext) -> list[ChatModel]:
    return xiaomi_models(ctx, "https://token-plan-sgp.xiaomimimo.com/v1")


PROVIDER = ProviderDefinition(
    id="xiaomi-token-plan-sgp",
    models=_models,
    compat={"openai-completions": XIAOMI_COMPLETIONS_COMPAT},
)
