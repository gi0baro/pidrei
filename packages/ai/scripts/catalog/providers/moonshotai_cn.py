"""Moonshot AI China (pi: catalog/providers/moonshotai-cn.ts)."""

from catalog.define import ChatModel, ProviderContext, ProviderDefinition
from catalog.providers.moonshotai import MOONSHOT_COMPLETIONS_COMPAT, finalize_moonshot_model, moonshot_models


def _models(ctx: ProviderContext) -> list[ChatModel]:
    return moonshot_models(ctx, "moonshotai-cn", "https://api.moonshot.cn/v1")


PROVIDER = ProviderDefinition(
    id="moonshotai-cn",
    models=_models,
    compat={"openai-completions": MOONSHOT_COMPLETIONS_COMPAT},
    finalize=finalize_moonshot_model,
)
