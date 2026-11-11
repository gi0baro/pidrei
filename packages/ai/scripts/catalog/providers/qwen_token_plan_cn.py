"""Qwen Token Plan China (pi: catalog/providers/qwen-token-plan-cn.ts)."""

from catalog.define import ChatModel, ProviderContext, ProviderDefinition
from catalog.providers.qwen_token_plan import CN_BASE_URL, QWEN_COMPLETIONS_COMPAT, qwen_token_plan_models


def _models(ctx: ProviderContext) -> list[ChatModel]:
    return qwen_token_plan_models(ctx, "alibaba-token-plan-cn", CN_BASE_URL)


PROVIDER = ProviderDefinition(
    id="qwen-token-plan-cn",
    models=_models,
    compat={"openai-completions": QWEN_COMPLETIONS_COMPAT},
)
