"""Qwen Token Plan China (pi: catalog/providers/qwen-token-plan-cn.ts)."""

from catalog.define import ChatModel, ProviderContext, ProviderDefinition
from catalog.providers.qwen_token_plan import QWEN_COMPLETIONS_COMPAT, qwen_token_plan_models


def _models(ctx: ProviderContext) -> list[ChatModel]:
    return qwen_token_plan_models(
        ctx,
        "alibaba-token-plan-cn",
        "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    )


PROVIDER = ProviderDefinition(
    id="qwen-token-plan-cn",
    models=_models,
    compat={"openai-completions": QWEN_COMPLETIONS_COMPAT},
)
