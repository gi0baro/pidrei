"""Z.AI Coding Plan China (pi: catalog/providers/zai-coding-cn.ts)."""

from catalog.define import ChatModel, ProviderContext, ProviderDefinition
from catalog.providers.zai import ZAI_COMPLETIONS_COMPAT, zai_coding_plan_models


def _models(ctx: ProviderContext) -> list[ChatModel]:
    return zai_coding_plan_models(ctx, "zhipuai-coding-plan", "https://open.bigmodel.cn/api/coding/paas/v4")


PROVIDER = ProviderDefinition(
    id="zai-coding-cn",
    models=_models,
    compat={"openai-completions": ZAI_COMPLETIONS_COMPAT},
)
