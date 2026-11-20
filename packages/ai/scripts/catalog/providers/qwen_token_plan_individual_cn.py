"""Qwen Token Plan Individual China (pi: catalog/providers/qwen-token-plan-individual-cn.ts)."""

from catalog.define import ChatModel, ProviderContext, ProviderDefinition
from catalog.providers.qwen_token_plan import CN_BASE_URL, QWEN_COMPLETIONS_COMPAT, qwen_token_plan_models
from catalog.providers.qwen_token_plan_individual import INDIVIDUAL_MODEL_IDS


# QwenCloud Token Plan Individual on the China endpoint: the Individual allowlist served from the China catalog.
def _models(ctx: ProviderContext) -> list[ChatModel]:
    return qwen_token_plan_models(ctx, "alibaba-token-plan-cn", CN_BASE_URL, INDIVIDUAL_MODEL_IDS)


PROVIDER = ProviderDefinition(
    id="qwen-token-plan-individual-cn",
    models=_models,
    compat={"openai-completions": QWEN_COMPLETIONS_COMPAT},
)
