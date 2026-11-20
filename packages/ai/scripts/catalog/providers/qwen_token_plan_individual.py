"""Qwen Token Plan Individual (pi: catalog/providers/qwen-token-plan-individual.ts)."""

from catalog.define import ChatModel, ProviderContext, ProviderDefinition
from catalog.providers.qwen_token_plan import INTERNATIONAL_BASE_URL, QWEN_COMPLETIONS_COMPAT, qwen_token_plan_models


# Token Plan Individual (Personal) text-model allowlist, verified 2026-11-09 against both regions' docs.
# It shares the international endpoint with a narrower catalog; qwen-token-plan-individual-cn uses it on the China endpoint.
# The dynamic "auto" router is not listed: it has no fixed model metadata.
# https://docs.qwencloud.com/token-plan/personal/token-plan-personal-overview
# https://platform.qianwenai.com/docs/token-plan/personal/token-plan-personal-overview
INDIVIDUAL_MODEL_IDS = frozenset(
    {
        "deepseek-v4-flash-0731",
        "deepseek-v4-pro",
        "deepseek-v4-pro-0813",
        "deepseek-v4.1-flash",
        "deepseek-v4.1-flash-1017",
        "glm-5.2",
        "glm-5.3",
        "qwen3.6-flash",
        "qwen3.7-max",
        "qwen3.7-plus",
        "qwen3.8-flash",
        "qwen3.8-max",
        "qwen4-max-preview",
    }
)


def _models(ctx: ProviderContext) -> list[ChatModel]:
    return qwen_token_plan_models(ctx, "alibaba-token-plan", INTERNATIONAL_BASE_URL, INDIVIDUAL_MODEL_IDS)


PROVIDER = ProviderDefinition(
    id="qwen-token-plan-individual",
    models=_models,
    compat={"openai-completions": QWEN_COMPLETIONS_COMPAT},
)
