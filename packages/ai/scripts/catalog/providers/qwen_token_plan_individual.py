"""Qwen Token Plan Individual (pi: catalog/providers/qwen-token-plan-individual.ts)."""

from model_data import assert_exact_model_ids

from catalog.define import ChatModel, ProviderContext, ProviderDefinition
from catalog.providers.qwen_token_plan import INTERNATIONAL_BASE_URL, QWEN_COMPLETIONS_COMPAT, qwen_token_plan_models


# QwenCloud Token Plan Individual text-model allowlist, verified 2026-11-01.
# It shares the international endpoint with a narrower catalog; qwen-token-plan-individual-cn uses it on the China endpoint.
# https://docs.qwencloud.com/token-plan/personal/token-plan-personal-overview
INDIVIDUAL_MODEL_IDS = frozenset(
    {
        "deepseek-v4-flash-0731",
        "deepseek-v4-pro",
        "deepseek-v4-pro-0813",
        "glm-5.2",
        "glm-5.3",
        "qwen3.6-flash",
        "qwen3.7-max",
        "qwen3.7-plus",
        "qwen3.8-flash",
        "qwen3.8-max",
    }
)


def _models(ctx: ProviderContext) -> list[ChatModel]:
    models = qwen_token_plan_models(ctx, "alibaba-token-plan", INTERNATIONAL_BASE_URL, INDIVIDUAL_MODEL_IDS)
    if ctx.strict:
        assert_exact_model_ids(
            ctx.provider_id,
            INDIVIDUAL_MODEL_IDS,
            [model["id"] for model in models],
        )
    return models


PROVIDER = ProviderDefinition(
    id="qwen-token-plan-individual",
    models=_models,
    compat={"openai-completions": QWEN_COMPLETIONS_COMPAT},
)
