"""Xiaomi MiMo (pi: catalog/providers/xiaomi.ts)."""

from typing import Any

from catalog.define import ChatModel, ProviderContext, ProviderDefinition, models_dev_models
from catalog.sources.models_dev import ModelsDevModel


def _map_model(model: ChatModel, _source: ModelsDevModel) -> ChatModel:
    if not model["reasoning"]:
        return model
    return {
        **model,
        "thinkingLevelMap": {
            "off": "off",
            "minimal": None,
            "low": None,
            "medium": None,
            "high": "high",
            "xhigh": None,
            "max": None,
        },
    }


def xiaomi_models(ctx: ProviderContext, base_url: str) -> list[ChatModel]:
    """Xiaomi MiMo. The `xiaomi` provider targets the API billing endpoint; the
    `xiaomi-token-plan-*` providers cover prepaid Token Plan endpoints per region.
    MiMo Chat Completions only documents thinking.type enabled/disabled and
    rejects some reasoning_effort values (e.g. "minimal"). Xiaomi documents all
    effort levels as identical, so expose only off and high."""
    return models_dev_models(
        ctx,
        ctx.provider_id,
        api="openai-completions",
        base_url=base_url,
        skip_deprecated=True,
        record_reasoning=False,
        compat={
            "supportsReasoningEffort": False,
            "requiresReasoningContentOnAssistantMessages": True,
            "thinkingFormat": "deepseek",
        },
        map_model=_map_model,
    )


XIAOMI_COMPLETIONS_COMPAT: dict[str, Any] = {"supportsStrictMode": True}


def _models(ctx: ProviderContext) -> list[ChatModel]:
    return xiaomi_models(ctx, "https://api.xiaomimimo.com/v1")


PROVIDER = ProviderDefinition(
    id="xiaomi",
    models=_models,
    compat={"openai-completions": XIAOMI_COMPLETIONS_COMPAT},
)
