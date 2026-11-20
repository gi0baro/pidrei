"""Qwen Token Plan (pi: catalog/providers/qwen-token-plan.ts)."""

from typing import Any

from catalog.define import ChatModel, ProviderContext, ProviderDefinition, models_dev_models
from catalog.reasoning import get_effort_thinking_level_map
from catalog.sources.models_dev import ModelsDevModel


# Qwen Token Plan: QwenCloud (international) and Qianwen AI Platform (China). Each region has its
# own endpoint and API keys (sk-sp- prefix). models.dev lists them as "alibaba-token-plan[-cn]".
# The former token-plan.{ap-southeast-1,cn-beijing}.maas.aliyuncs.com hosts resolve to the same
# gateways and keep working for configurations that still name them.
# https://docs.qwencloud.com/developer-guides/getting-started/regions
# https://platform.qianwenai.com/docs/developer-guides/clients-and-developer-tools/other-tools
INTERNATIONAL_BASE_URL = "https://token-plan.maas.qwencloudapi.com/compatible-mode/v1"
CN_BASE_URL = "https://token-plan.maas.qianwenaiapi.com/compatible-mode/v1"

_COMPAT: dict[str, Any] = {
    "thinkingFormat": "qwen",
    "supportsStore": False,
    "supportsReasoningEffort": True,
}
_FALLBACK_THINKING_LEVELS: dict[str, str | None] = {
    "minimal": None,
    "low": None,
    "medium": None,
    "high": "high",
    "xhigh": None,
    "max": "max",
}
_REASONING_EFFORT_FALLBACK_MODEL_IDS = frozenset({"glm-5", "glm-5.1"})


def qwen_token_plan_models(
    ctx: ProviderContext, source: str, base_url: str, model_ids: frozenset[str] | None = None
) -> list[ChatModel]:
    def include(model: ModelsDevModel) -> bool:
        return model_ids is None or model.id in model_ids

    def map_model(model: ChatModel, source: ModelsDevModel) -> ChatModel:
        thinking_level_map = get_effort_thinking_level_map(source.reasoning_options or [])
        if thinking_level_map is None and model["id"] in _REASONING_EFFORT_FALLBACK_MODEL_IDS:
            thinking_level_map = dict(_FALLBACK_THINKING_LEVELS)
        return {
            **model,
            "compat": (
                dict(_COMPAT) if thinking_level_map is not None else {**_COMPAT, "supportsReasoningEffort": False}
            ),
            **({"thinkingLevelMap": thinking_level_map} if thinking_level_map is not None else {}),
        }

    return models_dev_models(
        ctx,
        source,
        api="openai-completions",
        base_url=base_url,
        record_reasoning=False,
        # models.dev marks models dropped from the Token Plan lists deprecated, e.g. qwen3.8-max-preview,
        # deepseek-v3.2 and deepseek-v4-flash.
        skip_deprecated=True,
        include=include,
        map_model=map_model,
    )


QWEN_COMPLETIONS_COMPAT: dict[str, Any] = {"supportsStrictMode": True}


def _models(ctx: ProviderContext) -> list[ChatModel]:
    return qwen_token_plan_models(ctx, "alibaba-token-plan", INTERNATIONAL_BASE_URL)


PROVIDER = ProviderDefinition(
    id="qwen-token-plan",
    models=_models,
    compat={"openai-completions": QWEN_COMPLETIONS_COMPAT},
)
