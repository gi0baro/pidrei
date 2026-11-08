"""Z.AI (pi: catalog/providers/zai.ts)."""

from typing import Any

from catalog.define import ChatModel, ProviderContext, ProviderDefinition, models_dev_models
from catalog.reasoning import get_effort_thinking_level_map
from catalog.sources.models_dev import ModelsDevModel, models_dev_chat_models, models_dev_cost


# Tool-call streaming (`tool_stream`) is rejected by the GLM-4.5 generation.
_TOOL_STREAM_UNSUPPORTED_MODEL_IDS = frozenset({"glm-4.5", "glm-4.5-air", "glm-4.5-flash", "glm-4.5v"})


def zai_coding_plan_models(ctx: ProviderContext, source: str, base_url: str) -> list[ChatModel]:
    """The Z.AI coding plans. The international and China plans list the same models."""
    # Coding-plan entries carry plan prices; show the pay-as-you-go API rates.
    api_models = {model.id: model for model in models_dev_chat_models(ctx.sources.models_dev, "zai")}

    def map_model(model: ChatModel, source: ModelsDevModel) -> ChatModel:
        thinking_level_map = get_effort_thinking_level_map(source.reasoning_options or [])
        if thinking_level_map is not None and model["id"] in ("glm-5.2", "glm-5.2-highspeed"):
            thinking_level_map["off"] = "none"
        api_model = api_models.get(model["id"])
        return {
            **model,
            **({"thinkingLevelMap": thinking_level_map} if thinking_level_map is not None else {}),
            "cost": models_dev_cost(
                api_model.cost if api_model is not None and api_model.source.get("cost") is not None else source.cost
            ),
            "compat": {
                "supportsDeveloperRole": False,
                "thinkingFormat": "zai",
                **({"supportsReasoningEffort": True} if thinking_level_map is not None else {}),
                **({"zaiToolStream": True} if model["id"] not in _TOOL_STREAM_UNSUPPORTED_MODEL_IDS else {}),
            },
        }

    return models_dev_models(
        ctx,
        source,
        api="openai-completions",
        base_url=base_url,
        map_model=map_model,
    )


ZAI_COMPLETIONS_COMPAT: dict[str, Any] = {
    "supportsStore": False,
    "supportsDeveloperRole": False,
    "supportsReasoningEffort": False,
    "maxTokensField": "max_tokens",
    "thinkingFormat": "zai",
    "supportsStrictMode": True,
}


def _models(ctx: ProviderContext) -> list[ChatModel]:
    return zai_coding_plan_models(ctx, "zai-coding-plan", "https://api.z.ai/api/coding/paas/v4")


PROVIDER = ProviderDefinition(
    id="zai",
    models=_models,
    compat={"openai-completions": ZAI_COMPLETIONS_COMPAT},
)
