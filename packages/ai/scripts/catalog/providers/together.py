"""Together AI (pi: catalog/providers/together.ts)."""

from typing import Any

from catalog.define import ChatModel, ProviderContext, ProviderDefinition, models_dev_models
from catalog.sources.models_dev import ModelsDevModel


_BASE_COMPAT: dict[str, Any] = {
    "supportsStore": False,
    "supportsDeveloperRole": False,
    "supportsReasoningEffort": False,
    "maxTokensField": "max_tokens",
    "supportsStrictMode": False,
    "supportsLongCacheRetention": False,
}
_TOGGLE_REASONING_COMPAT: dict[str, Any] = {**_BASE_COMPAT, "thinkingFormat": "together"}
_REASONING_EFFORT_COMPAT: dict[str, Any] = {
    **_BASE_COMPAT,
    "supportsReasoningEffort": True,
    "thinkingFormat": "openai",
}
_TOGGLE_REASONING_EFFORT_COMPAT: dict[str, Any] = {
    **_TOGGLE_REASONING_COMPAT,
    "supportsReasoningEffort": True,
}

_REASONING_EFFORT_MODELS = frozenset({"openai/gpt-oss-20b", "openai/gpt-oss-120b"})
_TOGGLE_REASONING_EFFORT_MODELS = frozenset({"deepseek-ai/DeepSeek-V4-Pro-0813"})

_REASONING_EFFORT_LEVELS: dict[str, str | None] = {"off": None, "minimal": None}
_DEEPSEEK_V4_LEVELS: dict[str, str | None] = {
    "minimal": None,
    "low": None,
    "medium": None,
    "high": "high",
    "xhigh": None,
}
_TOGGLE_REASONING_LEVELS: dict[str, str | None] = {"minimal": None, "low": None, "medium": None}


def _map_model(model: ChatModel, _source: ModelsDevModel) -> ChatModel:
    if not model["reasoning"]:
        return {**model, "compat": dict(_BASE_COMPAT)}
    if model["id"] in _REASONING_EFFORT_MODELS:
        return {**model, "compat": dict(_REASONING_EFFORT_COMPAT), "thinkingLevelMap": dict(_REASONING_EFFORT_LEVELS)}
    if model["id"] in _TOGGLE_REASONING_EFFORT_MODELS:
        return {**model, "compat": dict(_TOGGLE_REASONING_EFFORT_COMPAT), "thinkingLevelMap": dict(_DEEPSEEK_V4_LEVELS)}
    return {**model, "compat": dict(_TOGGLE_REASONING_COMPAT), "thinkingLevelMap": dict(_TOGGLE_REASONING_LEVELS)}


def _models(ctx: ProviderContext) -> list[ChatModel]:
    return models_dev_models(
        ctx,
        "togetherai",
        api="openai-completions",
        base_url="https://api.together.ai/v1",
        skip_deprecated=True,
        map_model=_map_model,
    )


PROVIDER = ProviderDefinition(
    id="together",
    models=_models,
    compat={
        "openai-completions": {
            "supportsStore": False,
            "supportsDeveloperRole": False,
            "supportsReasoningEffort": False,
            "maxTokensField": "max_tokens",
            "thinkingFormat": "together",
            "supportsLongCacheRetention": False,
        },
    },
)
