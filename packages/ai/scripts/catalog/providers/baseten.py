"""Baseten (pi: catalog/providers/baseten.ts)."""

import copy
from typing import Any

from catalog.define import ChatModel, ProviderContext, ProviderDefinition, models_dev_models
from catalog.reasoning import get_effort_thinking_level_map
from catalog.sources.models_dev import ModelsDevModel


_BASE_COMPAT: dict[str, Any] = {
    "supportsStore": False,
    "supportsDeveloperRole": False,
    "supportsReasoningEffort": False,
    "supportsUsageInStreaming": True,
    "maxTokensField": "max_tokens",
    "supportsStrictMode": True,
    # Baseten automatic prompt caching needs session affinity so related
    # requests land on the same replica. See:
    # https://docs.baseten.co/inference/model-apis/pricing-and-limits
    "sendSessionAffinityHeaders": True,
    "supportsLongCacheRetention": False,
}
_REASONING_EFFORT_COMPAT: dict[str, Any] = {
    **_BASE_COMPAT,
    "supportsReasoningEffort": True,
    "thinkingFormat": "openai",
}
_TOGGLE_REASONING_COMPAT: dict[str, Any] = {
    **_BASE_COMPAT,
    "thinkingFormat": "baseten",
    "chatTemplateArgs": {"enable_thinking": {"$var": "thinking.enabled"}},
}
_TOGGLE_REASONING_EFFORT_COMPAT: dict[str, Any] = {
    **_REASONING_EFFORT_COMPAT,
    "thinkingFormat": "baseten",
    "chatTemplateArgs": {"enable_thinking": {"$var": "thinking.enabled"}},
}
_TOGGLE_LEVELS: dict[str, str | None] = {
    "off": "off",
    "minimal": None,
    "low": None,
    "medium": None,
    "high": "high",
    "xhigh": None,
    "max": None,
}
_GLM_52_LEVELS: dict[str, str | None] = {
    "off": "none",
    "minimal": None,
    "low": None,
    "medium": None,
    "high": "high",
    "xhigh": None,
    "max": "max",
}
_GLM_52_MODEL_IDS = frozenset({"zai-org/GLM-5.2", "zai-org/GLM-5.2-Fast"})


def _map(model: ChatModel, source: ModelsDevModel) -> ChatModel:
    options = source.reasoning_options or []
    is_glm_52 = model["id"] in _GLM_52_MODEL_IDS
    supports_toggle = is_glm_52 or any(option.get("type") == "toggle" for option in options)
    supports_effort = is_glm_52 or any(option.get("type") == "effort" for option in options)
    compat = (
        _TOGGLE_REASONING_EFFORT_COMPAT
        if supports_toggle and supports_effort
        else _TOGGLE_REASONING_COMPAT
        if supports_toggle
        else _REASONING_EFFORT_COMPAT
        if supports_effort
        else _BASE_COMPAT
    )
    thinking_level_map = (
        dict(_GLM_52_LEVELS)
        if is_glm_52
        else dict(_TOGGLE_LEVELS)
        if supports_toggle
        else get_effort_thinking_level_map(options)
    )
    return {
        **model,
        **({"thinkingLevelMap": thinking_level_map} if thinking_level_map is not None else {}),
        # Baseten's GLM-5.2 endpoints are text-only despite models.dev reporting image input.
        "input": ["text", "image"] if not is_glm_52 and source.image_input else ["text"],
        "compat": copy.deepcopy(compat),
    }


def _models(ctx: ProviderContext) -> list[ChatModel]:
    return models_dev_models(
        ctx,
        "baseten",
        api="openai-completions",
        base_url="https://inference.baseten.co/v1",
        skip_deprecated=True,
        record_reasoning=False,
        map_model=_map,
    )


PROVIDER = ProviderDefinition(
    id="baseten",
    models=_models,
    compat={"openai-completions": {"supportsStrictMode": True}},
)
