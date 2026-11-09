"""Fireworks (pi: catalog/providers/fireworks.ts)."""

from typing import Any

from catalog.define import (
    ChatModel,
    ProviderContext,
    ProviderDefinition,
    merge_compat,
    merge_thinking,
    models_dev_models,
)
from catalog.reasoning import get_effort_thinking_level_map
from catalog.sources.models_dev import ModelsDevModel


# Fireworks Anthropic-compatible API. Prompt caching uses automatic prefix
# matching and session affinity: x-session-affinity routes requests to the same
# replica for cache hits. cache_control on tools and eager_input_streaming are
# not supported.
# https://docs.fireworks.ai/tools-sdks/anthropic-compatibility
_ANTHROPIC_COMPAT: dict[str, Any] = {
    "allowEmptySignature": True,
    "sendSessionAffinityHeaders": True,
    "supportsEagerToolInputStreaming": False,
    "supportsCacheControlOnTools": False,
    "supportsLongCacheRetention": False,
}
_OPENAI_COMPAT: dict[str, Any] = {
    "supportsStore": False,
    "supportsDeveloperRole": False,
    "sendSessionAffinityHeaders": True,
    "supportsLongCacheRetention": False,
}
_KIMI_K3_COMPAT: dict[str, Any] = {
    **_OPENAI_COMPAT,
    "requiresReasoningContentOnAssistantMessages": True,
    "thinkingFormat": "openai",
}

# Verified against Fireworks Messages raw_output on 2026-09-10 (#9323).
# Fall back to verified support when models.dev omits effort metadata; this is
# not an allowlist. Any Fireworks Messages model advertising effort uses adaptive thinking.
_ADAPTIVE_THINKING_FALLBACK_MODELS = frozenset(
    {
        "accounts/fireworks/models/deepseek-v4-flash-0731",
        "accounts/fireworks/models/deepseek-v4-flash-vision-exp",
        "accounts/fireworks/models/deepseek-v4-pro-0813",
        "accounts/fireworks/models/qwen3p8-max",
        "accounts/fireworks/models/qwen3p8-2p4t-a95b",
    }
)


def _map(model: ChatModel, source: ModelsDevModel) -> ChatModel:
    if "glm-" in model["id"] or "kimi-k3" in model["id"]:
        return {
            **model,
            "api": "openai-completions",
            "baseUrl": "https://api.fireworks.ai/inference/v1",
            "compat": dict(_KIMI_K3_COMPAT if "kimi-k3" in model["id"] else _OPENAI_COMPAT),
        }
    # Use adaptive thinking for cataloged effort controls, with verified
    # fallbacks where models.dev is incomplete. New models need no allowlist entry.
    adaptive = (
        any(option.get("type") == "effort" for option in source.reasoning_options or [])
        or model["id"] in _ADAPTIVE_THINKING_FALLBACK_MODELS
    )
    return {**model, "compat": {**_ANTHROPIC_COMPAT, **({"forceAdaptiveThinking": True} if adaptive else {})}}


def _models(ctx: ProviderContext) -> list[ChatModel]:
    return models_dev_models(
        ctx,
        "fireworks-ai",
        api="anthropic-messages",
        # SDK appends /v1/messages.
        base_url="https://api.fireworks.ai/inference",
        map_model=_map,
    )


def _finalize(model: ChatModel, ctx: ProviderContext) -> None:
    if model["api"] == "anthropic-messages" and (model.get("compat") or {}).get("forceAdaptiveThinking"):
        # Qwen Max currently advertises only a toggle. Prefer upstream effort
        # metadata once available instead of replacing it with this fallback.
        if model["id"] == "accounts/fireworks/models/qwen3p8-max" and model.get("thinkingLevelMap") is None:
            model["thinkingLevelMap"] = get_effort_thinking_level_map(
                [{"type": "effort", "values": ["low", "medium", "xhigh"]}]
            )
        source = ctx.models_dev_source(model)
        if (
            (source is not None and any(option.get("type") == "toggle" for option in source.reasoning_options or []))
            # The 2.4T alias omits the verified toggle in models.dev.
            or model["id"] == "accounts/fireworks/models/qwen3p8-2p4t-a95b"
        ):
            merge_thinking(model, {"off": "none"})
        if model["id"] == "accounts/fireworks/models/deepseek-v4-pro-0813":
            merge_thinking(model, {"low": "low"})
    if "glm-5p2" in model["id"]:
        # GLM 5.2 and its fast router support off/high/max. Fireworks maps low
        # and medium to high, so do not expose those aliases as distinct levels.
        merge_thinking(model, {"off": "none", "minimal": None, "low": None, "medium": None, "max": "max"})
    if "kimi-k3" in model["id"]:
        # Fireworks maps medium to high on both APIs; do not expose it as a distinct level.
        merge_thinking(model, {"medium": None})
        # Kimi K3 accepts tool-bearing system messages; Fireworks passes them through.
        if model["api"] == "openai-completions":
            merge_compat(model, {"supportsMidConvoSystemMessages": True, "supportsMidConvoToolAdditions": True})


PROVIDER = ProviderDefinition(
    id="fireworks",
    models=_models,
    compat={"openai-completions": {"supportsStrictMode": True}},
    finalize=_finalize,
)
