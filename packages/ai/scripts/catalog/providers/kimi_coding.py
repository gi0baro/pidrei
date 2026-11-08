"""Kimi Coding (pi: catalog/providers/kimi-coding.ts)."""

from typing import Any

from catalog.define import ChatModel, ProviderContext, ProviderDefinition, models_dev_models
from catalog.families.kimi import KIMI_K3_COST, KIMI_K3_OPENAI_COMPAT, KIMI_OPENAI_COMPAT
from catalog.reasoning import get_effort_thinking_level_map
from catalog.sources.models_dev import ModelsDevModel


# Kimi Coding is subscription-backed, so models.dev reports zero cost. Use the
# equivalent Moonshot API rates to estimate the value of subscription usage.
_IMPLIED_COSTS: dict[str, dict[str, Any]] = {
    "k3": KIMI_K3_COST,
    "k3-256k": KIMI_K3_COST,
    "k3.1-pro": {"input": 2, "output": 10, "cacheRead": 0.2, "cacheWrite": 0},
    "k3.1-pro-highspeed": {"input": 4, "output": 20, "cacheRead": 0.2, "cacheWrite": 0},
    "kimi-for-coding": {"input": 0.95, "output": 4, "cacheRead": 0.19, "cacheWrite": 0},
    "kimi-for-coding-highspeed": {"input": 1.9, "output": 8, "cacheRead": 0.38, "cacheWrite": 0},
}


def _map_model(model: ChatModel, source: ModelsDevModel) -> ChatModel:
    # K3-style models report effort levels and dynamic tool loading (tools declared in
    # later system messages) in https://api.kimi.com/coding/v1/models; models.dev mirrors
    # the effort levels. kimi-for-coding-highspeed reports neither and rejects
    # tool-bearing system messages with a 400.
    is_k3_style = any(option.get("type") == "effort" for option in source.reasoning_options or [])
    implied_cost = _IMPLIED_COSTS.get(model["id"]) or {}
    # K3-style models send reasoning_effort, so expose the effort values models.dev lists.
    effort_levels = (
        get_effort_thinking_level_map(source.reasoning_options)
        if is_k3_style and source.reasoning_options is not None
        else None
    )
    return {
        **model,
        "compat": {
            **(KIMI_K3_OPENAI_COMPAT if is_k3_style else KIMI_OPENAI_COMPAT),
            # All models accept system text after the conversation starts.
            "supportsMidConvoSystemMessages": True,
            **({"supportsMidConvoToolAdditions": True} if is_k3_style else {}),
        },
        "cost": {
            "input": source.cost.input or implied_cost.get("input") or 0,
            "output": source.cost.output or implied_cost.get("output") or 0,
            "cacheRead": source.cost.cache_read or implied_cost.get("cacheRead") or 0,
            "cacheWrite": source.cost.cache_write or implied_cost.get("cacheWrite") or 0,
        },
        # Kimi Code reports every model as always-thinking.
        "thinkingLevelMap": {**(effort_levels or {}), "off": None},
    }


def _models(ctx: ProviderContext) -> list[ChatModel]:
    return models_dev_models(
        ctx,
        "kimi-code-plan-global",
        api="openai-completions",
        base_url="https://api.kimi.com/coding/v1",
        record_reasoning=False,
        map_model=_map_model,
    )


# Kimi Coding uses Chat Completions, the same protocol as the official Kimi Code client.
PROVIDER = ProviderDefinition(
    id="kimi-coding",
    models=_models,
    compat={"openai-completions": {"supportsStrictMode": True}},
)
