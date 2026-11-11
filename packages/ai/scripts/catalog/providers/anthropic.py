"""Anthropic (pi: catalog/providers/anthropic.ts)."""

import copy

from catalog.define import (
    ChatModel,
    ProviderContext,
    ProviderDefinition,
    add_missing,
    merge_compat,
    merge_thinking,
    models_dev_models,
    with_input_limits,
)
from catalog.families.anthropic import (
    claude_55_thinking_levels,
    enable_claude_mid_convo_effort,
    supports_claude_mid_convo_effort,
    supports_claude_mid_convo_system_messages,
)
from catalog.sources.models_dev import ModelsDevModel


# Models the API may fall back to, with the price the caller is billed.
_ALLOWED_FALLBACK_MODELS: dict[str, list[str]] = {
    "claude-fable-5": ["claude-opus-4-8", "claude-opus-5"],
    "claude-fable-5-2": ["claude-opus-4-8", "claude-opus-5"],
    "claude-opus-5": ["claude-opus-4-8"],
}

_MODELS_WITH_1M_CONTEXT = frozenset({"claude-opus-4-6", "claude-sonnet-4-6"})
_MODELS_55 = frozenset({"claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"})

# Released 2026-11-09; models.dev does not list it for Anthropic yet.
# https://platform.claude.com/docs/en/models/fable-5-2/whats-new-fable-5-2
_CLAUDE_FABLE_5_2: ChatModel = {
    "id": "claude-fable-5-2",
    "name": "Claude Fable 5.2",
    "api": "anthropic-messages",
    "baseUrl": "https://api.anthropic.com",
    "provider": "anthropic",
    "reasoning": True,
    "input": ["text", "image"],
    "cost": {"input": 8, "output": 40, "cacheRead": 0.2, "cacheWrite": 10},
    "contextWindow": 1000000,
    "maxTokens": 128000,
}


def _map(model: ChatModel, _source: ModelsDevModel) -> ChatModel:
    if model["id"] in _MODELS_55:
        merge_thinking(model, claude_55_thinking_levels(model["id"]))
    if model["id"] in _MODELS_WITH_1M_CONTEXT:
        model["contextWindow"] = 1000000
    return model


def _models(ctx: ProviderContext) -> list[ChatModel]:
    return add_missing(
        models_dev_models(
            ctx, "anthropic", api="anthropic-messages", base_url="https://api.anthropic.com", map_model=_map
        ),
        [copy.deepcopy(_CLAUDE_FABLE_5_2)],
    )


def _finalize(model: ChatModel, _ctx: ProviderContext) -> None:
    if model["api"] != "anthropic-messages":
        return
    if supports_claude_mid_convo_effort(model["id"]):
        enable_claude_mid_convo_effort(model)
    if supports_claude_mid_convo_system_messages(model["id"]):
        merge_compat(model, {"supportsMidConvoSystemMessages": True, "supportsMidConvoToolChanges": True})
    merge_compat(model, {"supportsStrictTools": True})
    # Ephemeral entries have a hard five-minute lifetime; `ttl: "1h"` extends
    # it to one hour. Only direct Anthropic is annotated so cache warming does
    # not assume equivalent behavior through proxies.
    # https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
    model["promptCache"] = {"short": 300, "long": 3600}
    with_input_limits(
        model,
        {
            "maxRequestBytes": 32 * 1024 * 1024,
            "images": {"maxPerRequest": 100 if model["contextWindow"] == 200000 else 600},
        },
    )


def _finalize_all(models: list[ChatModel], _ctx: ProviderContext) -> None:
    models_by_id = {model["id"]: model for model in models}
    for model_id, fallback_model_ids in _ALLOWED_FALLBACK_MODELS.items():
        model = models_by_id.get(model_id)
        if model is None:
            continue
        compatible_fallback_model_ids = (
            [fallback_id for fallback_id in fallback_model_ids if supports_claude_mid_convo_effort(fallback_id)]
            if (model.get("compat") or {}).get("supportsMidConvoEffort")
            else fallback_model_ids
        )
        allowed_fallback_models = [
            {"provider": fallback_model["provider"], "model": fallback_model["id"], "cost": fallback_model["cost"]}
            for fallback_model_id in compatible_fallback_model_ids
            if (fallback_model := models_by_id.get(fallback_model_id)) is not None
        ]
        if allowed_fallback_models:
            merge_compat(model, {"allowedFallbackModels": allowed_fallback_models})


PROVIDER = ProviderDefinition(id="anthropic", models=_models, finalize=_finalize, finalize_all=_finalize_all)
