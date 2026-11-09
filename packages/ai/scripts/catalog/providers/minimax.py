"""MiniMax (pi: catalog/providers/minimax.ts)."""

from catalog.define import ChatModel, ProviderContext, ProviderDefinition, models_dev_models
from catalog.reasoning import get_effort_thinking_level_map
from catalog.sources.models_dev import ModelsDevModel


# MiniMax asked to hide older models from its direct providers (#2596).
_HIDDEN_MODEL_IDS = frozenset({"MiniMax-M2", "MiniMax-M2.1", "MiniMax-M2.5", "MiniMax-M2.5-highspeed"})


def _include(model: ModelsDevModel) -> bool:
    return model.id not in _HIDDEN_MODEL_IDS


def _map_model(model: ChatModel, source: ModelsDevModel) -> ChatModel:
    # Models with effort levels (e.g. MiniMax-M3.1-Flash) take adaptive thinking with
    # output_config.effort and reject thinking.type="disabled" (no "none" effort, so off is null).
    thinking_level_map = get_effort_thinking_level_map(source.reasoning_options or [])
    return (
        {**model, "thinkingLevelMap": thinking_level_map, "compat": {"forceAdaptiveThinking": True}}
        if thinking_level_map is not None
        else model
    )


def minimax_models(ctx: ProviderContext, source: str, base_url: str) -> list[ChatModel]:
    """MiniMax's Anthropic-compatible API. The SDK appends /v1/messages."""
    return models_dev_models(
        ctx,
        source,
        api="anthropic-messages",
        base_url=base_url,
        record_reasoning=False,
        include=_include,
        map_model=_map_model,
    )


def _models(ctx: ProviderContext) -> list[ChatModel]:
    return minimax_models(ctx, "minimax", "https://api.minimax.io/anthropic")


PROVIDER = ProviderDefinition(
    id="minimax",
    models=_models,
)
