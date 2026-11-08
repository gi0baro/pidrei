"""xAI (pi: catalog/providers/xai.ts)."""

from catalog.define import ChatModel, ProviderContext, ProviderDefinition, merge_thinking, models_dev_models


# Retired or unsuitable for PiDrei's built-in catalog.
_EXCLUDED_MODEL_IDS = frozenset(
    {
        "grok-3",
        "grok-3-fast",
        "grok-4.20-0309-non-reasoning",
        "grok-4.20-0309-reasoning",
        "grok-build-0.1",
        "grok-code-fast-1",
    }
)


def _models(ctx: ProviderContext) -> list[ChatModel]:
    return models_dev_models(
        ctx,
        "xai",
        api="openai-responses",
        base_url="https://api.x.ai/v1",
        compat={"supportsLongCacheRetention": False},
        include=lambda source: source.id not in _EXCLUDED_MODEL_IDS,
    )


def _finalize(model: ChatModel, _ctx: ProviderContext) -> None:
    # Models without verified effort options must not send the undocumented
    # "none"/"minimal" efforts.
    if model["api"] == "openai-responses" and "thinkingLevelMap" not in model:
        merge_thinking(model, {"off": None, "minimal": None})


PROVIDER = ProviderDefinition(
    id="xai",
    models=_models,
    finalize=_finalize,
)
