"""Mistral (pi: catalog/providers/mistral.ts)."""

from catalog.define import ChatModel, ProviderContext, ProviderDefinition, models_dev_models
from catalog.reasoning import get_effort_thinking_level_map
from catalog.sources.models_dev import ModelsDevModel, round_cost


def _map_model(model: ChatModel, source: ModelsDevModel) -> ChatModel:
    # Models with effort values use `reasoning_effort` with these levels.
    # Reasoning models without them (Magistral) use `prompt_mode`.
    thinking_level_map = get_effort_thinking_level_map(source.reasoning_options or [])
    return {
        **model,
        **({"thinkingLevelMap": thinking_level_map} if thinking_level_map is not None else {}),
        "cost": {
            "input": source.cost.input or 0,
            "output": source.cost.output or 0,
            # Mistral bills cached input at 10% when models.dev does not say otherwise.
            "cacheRead": (
                source.cost.cache_read
                if source.cost.cache_read is not None
                else (round_cost(source.cost.input * 0.1) if source.cost.input else 0)
            ),
            "cacheWrite": source.cost.cache_write or 0,
        },
    }


# No skip_deprecated: models.dev marks Mistral models deprecated as soon as Mistral announces their
# retirement, while they keep working, e.g. devstral-2512: marked 2026-06-02, retired 2026-07-31.
def _models(ctx: ProviderContext) -> list[ChatModel]:
    return models_dev_models(
        ctx,
        "mistral",
        api="mistral-conversations",
        base_url="https://api.mistral.ai",
        record_reasoning=False,
        map_model=_map_model,
    )


PROVIDER = ProviderDefinition(
    id="mistral",
    models=_models,
)
