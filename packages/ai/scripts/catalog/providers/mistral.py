"""Mistral (pi: catalog/providers/mistral.ts)."""

from catalog.define import ChatModel, ProviderContext, ProviderDefinition, add_missing, models_dev_models
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


def _models(ctx: ProviderContext) -> list[ChatModel]:
    return add_missing(
        models_dev_models(
            ctx,
            "mistral",
            api="mistral-conversations",
            base_url="https://api.mistral.ai",
            record_reasoning=False,
            map_model=_map_model,
        ),
        [
            {
                "id": "mistral-medium-3.5",
                "name": "Mistral Medium 3.5",
                "api": "mistral-conversations",
                "provider": "mistral",
                "baseUrl": "https://api.mistral.ai",
                "reasoning": True,
                "thinkingLevelMap": get_effort_thinking_level_map([{"type": "effort", "values": ["none", "high"]}]),
                "input": ["text", "image"],
                "cost": {"input": 1.5, "output": 7.5, "cacheRead": 0, "cacheWrite": 0},
                "contextWindow": 262144,
                "maxTokens": 262144,
            },
        ],
    )


PROVIDER = ProviderDefinition(
    id="mistral",
    models=_models,
)
