"""Google (pi: catalog/providers/google.ts)."""

from catalog.define import ChatModel, ProviderContext, ProviderDefinition, models_dev_models, with_input_limits
from catalog.families.google import google_thinking_levels
from catalog.sources.models_dev import ModelsDevModel, models_dev_chat_models, models_dev_cost


# Rolling aliases take their metadata from the model they currently point to.
GOOGLE_ALIAS_TARGETS: dict[str, str] = {
    "gemini-flash-latest": "gemini-3.5-flash",
    "gemini-flash-lite-latest": "gemini-3.1-flash-lite",
}


def _models(ctx: ProviderContext) -> list[ChatModel]:
    sources = {model.id: model for model in models_dev_chat_models(ctx.sources.models_dev, "google")}

    def map_model(model: ChatModel, alias_source: ModelsDevModel) -> ChatModel:
        target = GOOGLE_ALIAS_TARGETS.get(model["id"])
        source = (target and sources.get(target)) or alias_source
        thinking_level_map = google_thinking_levels(model["id"], source.reasoning_options)
        return {
            **model,
            "reasoning": source.reasoning,
            **({"thinkingLevelMap": thinking_level_map} if thinking_level_map is not None else {}),
            "input": ["text", "image"] if source.image_input else ["text"],
            "cost": models_dev_cost(source.cost),
            "contextWindow": source.context_window or 4096,
            "maxTokens": source.max_tokens or 4096,
        }

    return models_dev_models(
        ctx,
        "google",
        api="google-generative-ai",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        record_reasoning=False,
        map_model=map_model,
    )


def _finalize(model: ChatModel, _ctx: ProviderContext) -> None:
    with_input_limits(model, {"maxRequestBytes": 20 * 1024 * 1024, "images": {"maxPerRequest": 3600}})


PROVIDER = ProviderDefinition(
    id="google",
    models=_models,
    finalize=_finalize,
)
