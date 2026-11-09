"""Google Vertex AI (pi: catalog/providers/google-vertex.ts)."""

from catalog.define import ChatModel, ProviderContext, ProviderDefinition, models_dev_models
from catalog.families.google import google_thinking_levels
from catalog.providers.google import GOOGLE_ALIAS_TARGETS
from catalog.sources.models_dev import ModelsDevModel, models_dev_chat_models


def _include(source: ModelsDevModel) -> bool:
    return source.id.startswith("gemini-") and source.id != "gemini-3.1-flash-lite-preview"


def _models(ctx: ProviderContext) -> list[ChatModel]:
    sources = {model.id: model for model in models_dev_chat_models(ctx.sources.models_dev, "google-vertex")}

    def map_model(model: ChatModel, alias_source: ModelsDevModel) -> ChatModel:
        target = GOOGLE_ALIAS_TARGETS.get(model["id"])
        source = (target and sources.get(target)) or alias_source
        thinking_level_map = google_thinking_levels(model["id"], source.reasoning_options)
        return {
            **model,
            "reasoning": source.reasoning,
            **({"thinkingLevelMap": thinking_level_map} if thinking_level_map is not None else {}),
            "input": ["text", "image"] if source.image_input else ["text"],
            "cost": {
                "input": source.cost.input or 0,
                "output": source.cost.output or 0,
                # models.dev reports Vertex cache_read/cache_write values for Gemini 2.5 Flash that
                # do not match the official Gemini API standard pricing table. PiDrei only accounts
                # cachedContentTokenCount as cacheRead.
                "cacheRead": 0.03 if model["id"] == "gemini-2.5-flash" else source.cost.cache_read or 0,
                "cacheWrite": 0,
            },
            "contextWindow": source.context_window or 4096,
            "maxTokens": source.max_tokens or 4096,
        }

    return models_dev_models(
        ctx,
        "google-vertex",
        api="google-vertex",
        base_url="https://{location}-aiplatform.googleapis.com",
        record_reasoning=False,
        # The models.dev catalog also includes Claude, OpenAI, and other MaaS models
        # that do not use the @google/genai Gemini streaming path.
        include=_include,
        map_model=map_model,
    )


PROVIDER = ProviderDefinition(
    id="google-vertex",
    models=_models,
)
