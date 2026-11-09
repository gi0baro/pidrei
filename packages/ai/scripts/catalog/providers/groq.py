"""Groq (pi: catalog/providers/groq.ts)."""

from catalog.define import ChatModel, ProviderContext, ProviderDefinition, merge_thinking, models_dev_models


def _models(ctx: ProviderContext) -> list[ChatModel]:
    return models_dev_models(ctx, "groq", api="openai-completions", base_url="https://api.groq.com/openai/v1")


def _finalize(model: ChatModel, _ctx: ProviderContext) -> None:
    if model["id"] == "qwen/qwen3.6-27b":
        merge_thinking(model, {"minimal": None, "low": None, "medium": None, "high": "default"})


PROVIDER = ProviderDefinition(
    id="groq",
    models=_models,
    compat={"openai-completions": {"supportsStrictMode": True}},
    finalize=_finalize,
)
