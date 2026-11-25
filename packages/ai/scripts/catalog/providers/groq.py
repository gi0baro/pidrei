"""Groq (pi: catalog/providers/groq.ts)."""

from catalog.define import ChatModel, ProviderContext, ProviderDefinition, models_dev_models


def _models(ctx: ProviderContext) -> list[ChatModel]:
    return models_dev_models(
        ctx,
        "groq",
        api="openai-completions",
        base_url="https://api.groq.com/openai/v1",
        # models.dev has marked Groq models deprecated only after Groq shut them down, then deletes
        # them weeks later, e.g. llama-3.3-70b-versatile: shut down 2026-08-16, marked 2026-10-21.
        skip_deprecated=True,
    )


PROVIDER = ProviderDefinition(
    id="groq",
    models=_models,
    compat={"openai-completions": {"supportsStrictMode": True}},
)
