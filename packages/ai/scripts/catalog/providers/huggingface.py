"""Hugging Face (pi: catalog/providers/huggingface.ts)."""

from catalog.define import ChatModel, ProviderContext, ProviderDefinition, models_dev_models


def _models(ctx: ProviderContext) -> list[ChatModel]:
    return models_dev_models(
        ctx,
        "huggingface",
        api="openai-completions",
        base_url="https://router.huggingface.co/v1",
        compat={"supportsDeveloperRole": False},
    )


PROVIDER = ProviderDefinition(
    id="huggingface",
    models=_models,
    compat={"openai-completions": {"supportsStrictMode": True}},
)
