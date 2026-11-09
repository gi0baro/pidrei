"""Meta (pi: catalog/providers/meta.ts)."""

from catalog.define import ChatModel, ProviderContext, ProviderDefinition, models_dev_models


def _models(ctx: ProviderContext) -> list[ChatModel]:
    return models_dev_models(ctx, "meta", api="openai-responses", base_url="https://api.meta.ai/v1")


PROVIDER = ProviderDefinition(
    id="meta",
    models=_models,
)
