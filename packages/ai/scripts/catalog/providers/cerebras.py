"""Cerebras (pi: catalog/providers/cerebras.ts)."""

from catalog.define import ChatModel, ProviderContext, ProviderDefinition, models_dev_models


def _models(ctx: ProviderContext) -> list[ChatModel]:
    return models_dev_models(ctx, "cerebras", api="openai-completions", base_url="https://api.cerebras.ai/v1")


PROVIDER = ProviderDefinition(
    id="cerebras",
    models=_models,
    compat={"openai-completions": {"supportsStore": False, "supportsDeveloperRole": False}},
)
