"""Vercel AI Gateway (pi: catalog/providers/vercel-ai-gateway.ts)."""

import copy
from typing import Any

from catalog.define import ChatModel, ProviderContext, ProviderDefinition
from catalog.families.kimi import KIMI_K3_MAX_TOKENS, is_kimi_k3_family_id


def _models(ctx: ProviderContext) -> list[ChatModel]:
    return [
        # Keep Kimi K3's canonical output limit when gateway metadata is missing or incorrect.
        {**model, "maxTokens": KIMI_K3_MAX_TOKENS}
        if model["id"].startswith("moonshotai/") and is_kimi_k3_family_id(model["id"][len("moonshotai/") :])
        else model
        for model in copy.deepcopy(ctx.sources.ai_gateway["chat"])
    ]


def _classifiers(ctx: ProviderContext) -> list[dict[str, Any]]:
    return copy.deepcopy(ctx.sources.ai_gateway["classifiers"])


PROVIDER = ProviderDefinition(
    id="vercel-ai-gateway",
    models=_models,
    classifiers=_classifiers,
)
