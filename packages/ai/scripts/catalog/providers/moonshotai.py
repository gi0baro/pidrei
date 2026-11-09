"""Moonshot AI (pi: catalog/providers/moonshotai.ts)."""

from catalog.define import (
    ChatModel,
    ProviderContext,
    ProviderDefinition,
    merge_compat,
    merge_thinking,
    models_dev_models,
)
from catalog.families.kimi import KIMI_K3_COST, KIMI_K3_OPENAI_COMPAT, KIMI_OPENAI_COMPAT, is_kimi_k3_family_id
from catalog.sources.models_dev import ModelsDevModel


_ALWAYS_THINKING_MODEL_IDS = frozenset({"kimi-k2.7-code", "kimi-k2.7-code-highspeed"})
# Kimi K2.6/K2.7 accept system text after the conversation starts but reject
# tool-bearing system messages.
_TEXT_ONLY_MID_CONVO_MODEL_IDS = frozenset({"kimi-k2.6", *_ALWAYS_THINKING_MODEL_IDS})


def _map_model(model: ChatModel, source: ModelsDevModel) -> ChatModel:
    is_kimi_k3 = is_kimi_k3_family_id(model["id"])
    return {
        **model,
        "reasoning": is_kimi_k3 or source.reasoning,
        # Moonshot does not bill cache writes; models.dev lists the input rate as cache_write for the K3 family.
        "cost": (
            {**KIMI_K3_COST}
            if model["id"] == "kimi-k3"
            else {
                "input": source.cost.input or 0,
                "output": source.cost.output or 0,
                "cacheRead": source.cost.cache_read or 0,
                "cacheWrite": 0 if is_kimi_k3 else source.cost.cache_write or 0,
            }
        ),
        "compat": {**(KIMI_K3_OPENAI_COMPAT if is_kimi_k3 else KIMI_OPENAI_COMPAT)},
    }


def moonshot_models(ctx: ProviderContext, source: str, base_url: str) -> list[ChatModel]:
    return models_dev_models(
        ctx,
        source,
        api="openai-completions",
        base_url=base_url,
        map_model=_map_model,
    )


MOONSHOT_COMPLETIONS_COMPAT = {
    "supportsStore": False,
    "supportsDeveloperRole": False,
    "supportsReasoningEffort": False,
    "maxTokensField": "max_tokens",
}


def finalize_moonshot_model(model: ChatModel, _ctx: ProviderContext) -> None:
    if model["id"] in _ALWAYS_THINKING_MODEL_IDS:
        # Kimi K2.7 Code is always-thinking. Official docs say
        # `thinking: { type: "disabled" }` is rejected, and callers can omit
        # the thinking parameter to use the enabled default.
        merge_thinking(model, {"off": None})
    is_kimi_k3 = is_kimi_k3_family_id(model["id"])
    if is_kimi_k3 or model["id"] in _TEXT_ONLY_MID_CONVO_MODEL_IDS:
        merge_compat(
            model,
            {
                "supportsMidConvoSystemMessages": True,
                # The Kimi K3 family also accepts tool-bearing system messages.
                **({"supportsMidConvoToolAdditions": True} if is_kimi_k3 else {}),
            },
        )


def _models(ctx: ProviderContext) -> list[ChatModel]:
    return moonshot_models(ctx, "moonshotai", "https://api.moonshot.ai/v1")


PROVIDER = ProviderDefinition(
    id="moonshotai",
    models=_models,
    compat={"openai-completions": MOONSHOT_COMPLETIONS_COMPAT},
    finalize=finalize_moonshot_model,
)
