"""Azure (pi: catalog/providers/azure.ts)."""

from typing import Any

from catalog.define import ChatModel, ProviderContext, ProviderDefinition, merge_compat
from catalog.families.openai import enable_openai_grammar_tools


# Azure Foundry deploys these with larger context windows than OpenAI's own short-tier defaults.
# See models-sold-directly-by-azure docs.
_CONTEXT_WINDOW_OVERRIDES: dict[str, int] = {
    "gpt-5.4": 1050000,
    "gpt-5.5": 1050000,
    "gpt-5.6-luna": 1050000,
    "gpt-5.6-sol": 1050000,
    "gpt-5.6-terra": 1050000,
    "gpt-6-astra": 1050000,
    "gpt-6-luna": 1050000,
    "gpt-6-sol": 1050000,
    "gpt-6.1-sol": 1050000,
    "gpt-6.1-terra": 1050000,
}

# Azure resells DeepSeek at its own rates. US data zone, checked 2026-09-16.
# https://azure.microsoft.com/en-us/pricing/details/ai-foundry-models/deepseek/
_DEEPSEEK_V4_PRO_COST: dict[str, Any] = {"input": 1.925, "output": 3.828, "cacheRead": 0.165, "cacheWrite": 0}


def _openai_models(models: list[ChatModel]) -> list[ChatModel]:
    """OpenAI models as Azure deploys them, including OpenAI's long-context price tier above 272K input tokens."""
    return [
        {
            **model,
            "api": "azure-openai-responses",
            "provider": "azure",
            "baseUrl": "",
            "contextWindow": _CONTEXT_WINDOW_OVERRIDES.get(model["id"], model["contextWindow"]),
        }
        for model in models
        if model["api"] == "openai-responses"
    ]


def _models(ctx: ProviderContext) -> list[ChatModel]:
    return [
        *_openai_models(ctx.provider_models("openai")),
        *(
            {**model, "provider": "azure", "baseUrl": "", "cost": dict(_DEEPSEEK_V4_PRO_COST)}
            for model in ctx.provider_models("deepseek")
            if model["id"] == "deepseek-v4-pro"
        ),
    ]


def _finalize(model: ChatModel, _ctx: ProviderContext) -> None:
    enable_openai_grammar_tools(model)
    if model["api"] == "openai-completions":
        # Azure 400s on DeepSeek's `thinking` field and on every prompt cache parameter, discards a
        # `developer` system message unbilled once reasoning_effort is set, and honours mid-convo ones (#9645).
        merge_compat(
            model,
            {
                "supportsDeveloperRole": False,
                "supportsMidConvoSystemMessages": True,
                "thinkingFormat": "openai",
                "supportsLongCacheRetention": False,
            },
        )


PROVIDER = ProviderDefinition(
    id="azure",
    models=_models,
    compat={"openai-completions": {"supportsStrictMode": True}},
    finalize=_finalize,
)
