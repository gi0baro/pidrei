"""DeepSeek (pi: catalog/providers/deepseek.ts)."""

import copy

from catalog.define import ChatModel, ProviderContext, ProviderDefinition, merge_compat
from catalog.families.deepseek import DEEPSEEK_COMPAT, DEEPSEEK_V4_FLASH_THINKING_LEVEL_MAP


# models.dev's DeepSeek entries lag the API; pricing from
# https://api-docs.deepseek.com/quick_start/pricing. DeepSeek also offers
# time-based off-peak rates, which the cost schema cannot represent yet.
_MODELS: list[ChatModel] = [
    {
        "id": "deepseek-flash",
        "name": "DeepSeek V4.1 Flash",
        "api": "openai-completions",
        "baseUrl": "https://api.deepseek.com",
        "provider": "deepseek",
        "reasoning": True,
        "thinkingLevelMap": DEEPSEEK_V4_FLASH_THINKING_LEVEL_MAP,
        "input": ["text", "image"],
        "cost": {"input": 0.3, "output": 1.2, "cacheRead": 0.006, "cacheWrite": 0},
        "contextWindow": 1000000,
        "maxTokens": 384000,
        "compat": DEEPSEEK_COMPAT,
    },
    {
        "id": "deepseek-v4-pro",
        "name": "DeepSeek V4 Pro",
        "api": "openai-completions",
        "baseUrl": "https://api.deepseek.com",
        "provider": "deepseek",
        "reasoning": True,
        "input": ["text"],
        "cost": {"input": 1.32, "output": 3.96, "cacheRead": 0.044, "cacheWrite": 0},
        "contextWindow": 1000000,
        "maxTokens": 384000,
        "compat": DEEPSEEK_COMPAT,
    },
]


def _models(_ctx: ProviderContext) -> list[ChatModel]:
    return copy.deepcopy(_MODELS)


def _finalize(model: ChatModel, _ctx: ProviderContext) -> None:
    # DeepSeek V4 Pro accepts plain system text in place mid-conversation.
    if model["id"] == "deepseek-v4-pro":
        merge_compat(model, {"supportsMidConvoSystemMessages": True})


PROVIDER = ProviderDefinition(
    id="deepseek",
    models=_models,
    compat={
        "openai-completions": {
            "supportsStore": False,
            "supportsDeveloperRole": False,
            "maxTokensField": "max_tokens",
            "requiresReasoningContentOnAssistantMessages": True,
            "thinkingFormat": "deepseek",
            "supportsStrictMode": True,
        },
    },
    finalize=_finalize,
)
