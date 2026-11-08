"""Ant Ling (pi: catalog/providers/ant-ling.ts)."""

from typing import Any

from catalog.define import ChatModel, ProviderContext, ProviderDefinition, merge_thinking


_COMPAT: dict[str, Any] = {
    "supportsStore": False,
    "supportsDeveloperRole": False,
    "supportsReasoningEffort": False,
    "maxTokensField": "max_tokens",
    "supportsLongCacheRetention": False,
}


def _model(model_id: str, name: str, reasoning: bool, cost: dict[str, Any]) -> ChatModel:
    return {
        "id": model_id,
        "name": name,
        "api": "openai-completions",
        "baseUrl": "https://api.ant-ling.com/v1",
        "provider": "ant-ling",
        "reasoning": reasoning,
        "input": ["text"],
        "cost": cost,
        "contextWindow": 262144,
        "maxTokens": 65536,
        "compat": {**_COMPAT, "thinkingFormat": "ant-ling"} if reasoning else dict(_COMPAT),
    }


def _models(_ctx: ProviderContext) -> list[ChatModel]:
    return [
        _model(
            "Ling-2.6-flash", "Ling 2.6 Flash", False, {"input": 0.01, "output": 0.02, "cacheRead": 0, "cacheWrite": 0}
        ),
        _model("Ling-2.6-1T", "Ling 2.6 1T", False, {"input": 0.06, "output": 0.25, "cacheRead": 0, "cacheWrite": 0}),
        _model("Ring-2.6-1T", "Ring 2.6 1T", True, {"input": 0.06, "output": 0.25, "cacheRead": 0, "cacheWrite": 0}),
    ]


def _finalize(model: ChatModel, _ctx: ProviderContext) -> None:
    # Ring reasons by default. Only high/xhigh have documented explicit effort controls.
    if model["reasoning"]:
        merge_thinking(
            model, {"off": None, "minimal": None, "low": None, "medium": None, "high": "high", "xhigh": "xhigh"}
        )


PROVIDER = ProviderDefinition(
    id="ant-ling",
    models=_models,
    compat={
        "openai-completions": {
            "supportsStore": False,
            "supportsDeveloperRole": False,
            "supportsReasoningEffort": False,
            "maxTokensField": "max_tokens",
            "thinkingFormat": "ant-ling",
            "supportsStrictMode": True,
            "supportsLongCacheRetention": False,
        },
    },
    finalize=_finalize,
)
