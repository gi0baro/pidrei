"""pi-shaped camelCase model wire serde and compat merging.

pi passes raw JS objects around (models.json fragments, pi.dev catalog
entries, ModelsStore JSON); pidrei's `Model`/compat are typed dataclasses, so
the conversions pi gets for free live here. Parsing of full catalog objects is
shared with the vendored-data loader (`pidrei_ai.models_generated`).
"""

import re
from dataclasses import fields
from typing import Any

from pidrei_ai.models_generated import parse_model_dict
from pidrei_ai.types import (
    AnthropicMessagesCompat,
    BedrockCompat,
    Model,
    ModelCompat,
    ModelCost,
    ModelCostTier,
    OpenAICompletionsCompat,
    OpenAIResponsesCompat,
)


__all__ = [
    "compat_to_dict",
    "cost_from_dict",
    "merge_compat",
    "model_to_dict",
    "parse_compat",
    "parse_model_dict",
]

_COMPAT_CLASSES: dict[str, type] = {
    "openai-completions": OpenAICompletionsCompat,
    "openai-responses": OpenAIResponsesCompat,
    "azure-openai-responses": OpenAIResponsesCompat,
    "openai-codex-responses": OpenAIResponsesCompat,
    "anthropic-messages": AnthropicMessagesCompat,
    "bedrock-converse-stream": BedrockCompat,
}

# Keys whose acronyms defeat generic conversion, in both directions.
_SPECIAL_TO_SNAKE = {"supportsOpenAIGrammarTools": "supports_openai_grammar_tools"}
_SPECIAL_TO_CAMEL = {snake: camel for camel, snake in _SPECIAL_TO_SNAKE.items()}

# Nested compat objects deep-merged one level by pi's mergeCompat.
_NESTED_COMPAT_KEYS = ("openRouterRouting", "vercelGatewayRouting", "chatTemplateKwargs")


def _snake(name: str) -> str:
    special = _SPECIAL_TO_SNAKE.get(name)
    if special is not None:
        return special
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def _camel(name: str) -> str:
    special = _SPECIAL_TO_CAMEL.get(name)
    if special is not None:
        return special
    parts = name.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


def parse_compat(api: str, raw: dict[str, Any] | None) -> ModelCompat | None:
    """Build the api-specific compat dataclass from a raw camelCase object.

    Unknown keys are dropped: pi's TypeBox schemas allow extra properties and
    its adapters never read keys outside the typed compat surface, so dropping
    them is observably equivalent. An api with no compat class yields None.
    """
    if raw is None:
        return None
    compat_class = _COMPAT_CLASSES.get(api)
    if compat_class is None:
        return None
    known = {field.name for field in fields(compat_class)}
    kwargs = {}
    for key, value in raw.items():
        name = _snake(key)
        if name in known:
            kwargs[name] = value
    return compat_class(**kwargs)


def compat_to_dict(compat: ModelCompat | None) -> dict[str, Any]:
    if compat is None:
        return {}
    raw: dict[str, Any] = {}
    for field in fields(compat):
        value = getattr(compat, field.name)
        if value is not None:
            raw[_camel(field.name)] = value
    return raw


def merge_compat(api: str, base: ModelCompat | None, override: dict[str, Any] | None) -> ModelCompat | None:
    """Mirror of pi's mergeCompat: shallow merge with one-level deep merge of
    the routing/kwargs objects, operating on the raw camelCase shape."""
    if not override:
        return base
    base_raw = compat_to_dict(base)
    merged = {**base_raw, **override}
    for key in _NESTED_COMPAT_KEYS:
        base_value = base_raw.get(key)
        override_value = override.get(key)
        if isinstance(base_value, dict) or isinstance(override_value, dict):
            merged[key] = {
                **(base_value if isinstance(base_value, dict) else {}),
                **(override_value if isinstance(override_value, dict) else {}),
            }
    return parse_compat(api, merged)


def cost_tiers_from_list(raw_tiers: list[dict[str, Any]]) -> list[ModelCostTier]:
    return [
        ModelCostTier(
            input=tier["input"],
            output=tier["output"],
            cache_read=tier["cacheRead"],
            cache_write=tier["cacheWrite"],
            input_tokens_above=tier["inputTokensAbove"],
        )
        for tier in raw_tiers
    ]


def cost_from_dict(raw: dict[str, Any]) -> ModelCost:
    tiers = cost_tiers_from_list(raw["tiers"]) if raw.get("tiers") else None
    return ModelCost(
        input=raw["input"],
        output=raw["output"],
        cache_read=raw["cacheRead"],
        cache_write=raw["cacheWrite"],
        tiers=tiers,
    )


def _cost_to_dict(cost: ModelCost) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "input": cost.input,
        "output": cost.output,
        "cacheRead": cost.cache_read,
        "cacheWrite": cost.cache_write,
    }
    if cost.tiers is not None:
        raw["tiers"] = [
            {
                "inputTokensAbove": tier.input_tokens_above,
                "input": tier.input,
                "output": tier.output,
                "cacheRead": tier.cache_read,
                "cacheWrite": tier.cache_write,
            }
            for tier in cost.tiers
        ]
    return raw


def model_to_dict(model: Model) -> dict[str, Any]:
    """Serialize a Model to the pi camelCase wire shape (inverse of parse_model_dict)."""
    raw: dict[str, Any] = {
        "id": model.id,
        "name": model.name,
        "api": model.api,
        "provider": model.provider,
        "baseUrl": model.base_url,
        "reasoning": model.reasoning,
        "input": list(model.input),
        "cost": _cost_to_dict(model.cost),
        "contextWindow": model.context_window,
        "maxTokens": model.max_tokens,
    }
    if model.thinking_level_map is not None:
        raw["thinkingLevelMap"] = dict(model.thinking_level_map)
    if model.headers is not None:
        raw["headers"] = dict(model.headers)
    if model.compat is not None:
        raw["compat"] = compat_to_dict(model.compat)
    return raw
