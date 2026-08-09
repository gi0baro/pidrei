"""Built-in model catalog (pi: src/models.generated.ts + providers/data/*.json).

Loads the vendored catalog JSON (produced by scripts/generate_models.py,
pi-shaped: camelCase keys, grouped `{api: {modelId: Model}}`) into typed
`Model` dataclasses. JSON `null` values in `thinkingLevelMap` are preserved as
present-with-None entries — the null-vs-missing distinction drives
`get_supported_thinking_levels`.
"""

import json
import re
from dataclasses import fields
from importlib import resources
from typing import Any

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


_DATA_DIR = resources.files("pidrei_ai.providers") / "data"


def _json_files(directory):
    """`.json` entries in a Traversable directory (importlib.resources)."""
    return [entry for entry in directory.iterdir() if entry.name.endswith(".json")]


_COMPAT_CLASSES: dict[str, type] = {
    "openai-completions": OpenAICompletionsCompat,
    "openai-responses": OpenAIResponsesCompat,
    "azure-openai-responses": OpenAIResponsesCompat,
    "openai-codex-responses": OpenAIResponsesCompat,
    "anthropic-messages": AnthropicMessagesCompat,
    "bedrock-converse-stream": BedrockCompat,
}


# Keys whose acronyms defeat generic camelCase -> snake_case conversion.
_SPECIAL_KEYS = {"supportsOpenAIGrammarTools": "supports_openai_grammar_tools"}


def _snake(name: str) -> str:
    special = _SPECIAL_KEYS.get(name)
    if special is not None:
        return special
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def _compat_fields(compat_class: type) -> frozenset[str]:
    return frozenset(field.name for field in fields(compat_class))


# Every compat field pi declares, across all APIs. pi's generator sometimes
# attaches a field belonging to another API's compat interface (e.g. the
# hardcoded `supportsReasoningEffort: false` for OpenCode's grok-build-0.1,
# which models.dev has since re-typed as an `openai-responses` model). TS
# carries the extra key along and no adapter ever reads it, so it is inert
# upstream; the typed dataclasses here cannot hold it, so it is dropped.
# A key no compat class declares is a genuinely new upstream field and still
# raises — that is what this check is for.
_ANY_COMPAT_FIELD = frozenset().union(*(_compat_fields(cls) for cls in set(_COMPAT_CLASSES.values())))


def _parse_compat(api: str, raw: dict[str, Any]) -> ModelCompat:
    compat_class = _COMPAT_CLASSES.get(api)
    if compat_class is None:
        raise ValueError(f"No compat class for api {api!r}")
    known = _compat_fields(compat_class)
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        name = _snake(key)
        if name not in known:
            if name in _ANY_COMPAT_FIELD:
                continue
            raise ValueError(f"Unknown {compat_class.__name__} field {key!r} in catalog data")
        kwargs[name] = value
    return compat_class(**kwargs)


def _parse_cost(raw: dict[str, Any]) -> ModelCost:
    tiers = [
        ModelCostTier(
            input=tier["input"],
            output=tier["output"],
            cache_read=tier["cacheRead"],
            cache_write=tier["cacheWrite"],
            input_tokens_above=tier["inputTokensAbove"],
        )
        for tier in raw.get("tiers", [])
    ]
    return ModelCost(
        input=raw.get("input", 0),
        output=raw.get("output", 0),
        cache_read=raw.get("cacheRead", 0),
        cache_write=raw.get("cacheWrite", 0),
        tiers=tiers or None,
    )


def _parse_model(raw: dict[str, Any]) -> Model:
    return Model(
        id=raw["id"],
        name=raw["name"],
        api=raw["api"],
        provider=raw["provider"],
        base_url=raw["baseUrl"],
        reasoning=raw["reasoning"],
        input=list(raw["input"]),
        cost=_parse_cost(raw["cost"]),
        context_window=raw["contextWindow"],
        max_tokens=raw["maxTokens"],
        thinking_level_map=dict(raw["thinkingLevelMap"]) if "thinkingLevelMap" in raw else None,
        headers=dict(raw["headers"]) if "headers" in raw else None,
        compat=_parse_compat(raw["api"], raw["compat"]) if "compat" in raw else None,
    )


def parse_model_dict(raw: dict[str, Any]) -> Model:
    """Parse one pi-shaped camelCase model object (vendored data, pi.dev catalog)."""
    return _parse_model(raw)


def _load_catalog() -> dict[str, list[Model]]:
    catalog: dict[str, list[Model]] = {}
    # Traversable has no glob(); iterdir() + an explicit key keeps the
    # filename ordering the generated catalogs rely on.
    for path in sorted(_json_files(_DATA_DIR), key=lambda entry: entry.name):
        if path.stem.startswith("_"):  # _manifest.json and friends
            continue
        provider_id = path.stem
        by_api = json.loads(path.read_text())
        catalog[provider_id] = [_parse_model(raw) for api_models in by_api.values() for raw in api_models.values()]
    return catalog


# Keyed by provider id, like pi's MODELS aggregate.
MODELS: dict[str, list[Model]] = _load_catalog()
