"""Built-in model catalog (pi: src/models.generated.ts, model-catalog.ts + providers/data/*.json).

Loads the vendored catalog JSON (produced by scripts/generate_models.py,
pi-shaped: camelCase keys, grouped `{api: {"type:id": model}}`) into typed
`Model`/`ImageModel`/`ClassifierModel` dataclasses, split by each entry's
`type` (an entry without one is a chat model). JSON `null` values in
`thinkingLevelMap` are preserved as present-with-None entries — the
null-vs-missing distinction drives `get_supported_thinking_levels`.
"""

import json
import re
from dataclasses import fields
from importlib import resources
from typing import Any

from pidrei_ai.types import (
    AnthropicAllowedFallbackModel,
    AnthropicMessagesCompat,
    AnyModel,
    BedrockCompat,
    ClassifierModel,
    ImageModel,
    MistralConversationsCompat,
    Model,
    ModelCompat,
    ModelCost,
    ModelCostTier,
    ModelImageInputLimits,
    ModelImageResizeOptions,
    ModelInputLimits,
    ModelType,
    OpenAICompletionsCompat,
    OpenAIResponsesCompat,
)
from pidrei_ai.utils.model_operations import is_model_type


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
    "mistral-conversations": MistralConversationsCompat,
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


def _parse_fallback_targets(raw: list[dict[str, Any]]) -> list[AnthropicAllowedFallbackModel]:
    return [
        AnthropicAllowedFallbackModel(
            provider=target["provider"], model=target["model"], cost=_parse_cost(target["cost"])
        )
        for target in raw
    ]


# Compat fields whose catalog value is a nested object rather than a scalar.
# Shared with `pidrei.core.model_wire`, which parses the same shape out of
# models.json and has to build the same dataclasses.
COMPAT_FIELD_PARSERS = {"allowed_fallback_models": _parse_fallback_targets}


def _parse_compat(api: str, raw: dict[str, Any]) -> ModelCompat:
    compat_class = _COMPAT_CLASSES.get(api)
    if compat_class is None:
        raise ValueError(f"No compat class for api {api!r}")
    known = _compat_fields(compat_class)
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        name = _snake(key)
        # Unknown keys are dropped, as pi does: its catalog parse is a plain
        # spread and no adapter reads outside the typed compat surface, so a
        # field from a newer pi.dev catalog (or one pi's generator attached
        # from another API's interface) is inert there and must be here too.
        if name not in known:
            continue
        parser = COMPAT_FIELD_PARSERS.get(name)
        kwargs[name] = parser(value) if parser is not None else value
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


def parse_input_limits(raw: dict[str, Any]) -> ModelInputLimits:
    """Parse pi's camelCase `inputLimits` object; absent keys stay None."""
    images = raw.get("images")
    resize = images.get("resize") if images is not None else None
    return ModelInputLimits(
        max_request_bytes=raw.get("maxRequestBytes"),
        images=(
            ModelImageInputLimits(
                resize=(
                    ModelImageResizeOptions(
                        max_width=resize.get("maxWidth"),
                        max_height=resize.get("maxHeight"),
                        max_bytes=resize.get("maxBytes"),
                        jpeg_quality=resize.get("jpegQuality"),
                    )
                    if resize is not None
                    else None
                ),
                max_per_message=images.get("maxPerMessage"),
                max_per_request=images.get("maxPerRequest"),
            )
            if images is not None
            else None
        ),
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
        input_limits=parse_input_limits(raw["inputLimits"]) if "inputLimits" in raw else None,
        prompt_cache=dict(raw["promptCache"]) if "promptCache" in raw else None,
        thinking_level_map=dict(raw["thinkingLevelMap"]) if "thinkingLevelMap" in raw else None,
        headers=dict(raw["headers"]) if "headers" in raw else None,
        compat=_parse_compat(raw["api"], raw["compat"]) if "compat" in raw else None,
        type="chat" if raw.get("type") == "chat" else None,
    )


def _parse_image_model(raw: dict[str, Any]) -> ImageModel:
    return ImageModel(
        id=raw["id"],
        name=raw["name"],
        api=raw["api"],
        provider=raw["provider"],
        base_url=raw["baseUrl"],
        input=list(raw["input"]),
        output=list(raw["output"]),
        cost=_parse_cost(raw["cost"]),
        input_limits=parse_input_limits(raw["inputLimits"]) if "inputLimits" in raw else None,
        headers=dict(raw["headers"]) if "headers" in raw else None,
    )


def _parse_classifier_model(raw: dict[str, Any]) -> ClassifierModel:
    return ClassifierModel(
        id=raw["id"],
        name=raw["name"],
        api=raw["api"],
        provider=raw["provider"],
        base_url=raw["baseUrl"],
        input=list(raw["input"]),
        cost=_parse_cost(raw["cost"]),
        context_window=raw["contextWindow"],
        input_limits=parse_input_limits(raw["inputLimits"]) if "inputLimits" in raw else None,
        headers=dict(raw["headers"]) if "headers" in raw else None,
    )


_PARSERS_BY_TYPE = {"chat": _parse_model, "image": _parse_image_model, "classifier": _parse_classifier_model}


def parse_model_dict(raw: dict[str, Any]) -> Model:
    """Parse one pi-shaped camelCase chat model object (vendored data, pi.dev catalog)."""
    return _parse_model(raw)


def parse_any_model_dict(raw: dict[str, Any]) -> AnyModel | None:
    """Parse one pi-shaped camelCase model object of any type. An entry
    without `type` is a chat model; one of a type this version does not know
    yields None (pi drops those wherever stored or fetched catalogs are read)."""
    parser = _PARSERS_BY_TYPE.get(raw.get("type", "chat"))
    return parser(raw) if parser is not None else None


def _load_catalog() -> dict[str, list[AnyModel]]:
    catalog: dict[str, list[AnyModel]] = {}
    # Traversable has no glob(); iterdir() + an explicit key keeps the
    # filename ordering the generated catalogs rely on.
    for path in sorted(_json_files(_DATA_DIR), key=lambda entry: entry.name):
        if path.stem.startswith("_"):  # _manifest.json and friends
            continue
        provider_id = path.stem
        by_api = json.loads(path.read_text())
        models = (parse_any_model_dict(raw) for api_models in by_api.values() for raw in api_models.values())
        catalog[provider_id] = [model for model in models if model is not None]
    return catalog


def _of_type(catalog: dict[str, list[AnyModel]], type: ModelType) -> dict[str, list[Any]]:
    """pi's `flatten*ModelCatalog`: one provider's catalog restricted to one type."""
    return {
        provider_id: [model for model in models if is_model_type(model, type)]
        for provider_id, models in catalog.items()
    }


_CATALOG = _load_catalog()

# Keyed by provider id, like pi's MODELS / IMAGE_MODELS / CLASSIFIER_MODELS aggregates.
MODELS: dict[str, list[Model]] = _of_type(_CATALOG, "chat")
IMAGE_MODELS: dict[str, list[ImageModel]] = _of_type(_CATALOG, "image")
CLASSIFIER_MODELS: dict[str, list[ClassifierModel]] = _of_type(_CATALOG, "classifier")
