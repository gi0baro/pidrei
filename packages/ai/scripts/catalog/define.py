"""`ProviderDefinition` and the helpers provider files use (pi: catalog/define.ts).

Models are plain dicts with the catalog's camelCase keys. Key order follows
pi's object literals (a merge appends a new key last, as assigning a new
property does in JS), so the vendored data stays diffable against pi's.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from catalog.sources import CatalogSources
from catalog.sources.models_dev import ModelsDevModel, models_dev_chat_models, models_dev_cost


ChatModel = dict[str, Any]
# Compat defaults per API. They sit under each model's own compat, so model
# values win. A function receives the model to vary defaults by model.
ApiCompatDefaults = dict[str, dict[str, Any] | Callable[[ChatModel], dict[str, Any]]]


@dataclass(frozen=True, slots=True)
class ProviderContext:
    """Passed to every provider hook."""

    # The provider being built.
    provider_id: str
    sources: CatalogSources
    # Fail instead of degrading when upstream data is unexpected.
    strict: bool
    # Copies of another provider's chat models before shared rules run.
    provider_models: Callable[[str], list[ChatModel]]
    # The models.dev entry a model was built from, if it was recorded.
    models_dev_source: Callable[[ChatModel], ModelsDevModel | None]
    # Record the models.dev entry so shared rules can use its reasoning metadata.
    record_models_dev_source: Callable[[ChatModel, ModelsDevModel], None]


@dataclass(frozen=True, slots=True)
class ProviderDefinition:
    """One built-in provider's catalog. The build runs, in order:
    1. `models()`: upstream models plus authored ones, with data corrections.
    2. `compat` defaults and the shared model-family rules in `families/`.
    3. `finalize()`: provider-specific behavior that must win over shared rules.
    4. Shared normalization (image input limits), then `finalize_all()`.
    """

    id: str
    models: Callable[[ProviderContext], list[ChatModel]] | None = None
    images: Callable[[ProviderContext], list[dict[str, Any]]] | None = None
    classifiers: Callable[[ProviderContext], list[dict[str, Any]]] | None = None
    compat: ApiCompatDefaults | None = None
    finalize: Callable[[ChatModel, ProviderContext], None] | None = None
    finalize_all: Callable[[list[ChatModel], ProviderContext], None] | None = None


def merge_compat(model: ChatModel, compat: dict[str, Any]) -> None:
    """Shallow-merge compat flags into a model."""
    model["compat"] = {**(model.get("compat") or {}), **compat}


def merge_thinking(model: ChatModel, mapping: dict[str, str | None]) -> None:
    """Shallow-merge thinking level mappings into a model. Later values win."""
    model["thinkingLevelMap"] = {**(model.get("thinkingLevelMap") or {}), **mapping}


def with_input_limits(model: ChatModel, limits: dict[str, Any]) -> None:
    """Provider input limits for image-capable models. Limits the model already carries take precedence."""
    if "image" not in model["input"]:
        return
    current = model.get("inputLimits") or {}
    model["inputLimits"] = {
        **limits,
        **current,
        "images": {**(limits.get("images") or {}), **(current.get("images") or {})},
    }


def add_missing(models: list[ChatModel], additions: list[ChatModel]) -> list[ChatModel]:
    """Add models only when upstream has not listed them (yet)."""
    for addition in additions:
        if not any(model["id"] == addition["id"] for model in models):
            models.append(addition)
    return models


def models_dev_models(
    ctx: ProviderContext,
    source_provider: str,
    *,
    api: str,
    base_url: str,
    compat: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    protocol: str | None = None,
    skip_deprecated: bool = False,
    record_reasoning: bool = True,
    default_context_window: int | None = None,
    default_max_tokens: int | None = None,
    include: Callable[[ModelsDevModel], bool] | None = None,
    map_model: Callable[[ChatModel, ModelsDevModel], ChatModel | None] | None = None,
) -> list[ChatModel]:
    """Map one models.dev provider's tool-capable chat models onto PiDrei models.

    `protocol` keeps only models models.dev lists on that API protocol (e.g.
    "responses"); `skip_deprecated` skips models models.dev marks deprecated;
    `record_reasoning` exposes models.dev effort levels through the shared rules;
    the default limits apply when models.dev has none (4096 each otherwise);
    `include` is an extra filter (models without tool support are always
    skipped); `map_model` (pi's `map`) adjusts or replaces the mapped model,
    None skips it.
    """
    models: list[ChatModel] = []
    for source in models_dev_chat_models(ctx.sources.models_dev, source_provider):
        if not source.tool_call:
            continue
        if protocol and not any(entry.protocol == protocol for entry in source.apis):
            continue
        if skip_deprecated and source.status == "deprecated":
            continue
        if include is not None and not include(source):
            continue
        base: ChatModel = {
            "id": source.id,
            "name": source.name or source.id,
            "api": api,
            "provider": ctx.provider_id,
            "baseUrl": base_url,
            "reasoning": source.reasoning,
            "input": ["text", "image"] if source.image_input else ["text"],
            "cost": models_dev_cost(source.cost),
            "contextWindow": source.context_window or default_context_window or 4096,
            "maxTokens": source.max_tokens or default_max_tokens or 4096,
            **({"headers": dict(headers)} if headers is not None else {}),
            **({"compat": dict(compat)} if compat is not None else {}),
        }
        model = map_model(base, source) if map_model is not None else base
        if model is None:
            continue
        if record_reasoning:
            ctx.record_models_dev_source(model, source)
        models.append(model)
    return models
