"""Runs the provider definitions in a fixed order (pi: catalog/build.ts)."""

import copy
from typing import Any

from catalog.define import ApiCompatDefaults, ChatModel, ProviderContext, ProviderDefinition, merge_thinking
from catalog.families.anthropic import apply_claude_family_rules
from catalog.families.deepseek import apply_deepseek_compat_rules, apply_deepseek_family_rules
from catalog.families.openai import apply_openai_family_rules
from catalog.reasoning import get_effort_thinking_level_map
from catalog.sources import CatalogSources
from catalog.sources.models_dev import ModelsDevModel


# {"chat": [...], "image": [...], "classifier": [...]}
ProviderCatalog = dict[str, list[dict[str, Any]]]

# Keep the generated default no less restrictive than coding-agent's historical
# image preprocessing. Provider limits can narrow this profile, but unknown
# providers retain the cache-safe 2000px / 4.5 MiB behavior.
DEFAULT_IMAGE_RESIZE = {
    "maxWidth": 2000,
    "maxHeight": 2000,
    "maxBytes": int(4.5 * 1024 * 1024),
    "jpegQuality": 80,
}


def _model_key(model: dict[str, Any]) -> str:
    return f"{model['provider']}:{model['id']}"


def _apply_compat_defaults(model: ChatModel, defaults: ApiCompatDefaults | None) -> None:
    entry = (defaults or {}).get(model["api"])
    if entry is None:
        return
    compat = entry(model) if callable(entry) else entry
    model["compat"] = {**compat, **(model.get("compat") or {})}


def _sends_native_effort(model: ChatModel) -> bool:
    """Whether the model's API sends models.dev effort values to the host as-is."""
    api = model["api"]
    compat = model.get("compat") or {}
    if api == "anthropic-messages":
        return compat.get("forceAdaptiveThinking") is True
    if api in ("openai-responses", "azure-openai-responses", "openai-codex-responses"):
        return True
    if api != "openai-completions":
        return False
    thinking_format = compat.get("thinkingFormat")
    supports_reasoning_effort = compat.get("supportsReasoningEffort")
    return (thinking_format if thinking_format is not None else "openai") == "openai" and (
        supports_reasoning_effort if supports_reasoning_effort is not None else True
    )


def _apply_models_dev_effort(model: ChatModel, source: ModelsDevModel | None) -> None:
    if source is None or not source.reasoning_options or not _sends_native_effort(model):
        return
    mapping = get_effort_thinking_level_map(source.reasoning_options)
    if mapping:
        merge_thinking(model, mapping)


def _apply_image_resize_defaults(model: dict[str, Any]) -> None:
    if "image" not in model["input"]:
        return
    input_limits = model.get("inputLimits") or {}
    images = input_limits.get("images") or {}
    model["inputLimits"] = {
        **input_limits,
        "images": {**images, "resize": {**DEFAULT_IMAGE_RESIZE, **(images.get("resize") or {})}},
    }


def build_catalog(
    definitions: list[ProviderDefinition], sources: CatalogSources, strict: bool
) -> dict[str, ProviderCatalog]:
    """Build every provider's catalog from the fetched sources."""
    definitions_by_id = {definition.id: definition for definition in definitions}
    models_dev_sources: dict[str, ModelsDevModel] = {}
    stage_one: dict[str, list[ChatModel]] = {}
    building: set[str] = set()

    def context(provider_id: str) -> ProviderContext:
        return ProviderContext(
            provider_id=provider_id,
            sources=sources,
            strict=strict,
            provider_models=lambda other_id: copy.deepcopy(build_models(other_id)),
            models_dev_source=lambda model: models_dev_sources.get(_model_key(model)),
            record_models_dev_source=lambda model, source: models_dev_sources.__setitem__(_model_key(model), source),
        )

    def build_models(provider_id: str) -> list[ChatModel]:
        built = stage_one.get(provider_id)
        if built is not None:
            return built
        definition = definitions_by_id.get(provider_id)
        if definition is None:
            raise RuntimeError(f"Unknown provider: {provider_id}")
        if provider_id in building:
            raise RuntimeError(f"Provider {provider_id} depends on itself")
        building.add(provider_id)
        models = definition.models(context(provider_id)) if definition.models is not None else []
        building.discard(provider_id)
        for model in models:
            if model["provider"] != provider_id:
                raise RuntimeError(f"{provider_id} produced {_model_key(model)} for another provider")
        stage_one[provider_id] = models
        return models

    for definition in definitions:
        build_models(definition.id)

    catalogs: dict[str, ProviderCatalog] = {}
    for definition in definitions:
        ctx = context(definition.id)
        chat: list[ChatModel] = []
        seen: set[str] = set()
        for model in stage_one.get(definition.id) or []:
            # The first source listing a model wins.
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            chat.append(model)

        for model in chat:
            _apply_compat_defaults(model, definition.compat)
            apply_deepseek_compat_rules(model)
            _apply_models_dev_effort(model, ctx.models_dev_source(model))
            apply_openai_family_rules(model)
            apply_claude_family_rules(model)
            apply_deepseek_family_rules(model)
        if definition.finalize is not None:
            for model in chat:
                definition.finalize(model, ctx)
        for model in chat:
            if "compat" in model and not model["compat"]:
                del model["compat"]
            _apply_image_resize_defaults(model)
        if definition.finalize_all is not None:
            definition.finalize_all(chat, ctx)

        image = definition.images(ctx) if definition.images is not None else []
        for model in image:
            _apply_image_resize_defaults(model)
        classifier = definition.classifiers(ctx) if definition.classifiers is not None else []
        catalogs[definition.id] = {
            "chat": [{**model, "type": "chat"} for model in chat],
            "image": image,
            "classifier": classifier,
        }
    return catalogs
