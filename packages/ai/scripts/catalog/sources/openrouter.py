"""OpenRouter catalog (pi: catalog/sources/openrouter.ts)."""

from typing import Any

import tonio.colored as tonio
from punkreq.tonio import Client

from catalog.reasoning import get_effort_thinking_level_map
from pidrei_ai.providers.openrouter_api import openrouter_cost


OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# {"chat": [...], "images": [...], "classifiers": [...]}
OpenRouterCatalog = dict[str, list[dict[str, Any]]]


def get_openrouter_thinking_level_map(reasoning: dict[str, Any] | None) -> dict[str, str | None] | None:
    """Convert OpenRouter's reasoning metadata into PiDrei model capabilities."""
    if not reasoning:
        return None
    supported_efforts = reasoning.get("supported_efforts")
    if not supported_efforts:
        return {"off": None} if reasoning.get("mandatory") is True else None

    # OpenRouter's supported_efforts uses the same effort values as models.dev reasoning_options,
    # so both sources can share the same PiDrei thinking-level conversion.
    mapping = get_effort_thinking_level_map([{"type": "effort", "values": supported_efforts}])
    if not mapping:
        return {"off": None} if reasoning.get("mandatory") is True else None
    return {**mapping, "off": None if reasoning.get("mandatory") is True else "none"}


_MODALITIES = ("text", "image")


def _modalities(values: list[str] | None) -> list[str]:
    seen: list[str] = []
    for value in values or []:
        if value in _MODALITIES and value not in seen:
            seen.append(value)
    return seen


def build_openrouter_catalog(
    listed: list[dict[str, Any]],
    image_listed: list[dict[str, Any]],
    decision_listed: list[dict[str, Any]],
) -> OpenRouterCatalog:
    """Build the OpenRouter catalog from the default listing and the
    `output_modalities=image` and `output_modalities=decisions` listings. The
    default listing omits image-only and decision models, so those come from
    the other listings. An upstream model may appear in several results; it then
    gets separate entries per operation."""
    chat: list[dict[str, Any]] = []
    for model in listed:
        supported = model.get("supported_parameters") or []
        # Only include models that support tools
        if "tools" not in supported:
            continue
        # Parse input modalities
        model_input = ["text"]
        if "image" in ((model.get("architecture") or {}).get("modality") or ""):
            model_input.append("image")

        top_provider = model.get("top_provider") or {}
        thinking_level_map = get_openrouter_thinking_level_map(model.get("reasoning"))
        # Includes the `~anthropic/claude-*-latest` aliases, which OpenRouter's Messages endpoint accepts.
        use_anthropic_messages = model["id"].startswith(("anthropic/", "~anthropic/")) and not model["id"].endswith(
            ":batch"
        )
        chat.append(
            {
                "type": "chat",
                "id": model["id"],
                "name": model["name"],
                "api": "anthropic-messages" if use_anthropic_messages else "openai-completions",
                "baseUrl": "https://openrouter.ai/api" if use_anthropic_messages else "https://openrouter.ai/api/v1",
                "provider": "openrouter",
                "reasoning": "reasoning" in supported,
                **({"thinkingLevelMap": thinking_level_map} if thinking_level_map else {}),
                "input": model_input,
                "cost": openrouter_cost(model.get("pricing")),
                "contextWindow": top_provider.get("context_length") or model.get("context_length") or 4096,
                "maxTokens": top_provider.get("max_completion_tokens") or 4096,
            }
        )

    images: list[dict[str, Any]] = []
    for model in image_listed:
        if any(entry["id"] == model["id"] for entry in images):
            continue
        architecture = model.get("architecture") or {}
        output = _modalities(architecture.get("output_modalities"))
        if "image" not in output:
            continue
        model_input = _modalities(architecture.get("input_modalities"))
        images.append(
            {
                "type": "image",
                "id": model["id"],
                "name": model["name"],
                "api": "openrouter-images",
                "provider": "openrouter",
                "baseUrl": "https://openrouter.ai/api/v1",
                "input": model_input or ["text"],
                "output": output,
                "cost": openrouter_cost(model.get("pricing")),
            }
        )

    # Decision models such as TypeSafe's Jev are served through OpenRouter's
    # TypeSafe-compatible System One endpoint.
    classifiers: list[dict[str, Any]] = []
    for model in decision_listed:
        if any(entry["id"] == model["id"] for entry in classifiers):
            continue
        architecture = model.get("architecture") or {}
        if "decisions" not in (architecture.get("output_modalities") or []):
            continue
        model_input = _modalities(architecture.get("input_modalities"))
        top_provider = model.get("top_provider") or {}
        classifiers.append(
            {
                "type": "classifier",
                "id": model["id"],
                "name": model["name"],
                "api": "typesafe-system-one",
                "provider": "openrouter",
                "baseUrl": "https://openrouter.ai/api/v1",
                "input": model_input or ["text"],
                "cost": openrouter_cost(model.get("pricing")),
                "contextWindow": top_provider.get("context_length") or model.get("context_length") or 4096,
                # OpenRouter answers TypeSafe `assets` with 200 but drops the images.
                "compat": {"imageInput": "state-image-urls", "sendImageOptInHeader": False},
            }
        )

    return {"chat": chat, "images": images, "classifiers": classifiers}


async def _fetch_openrouter_list(client: Client, query: str) -> list[dict[str, Any]]:
    response = await client.get(f"{OPENROUTER_MODELS_URL}{query}")
    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter API returned {response.status_code}")
    data = await response.json()
    return data.get("data") or []


async def fetch_openrouter_catalog(client: Client) -> OpenRouterCatalog:
    """Fetch the tool-capable chat, image, and decision listings."""
    listed, image_listed, decision_listed = await tonio.spawn(
        _fetch_openrouter_list(client, ""),
        _fetch_openrouter_list(client, "?output_modalities=image"),
        _fetch_openrouter_list(client, "?output_modalities=decisions"),
    )
    return build_openrouter_catalog(listed, image_listed, decision_listed)
