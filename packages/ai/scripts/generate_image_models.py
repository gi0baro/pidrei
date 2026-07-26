"""Port of pi's scripts/generate-image-models.ts.

Fetches OpenRouter's catalog and writes the vendored image-model JSON that
`image_models_generated.py` loads. `parse_openrouter_image_models` is the pure
half, and the half pi's spec covers.
"""

import json
from pathlib import Path
from typing import Any


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_URL = f"{OPENROUTER_BASE_URL}/models"

_MODALITIES = ("text", "image")


def _modalities(values: Any) -> list[str]:
    seen: list[str] = []
    for modality in values or []:
        if modality in _MODALITIES and modality not in seen:
            seen.append(modality)
    return seen


def _price(pricing: dict[str, Any], key: str) -> float:
    try:
        return float(pricing.get(key) or "0") * 1_000_000
    except TypeError, ValueError:
        return 0.0


def parse_openrouter_image_models(payload: Any, strict: bool) -> list[dict[str, Any]]:
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list) or len(data) == 0:
        if strict:
            raise ValueError("OpenRouter API returned a missing or empty image model list")
        return []

    models: list[dict[str, Any]] = []
    for model in data:
        architecture = model.get("architecture") or {}
        input_modalities = _modalities(architecture.get("input_modalities"))
        output_modalities = _modalities(architecture.get("output_modalities"))

        if "image" not in output_modalities:
            continue
        if not input_modalities:
            input_modalities.append("text")

        pricing = model.get("pricing") or {}
        models.append(
            {
                "id": model.get("id"),
                "name": model.get("name"),
                "api": "openrouter-images",
                "provider": "openrouter",
                "baseUrl": OPENROUTER_BASE_URL,
                "input": input_modalities,
                "output": output_modalities,
                "cost": {
                    "input": _price(pricing, "prompt"),
                    "output": _price(pricing, "completion"),
                    "cacheRead": _price(pricing, "input_cache_read"),
                    "cacheWrite": _price(pricing, "input_cache_write"),
                },
            }
        )

    if strict and len(models) == 0:
        raise ValueError("OpenRouter API returned no usable image models")
    return models


def write_catalog(models: list[dict[str, Any]], destination: Path) -> None:
    """One file per provider, named for it, under `providers/image_data/`."""
    destination.write_text(json.dumps({model["id"]: model for model in models}, indent=2) + "\n")
