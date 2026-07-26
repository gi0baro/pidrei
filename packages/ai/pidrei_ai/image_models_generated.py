"""Built-in image model catalog (pi: src/image-models.generated.ts).

Loads the vendored catalog JSON into `ImagesModel` dataclasses, the same shape
`models_generated.py` uses for the chat catalog.
"""

import json
from pathlib import Path

from pidrei_ai.types import ImagesModel, ModelCost


# Deliberately not under `providers/data/`: that directory is the chat catalog,
# and its manifest validator owns every file in it.
_DATA_DIR = Path(__file__).parent / "providers" / "image_data"


def _to_model(raw: dict) -> ImagesModel:
    cost = raw.get("cost") or {}
    return ImagesModel(
        id=raw["id"],
        name=raw["name"],
        api=raw["api"],
        provider=raw["provider"],
        base_url=raw["baseUrl"],
        input=list(raw.get("input") or []),
        output=list(raw.get("output") or []),
        cost=ModelCost(
            input=cost.get("input", 0.0),
            output=cost.get("output", 0.0),
            cache_read=cost.get("cacheRead", 0.0),
            cache_write=cost.get("cacheWrite", 0.0),
        ),
        headers=raw.get("headers"),
    )


def _load() -> dict[str, dict[str, ImagesModel]]:
    catalog: dict[str, dict[str, ImagesModel]] = {}
    for path in sorted(_DATA_DIR.glob("*.json")):
        raw = json.loads(path.read_text())
        catalog[path.stem] = {id: _to_model(model) for id, model in raw.items()}
    return catalog


IMAGE_MODELS: dict[str, dict[str, ImagesModel]] = _load()
