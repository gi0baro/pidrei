"""Mirror of pi's catalog-generation-utils.ts.

pi writes a `node --import` preload that serves every upstream catalog the
generator fetches and runs the generator in a subprocess. PiDrei's generator
fetches everything up front (`load_catalog_sources`) and builds in memory, so
the mock is the `CatalogSources` the fetch would return: the given models.dev
v2 catalog, and the same minimal responses for the other sources. The build
runs in-process against it.
"""

import sys
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from catalog.build import build_catalog
from catalog.providers import PROVIDERS
from catalog.sources import CatalogSources
from catalog.sources.openrouter import build_openrouter_catalog


def upstream_sources(models_dev: dict[str, Any]) -> CatalogSources:
    """The given models.dev v2 catalog, and minimal responses for the other sources."""
    return CatalogSources(
        models_dev=models_dev,
        models_dev_decisions={
            "typesafe/jev-latest": {
                "id": "typesafe/jev-latest",
                "name": "Jev",
                "type": "decision",
                "limit": {"context": 64000},
            }
        },
        open_router=build_openrouter_catalog(
            [],
            [
                {
                    "id": "test/image",
                    "name": "Test Image",
                    "architecture": {"input_modalities": ["text"], "output_modalities": ["image"]},
                    "pricing": {},
                }
            ],
            [],
        ),
        ai_gateway={"chat": [], "classifiers": []},
        # pi fetches NVIDIA NIM only when models.dev lists NVIDIA models; the mock lists none.
        nvidia_model_ids={},
    )


def generate_chat_models(models_dev: dict[str, Any], provider: str, *, strict: bool = False) -> dict[str, dict]:
    """The provider's chat catalog keyed by id (pi's `--json-output` `providers/<id>.json`)."""
    catalogs = build_catalog(PROVIDERS, upstream_sources(models_dev), strict)
    models: dict[str, dict] = {}
    for model in catalogs[provider]["chat"]:
        models.setdefault(model["id"], model)
    return dict(sorted(models.items()))
