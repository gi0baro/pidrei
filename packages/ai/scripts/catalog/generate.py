"""Model catalog generator (pi: catalog/generate.ts).

Fetches every upstream source, builds each provider's catalog from its
definition in `providers/`, and writes PiDrei's catalog JSON (camelCase keys,
grouped `{api: {"type:id": model}}`, sorted) of chat, image and classifier
models into `pidrei_ai/providers/data/<provider>.json`.

Deviations from pi: the generator is always strict (see `sources/__init__.py`);
it writes only the internal data (PiDrei has no typed wrappers to regenerate,
and pi's `--data-only`, `--json-only`, `--json-output` and `--pretty` flags,
which serve pi.dev's public catalog, are not ported).

Run: make models-data   (network access required)
"""

import json
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


# Run as a script: the `catalog` package and the sibling `model_data` script
# live in packages/ai/scripts, which is not an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tonio.colored as tonio
from model_data import (
    MODEL_DATA_MANIFEST_FILE,
    ModelDataStructure,
    create_model_data_manifest,
    group_provider_model_data,
    validate_generated_model_data,
    validate_model_data_directory,
)
from punkreq.tonio import Client

from catalog.build import ProviderCatalog, build_catalog
from catalog.providers import PROVIDERS
from catalog.sources import load_catalog_sources
from catalog.sources.models_dev import MODELS_DEV_URL


DATA_DIR = Path(__file__).resolve().parents[2] / "pidrei_ai" / "providers" / "data"


def _sorted_by_id(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Key models by id, keeping the first entry per id, sorted by id."""
    by_id: dict[str, dict[str, Any]] = {}
    for model in models:
        by_id.setdefault(model["id"], model)
    return [by_id[model_id] for model_id in sorted(by_id)]


def write_model_data(providers: dict[str, ProviderCatalog]) -> None:
    # Serialize into memory first, so a failed integrity check leaves the committed
    # catalog untouched (pi stages into a temp dir and swaps; same guarantee). The
    # data is grouped by API and keyed by `type:id`.
    file_contents: dict[str, str] = {}
    structure: ModelDataStructure = {}
    for provider_id in sorted(providers):
        catalog = providers[provider_id]
        provider_models = [
            *_sorted_by_id(catalog["chat"]),
            *_sorted_by_id(catalog["image"]),
            *_sorted_by_id(catalog["classifier"]),
        ]
        by_api, structure[provider_id] = group_provider_model_data(provider_id, provider_models)
        file_contents[f"{provider_id}.json"] = json.dumps(by_api, indent=2) + "\n"

    manifest = create_model_data_manifest(
        structure,
        file_contents,
        datetime.now(UTC).isoformat(timespec="seconds"),
        source=MODELS_DEV_URL,
    )

    with tempfile.TemporaryDirectory(prefix="pidrei-model-data-") as staging:
        staging_dir = Path(staging)
        for filename, content in file_contents.items():
            (staging_dir / filename).write_text(content)
        (staging_dir / MODEL_DATA_MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n")
        validate_model_data_directory(structure, staging_dir)

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for stale in DATA_DIR.glob("*.json"):
            if stale.name != MODEL_DATA_MANIFEST_FILE and stale.name not in file_contents:
                stale.unlink()
        for path in staging_dir.glob("*.json"):
            (DATA_DIR / path.name).write_text(path.read_text())
        for filename in sorted(file_contents):
            print(f"Wrote {len(structure[filename.removesuffix('.json')])} models to {DATA_DIR / filename}")
    validate_generated_model_data(DATA_DIR)


@tonio.main
async def main() -> None:
    async with Client() as client:
        sources = await load_catalog_sources(client)
    catalogs = build_catalog(PROVIDERS, sources, strict=True)
    # Providers without any models are left out, as if they did not exist.
    providers = {
        provider_id: catalog
        for provider_id, catalog in catalogs.items()
        if len(catalog["chat"]) + len(catalog["image"]) + len(catalog["classifier"]) > 0
    }

    write_model_data(providers)

    chat_models = [model for catalog in providers.values() for model in catalog["chat"]]
    print("\nModel Statistics:")
    print(f"  Total tool-capable models: {len(chat_models)}")
    print(f"  Reasoning-capable models: {sum(1 for model in chat_models if model['reasoning'])}")
    for provider_id, catalog in providers.items():
        print(
            f"  {provider_id}: {len(catalog['chat'])} chat models, {len(catalog['image'])} image models, "
            f"{len(catalog['classifier'])} classifier models"
        )


if __name__ == "__main__":
    main()
