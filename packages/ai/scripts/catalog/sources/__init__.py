"""Every upstream source the catalog is built from (pi: catalog/sources/index.ts).

Deviation from pi: pi's `loadCatalogSources` logs a failed source and carries on
with an empty one unless `--strict` is passed, which would silently vendor a
catalog missing a whole provider. Here every fetch failure is fatal: a failed
run beats a quietly truncated artifact. Radius is not a PiDrei provider, so its
source is not fetched.
"""

from dataclasses import dataclass
from typing import Any

from punkreq.tonio import Client

from catalog.sources.models_dev import MODELS_DEV_DECISIONS_URL, MODELS_DEV_URL
from catalog.sources.openrouter import OpenRouterCatalog, fetch_openrouter_catalog
from catalog.sources.vercel_ai_gateway import AiGatewayCatalog, fetch_ai_gateway_catalog


NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


@dataclass(frozen=True, slots=True)
class CatalogSources:
    """Everything fetched from upstream before the catalog is built."""

    # The raw models.dev v2 catalog: {provider: {"models": {id: model}}}.
    models_dev: dict[str, Any]
    models_dev_decisions: dict[str, Any]
    open_router: OpenRouterCatalog
    ai_gateway: AiGatewayCatalog
    # Live NVIDIA NIM ids, keyed by both the raw and the normalized id.
    nvidia_model_ids: dict[str, str]


async def _fetch_json(client: Client, url: str, label: str) -> Any:
    response = await client.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"{label} returned {response.status_code}")
    return await response.json()


def normalize_nvidia_model_id(model_id: str) -> str:
    return model_id.lower().replace("_", ".")


async def _fetch_nvidia_model_ids(client: Client) -> dict[str, str]:
    data = await _fetch_json(client, f"{NVIDIA_BASE_URL}/models", "NVIDIA NIM API")
    model_ids: dict[str, str] = {}
    for model in data.get("data") or []:
        model_ids[model["id"]] = model["id"]
        model_ids[normalize_nvidia_model_id(model["id"])] = model["id"]
    return model_ids


async def load_catalog_sources(client: Client) -> CatalogSources:
    """Fetch every upstream source. Any failure aborts generation."""
    print("Fetching models.dev catalog...")
    models_dev = await _fetch_json(client, MODELS_DEV_URL, "models.dev API")
    print("Fetching models.dev decision models...")
    models_dev_decisions = await _fetch_json(client, MODELS_DEV_DECISIONS_URL, "models.dev decision API")
    nvidia_model_ids: dict[str, str] = {}
    if (models_dev.get("nvidia") or {}).get("models"):
        print("Fetching NVIDIA NIM models...")
        nvidia_model_ids = await _fetch_nvidia_model_ids(client)
    print("Fetching OpenRouter models...")
    open_router = await fetch_openrouter_catalog(client)
    if not open_router["images"]:
        raise RuntimeError("OpenRouter API returned no usable image models")
    print("Fetching Vercel AI Gateway models...")
    ai_gateway = await fetch_ai_gateway_catalog(client)

    return CatalogSources(
        models_dev=models_dev,
        models_dev_decisions=models_dev_decisions,
        open_router=open_router,
        ai_gateway=ai_gateway,
        nvidia_model_ids=nvidia_model_ids,
    )
