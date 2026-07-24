"""Port of pi's builtin provider aggregate (packages/ai/src/providers/all.ts).

Provider factories join `builtin_providers()` as their adapters land
(PLAN.md); the catalog reads (`get_builtin_model`/`get_builtin_models`) cover
every provider present in the vendored data regardless.
"""

import json
from datetime import datetime
from pathlib import Path

from pidrei_ai.models_generated import MODELS
from pidrei_ai.providers.anthropic import anthropic_provider
from pidrei_ai.registry import Models, Provider, create_models
from pidrei_ai.types import Model


def get_builtin_model(provider: str, model_id: str) -> Model | None:
    """Read of the generated built-in catalog."""
    for model in MODELS.get(provider, []):
        if model.id == model_id:
            return model
    return None


def get_builtin_models(provider: str) -> list[Model]:
    return list(MODELS.get(provider, []))


def get_builtin_providers() -> list[str]:
    return list(MODELS.keys())


def get_builtin_model_data_generated_at() -> int | None:
    """Generation timestamp (ms) shared by all built-in provider catalogs."""
    manifest_path = Path(__file__).parent / "data" / "_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
        return int(datetime.fromisoformat(manifest["generatedAt"]).timestamp() * 1000)
    except Exception:
        return None


def builtin_providers() -> list[Provider]:
    """All built-in providers, freshly constructed."""
    return [
        anthropic_provider(),
        # openai_provider() joins with the openai-responses adapter (PLAN.md),
        # followed by the rest of pi's ~38 providers with theirs.
    ]


def builtin_models(**options) -> Models:
    """A `Models` collection with every built-in provider registered."""
    models = create_models(**options)
    for provider in builtin_providers():
        models.set_provider(provider)
    return models
