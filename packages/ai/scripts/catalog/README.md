# Model catalog

`make models-data` (`uv run python packages/ai/scripts/catalog/generate.py`) fetches upstream catalogs and writes the built-in model catalog to `pidrei_ai/providers/data/<provider>.json`: model values, grouped by API, plus `manifest.json` (committed).

## Layout

| Path | Contents |
| --- | --- |
| `sources/` | Upstream fetchers. `models_dev.py` normalizes the [models.dev v2](https://models.dev/experimental/v2.0/api.json) catalog; OpenRouter and Vercel AI Gateway have their own modules, NVIDIA NIM is fetched in `sources/__init__.py`. All sources are fetched before anything is built, and any failed fetch aborts the run. |
| `providers/<id>.py` | One module per built-in provider: where its models come from, its API and base URL, compat defaults, corrections, and models upstream does not list. Register new modules in `providers/__init__.py`. |
| `families/` | Facts about a model family that hold on every host, such as Claude effort levels or GPT-6 thinking levels. They apply to every provider's models. |
| `define.py` | `ProviderDefinition` and the helpers provider modules use. |
| `build.py` | Runs the provider definitions in a fixed order. |

## Build order

For each provider:

1. `models(ctx)` returns its chat models: upstream entries mapped to PiDrei models, data corrections (limits, prices), and authored models. `ctx.provider_models(id)` returns another provider's models from this step, e.g. Azure derives from OpenAI.
2. `compat` defaults for each API are merged under each model's own compat. Then the shared rules run: models.dev effort levels for APIs that send them natively, and the rules in `families/`.
3. `finalize(model, ctx)` applies provider-specific behavior. It runs after the shared rules, so its values win.
4. Image input limits get the default resize profile, then `finalize_all(models, ctx)` runs cross-model logic such as Anthropic's allowed fallback models.

## Adding a provider

Most providers are a models.dev provider served on one API:

```python
from catalog.define import ProviderDefinition, models_dev_models


def _models(ctx):
    return models_dev_models(ctx, "example", api="openai-completions", base_url="https://api.example.com/v1")


PROVIDER = ProviderDefinition(
    id="example",
    models=_models,
    compat={"openai-completions": {"supportsStore": False, "supportsDeveloperRole": False}},
)
```

`models_dev_models()` keeps tool-capable chat models and maps names, input modalities, costs (including context tiers), and limits. Use its `include` and `map_model` options for provider-specific filtering and fields. Prefer fixing wrong upstream data in models.dev; keep local corrections small and say why they exist.

Models are plain dicts with the catalog's camelCase keys. `build_catalog` can run several times in one process (the tests do), so never put a module-level dict into a model without copying it.
