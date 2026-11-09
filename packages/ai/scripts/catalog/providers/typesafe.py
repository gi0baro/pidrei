"""TypeSafe (pi: catalog/providers/typesafe.ts)."""

from typing import Any

from catalog.define import ProviderContext, ProviderDefinition


def _classifiers(ctx: ProviderContext) -> list[dict[str, Any]]:
    metadata = ctx.sources.models_dev_decisions.get("typesafe/jev-latest")
    if not metadata or metadata.get("type") != "decision":
        if ctx.strict:
            raise RuntimeError("models.dev did not return decision model typesafe/jev-latest")
        return []
    return [
        {
            "type": "classifier",
            "id": "jev-latest",
            "name": metadata.get("name"),
            "api": "typesafe-system-one",
            "provider": "typesafe",
            "baseUrl": "https://api.typesafe.ai/v1/",
            # models.dev lists text input only, but the API accepts image assets as an experimental
            # feature behind an opt-in header, which the typesafe-system-one API sends.
            # https://github.com/typesafe-ai/typesafe-sdk-js (v0.7.0 changelog)
            "input": ["text", "image"],
            # The canonical models.dev entry has no direct-provider pricing. System One reports token usage,
            # so classify() results carry token counts but price them at zero.
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": (metadata.get("limit") or {}).get("context") or 64000,
            "compat": {"imageInput": "assets", "sendImageOptInHeader": True},
        },
    ]


PROVIDER = ProviderDefinition(
    id="typesafe",
    classifiers=_classifiers,
)
