"""Vercel AI Gateway catalog and pricing (pi: catalog/sources/vercel-ai-gateway.ts)."""

import math
from typing import Any

from punkreq.tonio import Client

from catalog.sources.models_dev import round_cost


AI_GATEWAY_MODELS_URL = "https://ai-gateway.vercel.sh/v1/models"
AI_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh"
# TypeSafe-compatible System One endpoint for evaluation models.
# https://vercel.com/docs/ai-gateway/sdks-and-apis/typesafe
AI_GATEWAY_TYPESAFE_BASE_URL = "https://ai-gateway.vercel.sh/typesafe/v1"

# {"chat": [...], "classifiers": [...]}
AiGatewayCatalog = dict[str, list[dict[str, Any]]]

# Each rate and the field listing its prompt-length brackets. A bracket's `min` is
# inclusive and its `max` exclusive, both in prompt tokens.
_RATE_FIELDS = (
    ("input", "input_tiers"),
    ("output", "output_tiers"),
    ("cacheRead", "input_cache_read_tiers"),
    ("cacheWrite", "input_cache_write_tiers"),
)


def _to_number(value: Any) -> float:
    """JS `typeof value === "number" ? value : parseFloat(value ?? "0")`, NaN as 0."""
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(value if value is not None else "0")
    except ValueError:
        return 0.0


def _per_million(value: Any) -> float:
    # AI Gateway pricing is in $/token.
    parsed = _to_number(value)
    return round_cost(parsed * 1_000_000) if math.isfinite(parsed) else 0


def get_ai_gateway_cost(pricing: dict[str, Any] | None) -> dict[str, Any]:
    """Convert AI Gateway pricing to $/million tokens. Each rate lists its own prompt-length brackets;
    every bracket start above zero becomes a request-wide tier with the rates in effect there."""
    pricing = pricing or {}
    base = {
        "input": _per_million(pricing.get("input")),
        "output": _per_million(pricing.get("output")),
        "cacheRead": _per_million(pricing.get("input_cache_read")),
        "cacheWrite": _per_million(pricing.get("input_cache_write")),
    }

    starts = {
        bracket["min"]
        for _rate, field in _RATE_FIELDS
        for bracket in pricing.get(field) or []
        if bracket.get("min") is not None and bracket["min"] > 0
    }

    tiers = []
    for start in sorted(starts):
        tier = {"inputTokensAbove": start - 1, **base}
        for rate, field in _RATE_FIELDS:
            bracket = next(
                (
                    candidate
                    for candidate in pricing.get(field) or []
                    if (candidate.get("min") or 0) <= start
                    and ("max" not in candidate or (candidate["max"] is not None and start < candidate["max"]))
                ),
                None,
            )
            if bracket is not None and "cost" in bracket:
                tier[rate] = _per_million(bracket["cost"])
        tiers.append(tier)

    return {**base, "tiers": tiers} if tiers else base


async def fetch_ai_gateway_catalog(client: Client) -> AiGatewayCatalog:
    """Fetch tool-capable chat models and evaluation (classifier) models."""
    response = await client.get(AI_GATEWAY_MODELS_URL)
    if response.status_code != 200:
        raise RuntimeError(f"Vercel AI Gateway API returned {response.status_code}")
    data = await response.json()
    items = data.get("data") if isinstance(data.get("data"), list) else []
    chat: list[dict[str, Any]] = []
    classifiers: list[dict[str, Any]] = []

    for model in items:
        # Evaluation models such as TypeSafe's Jev are served through the
        # TypeSafe-compatible System One endpoint.
        if model.get("type") == "evaluation":
            pricing = model.get("pricing") or {}
            classifiers.append(
                {
                    "type": "classifier",
                    "id": model["id"],
                    "name": model.get("name") or model["id"],
                    "api": "typesafe-system-one",
                    "provider": "vercel-ai-gateway",
                    "baseUrl": AI_GATEWAY_TYPESAFE_BASE_URL,
                    "input": ["text"],
                    "cost": {
                        "input": _per_million(pricing.get("input")),
                        "output": _per_million(pricing.get("output")),
                        "cacheRead": 0,
                        "cacheWrite": 0,
                    },
                    "contextWindow": model.get("context_window") or 4096,
                }
            )
            continue
        tags = model.get("tags") if isinstance(model.get("tags"), list) else []
        # Only include models that support tools
        if "tool-use" not in tags:
            continue

        chat.append(
            {
                "id": model["id"],
                "name": model.get("name") or model["id"],
                "api": "anthropic-messages",
                "baseUrl": AI_GATEWAY_BASE_URL,
                "provider": "vercel-ai-gateway",
                "reasoning": "reasoning" in tags,
                "input": ["text", "image"] if "vision" in tags else ["text"],
                "compat": {"allowEmptySignature": True},
                "cost": get_ai_gateway_cost(model.get("pricing")),
                "contextWindow": model.get("context_window") or 4096,
                "maxTokens": model.get("max_tokens") or 4096,
            }
        )
    return {"chat": chat, "classifiers": classifiers}
