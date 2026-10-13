"""Mirror of pi ai src/providers/openrouter-api.ts.

OpenRouter's model listing shapes (`GET /models` and `GET /models/user`) and
the conversion of its per-token prices to a model cost. Shared by the catalog
generator (`scripts/generate_models.py`) and the per-key catalog in
`providers/openrouter.py`. Costs are returned in pi's catalog JSON shape
(camelCase keys); `models_generated.parse_cost` turns one into a `ModelCost`.
"""

import re
from typing import Any, NotRequired, TypedDict


class OpenRouterPricingOverride(TypedDict, total=False):
    """A conditional price. `min_prompt_tokens` selects prompt-length pricing; `utc_*` fields select
    time-of-day or weekday pricing. Missing rates keep the base price."""

    min_prompt_tokens: int
    utc_start: int
    utc_end: int
    utc_days: list[str]
    prompt: str
    completion: str
    input_cache_read: str
    input_cache_write: str


class OpenRouterPricing(TypedDict, total=False):
    """OpenRouter's per-token prices, as decimal strings."""

    prompt: str
    completion: str
    input_cache_read: str
    input_cache_write: str
    overrides: list[OpenRouterPricingOverride]


class OpenRouterModel(TypedDict):
    """The fields pidrei reads from a model in OpenRouter's `GET /models` and `GET /models/user`."""

    id: str
    context_length: NotRequired[int | None]
    top_provider: NotRequired[dict[str, int | None]]
    pricing: NotRequired[OpenRouterPricing]


# JavaScript's `Number.parseFloat`: the longest leading decimal literal, NaN when there is none.
_LEADING_FLOAT_RE = re.compile(r"\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")


def _parse_float(value: str | None) -> float:
    match = _LEADING_FLOAT_RE.match(value or "")
    return float(match.group(1)) if match else float("nan")


def _per_million(value: str | None, fallback: float) -> float:
    """Convert one $/token price to $/1M tokens. 0 means free. Missing, invalid and negative prices use
    the fallback: OpenRouter reports -1 for router models such as `openrouter/auto`, whose price
    depends on the model they pick."""
    per_token = _parse_float(value)
    # NaN compares false, like pi's `perToken >= 0`.
    return float(f"{per_token * 1_000_000:.6f}") if per_token >= 0 else fallback


def openrouter_cost(pricing: OpenRouterPricing | None) -> dict[str, Any]:
    """Convert OpenRouter's $/token prices to pidrei's $/1M tokens. Missing, invalid and negative base
    prices are 0. Prompt-length overrides become request-wide tiers; a rate a tier does not list
    keeps the base price. Time-of-day overrides are skipped because a model cost cannot express them."""
    pricing = pricing or {}
    base = {
        "input": _per_million(pricing.get("prompt"), 0),
        "output": _per_million(pricing.get("completion"), 0),
        "cacheRead": _per_million(pricing.get("input_cache_read"), 0),
        "cacheWrite": _per_million(pricing.get("input_cache_write"), 0),
    }
    # pi tests `=== undefined`: a field present as JSON null still counts as given.
    tiers = [
        {
            "inputTokensAbove": override["min_prompt_tokens"],
            "input": _per_million(override.get("prompt"), base["input"]),
            "output": _per_million(override.get("completion"), base["output"]),
            "cacheRead": _per_million(override.get("input_cache_read"), base["cacheRead"]),
            "cacheWrite": _per_million(override.get("input_cache_write"), base["cacheWrite"]),
        }
        for override in pricing.get("overrides") or []
        if "min_prompt_tokens" in override
        and "utc_start" not in override
        and "utc_end" not in override
        and "utc_days" not in override
    ]
    return {**base, "tiers": tiers} if tiers else base
