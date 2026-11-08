"""models.dev v2 catalog (pi: catalog/sources/models-dev.ts).

The schema is published as experimental, so everything the catalog reads goes
through `normalize_models_dev_model()` below.
https://github.com/anomalyco/models.dev/blob/dev/packages/core/src/schema-v2.ts
"""

from dataclasses import dataclass, field
from typing import Any


MODELS_DEV_URL = "https://models.dev/experimental/v2.0/api.json"
# v2 has no provider-agnostic decision metadata endpoint; v1 still serves it.
MODELS_DEV_DECISIONS_URL = "https://models.dev/models.json?type=decision"


@dataclass(frozen=True, slots=True)
class ModelsDevCostTier:
    """A context-size tier. Unlisted rates fall back to the base rates."""

    above: float
    input: float | None = None
    output: float | None = None
    cache_read: float | None = None
    cache_write: float | None = None


@dataclass(frozen=True, slots=True)
class ModelsDevCost:
    """Prices as models.dev lists them. Missing prices stay None."""

    input: float | None = None
    output: float | None = None
    cache_read: float | None = None
    cache_write: float | None = None
    tiers: tuple[ModelsDevCostTier, ...] | None = None


@dataclass(frozen=True, slots=True)
class ModelsDevApi:
    """One API a model is served on, in the order models.dev lists them."""

    protocol: str
    base_url: str
    aisdk: str | None = None


@dataclass(frozen=True, slots=True)
class ModelsDevModel:
    """A models.dev model normalized for the catalog."""

    id: str
    name: str
    type: str
    status: str | None
    tool_call: bool
    structured_output: bool
    reasoning: bool
    # None when the model does not reason. Options keep models.dev's shape:
    # {"type": "toggle"}, {"type": "effort", "values": [...]}, {"type": "budget_tokens", ...}.
    reasoning_options: list[dict[str, Any]] | None
    image_input: bool
    text_input: bool
    text_output: bool
    context_window: int | None
    max_tokens: int | None
    cost: ModelsDevCost
    apis: tuple[ModelsDevApi, ...]
    # The full v2 entry, for fields not normalized above.
    source: dict[str, Any] = field(repr=False)


def _first_price(write: Any) -> float | None:
    if isinstance(write, dict):
        return next(iter(write.values()), None)
    return write


# models.dev lists one cache write price per TTL; the first is the host's default.
def _cache_write_price(cache: dict[str, Any] | None) -> float | None:
    cache = cache or {}
    price = _first_price(cache.get("write"))
    return price if price is not None else _first_price((cache.get("explicit") or {}).get("write"))


def _context_threshold(when: dict[str, Any]) -> float | None:
    if list(when) != ["context"]:
        return None
    context = when["context"] or {}
    size = context.get("gt") if context.get("gt") is not None else context.get("gte")
    return size if isinstance(size, int | float) and not isinstance(size, bool) else None


def _per_token(unit: str | None) -> bool:
    return unit is None or unit == "token"


def _normalize_cost(cost: dict[str, Any] | None) -> ModelsDevCost:
    text = (cost or {}).get("text") or {}
    input_cost = text.get("input")
    output_cost = text.get("output")
    if (
        input_cost is None
        or output_cost is None
        or not _per_token(input_cost.get("unit"))
        or not _per_token(output_cost.get("unit"))
    ):
        return ModelsDevCost()

    # Only context-size overrides map onto PiDrei's cost tiers. Region, service-tier,
    # and time-based prices have no PiDrei equivalent yet.
    tiers: dict[float, dict[str, Any]] = {}
    for override in input_cost.get("overrides") or []:
        above = _context_threshold(override.get("when") or {})
        if above is None:
            continue
        entry = tiers.setdefault(above, {})
        if override.get("price") is not None:
            entry["input"] = override["price"]
        override_cache = override.get("cache") or {}
        if override_cache.get("read") is not None:
            entry["cache_read"] = override_cache["read"]
        cache_write = _cache_write_price(override_cache)
        if cache_write is not None:
            entry["cache_write"] = cache_write
    for override in output_cost.get("overrides") or []:
        above = _context_threshold(override.get("when") or {})
        if above is None:
            continue
        if override.get("price") is not None:
            tiers.setdefault(above, {})["output"] = override["price"]

    input_cache = input_cost.get("cache") or {}
    return ModelsDevCost(
        input=input_cost.get("price"),
        output=output_cost.get("price"),
        cache_read=input_cache.get("read"),
        cache_write=_cache_write_price(input_cache),
        tiers=tuple(ModelsDevCostTier(above=above, **rates) for above, rates in tiers.items()) if tiers else None,
    )


def _reasoning_options(reasoning: dict[str, Any]) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    if reasoning.get("toggle"):
        options.append({"type": "toggle"})
    if "effort" in reasoning:
        options.append({"type": "effort", "values": reasoning["effort"]})
    if "budget" in reasoning:
        options.append({"type": "budget_tokens", **(reasoning["budget"] or {})})
    return options


def normalize_models_dev_model(model: dict[str, Any]) -> ModelsDevModel:
    modalities = model.get("modalities") or {}
    model_input = modalities.get("input") or []
    capabilities = model.get("capabilities") or {}
    reasoning = capabilities.get("reasoning") or {}
    limit = model.get("limit") or {}
    return ModelsDevModel(
        id=model["id"],
        name=model.get("name"),
        type=model.get("type"),
        status=model.get("status"),
        tool_call=(capabilities.get("tools") or {}).get("supported") is True,
        structured_output=capabilities.get("structured_output") is True,
        reasoning=reasoning.get("supported") is True,
        reasoning_options=_reasoning_options(reasoning) if reasoning.get("supported") else None,
        image_input=any(modality == "image" or modality.startswith("image/") for modality in model_input),
        text_input="text" in model_input,
        text_output="text" in (modalities.get("output") or []),
        context_window=limit.get("context"),
        max_tokens=limit.get("output"),
        cost=_normalize_cost(model.get("cost")),
        apis=tuple(
            ModelsDevApi(protocol=protocol, base_url=entry.get("base_url"), aisdk=(entry.get("sdk") or {}).get("aisdk"))
            for protocol, entry in (model.get("api") or {}).items()
        ),
        source=model,
    )


def round_cost(value: float) -> float:
    return float(f"{value:.6f}")


def models_dev_cost(cost: ModelsDevCost) -> dict[str, Any]:
    """Converts models.dev prices to PiDrei's cost shape. Missing rates are free."""
    base = {
        "input": cost.input or 0,
        "output": cost.output or 0,
        "cacheRead": cost.cache_read or 0,
        "cacheWrite": cost.cache_write or 0,
    }
    tiers = [
        {
            "inputTokensAbove": tier.above,
            "input": base["input"] if tier.input is None else tier.input,
            "output": base["output"] if tier.output is None else tier.output,
            "cacheRead": base["cacheRead"] if tier.cache_read is None else tier.cache_read,
            "cacheWrite": base["cacheWrite"] if tier.cache_write is None else tier.cache_write,
        }
        for tier in cost.tiers or ()
    ]
    return {**base, "tiers": tiers} if tiers else base


def models_dev_chat_models(catalog: dict[str, Any], provider: str) -> list[ModelsDevModel]:
    """Chat models of one models.dev provider. Specialized types (image, speech,
    realtime, decision, ...) only appear in v2 and are left out here."""
    models = ((catalog.get(provider) or {}).get("models") or {}).values()
    # Unlike PiDrei models, models.dev v2 always states the type.
    return [normalize_models_dev_model(model) for model in models if model.get("type") == "chat"]
