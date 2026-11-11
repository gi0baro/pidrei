"""Mirror of pi packages/ai/src/cache-warming.ts.

Prompt cache warming policy shared by pidrei's hosts: when a refresh is due,
whether a request can be replayed, and whether a refresh pays for itself.
Hosts own the scheduling (`pidrei.core.cache_warmer`); this module owns the
numbers.
"""

import dataclasses
from dataclasses import dataclass
from typing import Literal

from pidrei_ai.builders import UsageBuilder
from pidrei_ai.registry import calculate_cost
from pidrei_ai.types import Model, SimpleStreamOptions
from pidrei_ai.utils.provider_env import get_provider_env_value
from pidrei_utils.cancel import CancelToken


# Whether a refresh is sent ("warm") or warming ends until the next real request ("stop").
type CacheWarmingAction = Literal["warm", "stop"]

# "streaming" while the agent run that sent the request is still active; "idle" after it settled.
type CacheWarmingPhase = Literal["streaming", "idle"]


@dataclass(slots=True)
class CacheWarmingDecision:
    """Inputs and outcome of one warm-or-stop decision."""

    phase: CacheWarmingPhase
    # Price of this refresh: a cache read of the prompt plus one output token.
    warm_cost: float
    # Extra price of the next real request if the cache entry is lost.
    miss_cost: float
    # Estimated chance that a real request arrives before the entry expires.
    continuation_probability: float
    # `continuation_probability * miss_cost - warm_cost`.
    expected_savings: float
    # False when the prompt size or the model's prices are unknown.
    economics_available: bool
    # "warm" when `expected_savings` is at least `CACHE_WARMING_MINIMUM_EXPECTED_SAVINGS`.
    action: CacheWarmingAction


# Streaming warming never continues past this long after the real request that started it.
CACHE_WARMING_MAX_AGE_MS = 60 * 60_000

# Idle warming uses a shorter horizon because continuation estimates become less reliable with age.
CACHE_WARMING_MAX_IDLE_AGE_MS = 30 * 60_000

# A refresh is sent only when it is expected to save at least this many dollars.
CACHE_WARMING_MINIMUM_EXPECTED_SAVINGS = 0.05

# Chance that a real request arrives before the cache entry expires while the agent sits idle.
# Measured from pi's own usage; per-session estimates were not better than this constant.
CACHE_WARMING_IDLE_CONTINUATION_PROBABILITY = 0.15


def get_cache_warming_delay_ms(ttl_ms: float) -> int | None:
    """Refresh at 90% of the TTL while preserving at least ten seconds of margin."""
    if ttl_ms <= 10_000:
        return None
    return max(1, int(min(ttl_ms * 0.9, ttl_ms - 10_000)))


def get_prompt_cache_ttl_ms(model: Model, options: SimpleStreamOptions | None) -> float | None:
    """Lifetime of the prompt cache entry a request writes, from the model's
    `prompt_cache` tier for the retention the request used. None when the model
    has no lifetime for that tier or caching is off."""
    retention = options.cache_retention if options is not None else None
    if retention is None:
        env = options.env if options is not None else None
        retention = "long" if get_provider_env_value("PIDREI_CACHE_RETENTION", env) == "long" else "short"
    if retention == "none":
        return None
    seconds = (model.prompt_cache or {}).get(retention)
    return None if seconds is None else seconds * 1000


def is_cache_warming_replayable(model: Model, options: SimpleStreamOptions | None) -> bool:
    """Whether replaying the request with a one-token output cap leaves its cache
    entry untouched. Anthropic's budget-based thinking (Claude models without
    adaptive thinking) derives `budget_tokens` from `max_tokens`; the replay
    would get a different budget, which Anthropic keys the message cache on,
    and the model could still think for thousands of tokens."""
    if options is None or not options.reasoning or model.api != "anthropic-messages":
        return True
    return getattr(model.compat, "force_adaptive_thinking", None) is True


def get_cache_warming_request_options[T: SimpleStreamOptions](options: T, cancel: CancelToken) -> T:
    """The options of a refresh: the request's own, capped at one output token, without retries,
    and without replacing the conversation's provider thread state (the refresh repeats the last
    request). `cancel` stands where pi passes the refresh's abort signal."""
    return dataclasses.replace(options, max_tokens=1, disable_continuation=True, max_retries=0, cancel=cancel)


def evaluate_cache_warming(model: Model, prompt_tokens: int, phase: CacheWarmingPhase) -> CacheWarmingDecision:
    """Decide whether refreshing the cache entry of a `prompt_tokens`-token request pays for itself in `phase`."""
    cache_hit_cost = _price(model, cache_read=prompt_tokens)
    cache_miss_cost = (
        _price(model, cache_write=prompt_tokens) if model.cost.cache_write > 0 else _price(model, input=prompt_tokens)
    )
    warm_cost = _price(model, cache_read=prompt_tokens, output=1)
    miss_cost = max(0.0, cache_miss_cost - cache_hit_cost)
    continuation_probability = CACHE_WARMING_IDLE_CONTINUATION_PROBABILITY if phase == "idle" else 1
    economics_available = prompt_tokens > 0 and (cache_hit_cost > 0 or cache_miss_cost > 0)
    expected_savings = continuation_probability * miss_cost - warm_cost
    return CacheWarmingDecision(
        phase=phase,
        warm_cost=warm_cost,
        miss_cost=miss_cost,
        continuation_probability=continuation_probability,
        expected_savings=expected_savings,
        economics_available=economics_available,
        action="warm" if expected_savings >= CACHE_WARMING_MINIMUM_EXPECTED_SAVINGS else "stop",
    )


def _price(model: Model, **tokens: int) -> float:
    return calculate_cost(model, UsageBuilder(**tokens)).total
