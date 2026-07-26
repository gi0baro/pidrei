"""Mirror of pi coding-agent src/core/cache-stats.ts."""

from dataclasses import dataclass
from typing import Any

from pidrei_ai.types import AssistantMessage


# Prompt-cache TTL: idle gaps longer than this are worth mentioning as the
# likely cause of a miss. Anthropic's default cache TTL is 5 minutes.
CACHE_TTL_MS = 5 * 60 * 1000

# Per-turn misses at or below this are cache breakpoint granularity noise.
_NOISE_FLOOR_TOKENS = 1024


@dataclass(slots=True)
class CacheMiss:
    """A counted cache miss on a single assistant message."""

    # Prompt tokens that were in the previous turn's prompt but not read from cache.
    missed_tokens: int
    # Extra dollars paid vs. a full cache hit; 0 when pricing is unknown.
    missed_cost: float
    # Milliseconds since the previous request (which last refreshed the cache).
    idle_ms: float
    # True when the model changed relative to the previous request.
    model_changed: bool


@dataclass(slots=True)
class CacheWasteTotals:
    missed_tokens: int = 0
    missed_cost: float = 0.0
    # Number of counted misses (turns above the noise floor).
    miss_count: int = 0


@dataclass(slots=True)
class _PreviousRequest:
    """The last request seen by the scan; everything in its prompt should be cached."""

    prompt_tokens: int
    model_key: str
    timestamp: int
    # Sticky: some earlier request in this scan segment reported cache activity.
    # Distinguishes a total miss on a cache-read-only provider (OpenAI-style,
    # writes unreported) from a provider that never reports caching at all.
    reported_cache: bool


def _detect_miss(prev: _PreviousRequest | None, message: AssistantMessage, models: Any) -> CacheMiss | None:
    """Compute the cache miss for one assistant message relative to the previous
    request. Returns None when nothing is counted: first turn, after a reset, no
    cache activity ever reported (provider without cache support), or miss below
    the noise floor."""
    usage = message.usage
    prompt_tokens = usage.input + usage.cache_read + usage.cache_write
    # A zero-cache turn only counts when cache activity was reported before:
    # on cache-read-only providers that is a total miss, while on providers
    # that never report caching it means nothing.
    if not prev or prompt_tokens <= 0 or (usage.cache_read + usage.cache_write == 0 and not prev.reported_cache):
        return None

    missed_tokens = min(prev.prompt_tokens, prompt_tokens) - usage.cache_read
    if missed_tokens <= _NOISE_FLOOR_TOKENS:
        return None

    # Extra cost = missed tokens billed at the actual paid rate (input/cacheWrite,
    # incl. write premium) instead of the cache-read rate. Missed tokens can only
    # land in the input or cacheWrite buckets, so the paid rate comes straight
    # from this message's own cost breakdown.
    paid_tokens = usage.input + usage.cache_write
    paid_per_token = (usage.cost.input + usage.cost.cache_write) / paid_tokens if paid_tokens > 0 else 0
    if usage.cache_read > 0:
        read_per_token = usage.cost.cache_read / usage.cache_read
    else:
        model = models.get_model(message.provider, message.model)
        read_per_token = (model.cost.cache_read if model is not None else 0) / 1_000_000

    return CacheMiss(
        missed_tokens=missed_tokens,
        missed_cost=missed_tokens * max(0, paid_per_token - read_per_token),
        idle_ms=max(0, message.timestamp - prev.timestamp),
        model_changed=f"{message.provider}/{message.model}" != prev.model_key,
    )


def _as_previous_request(message: AssistantMessage, reported_cache: bool) -> _PreviousRequest | None:
    usage = message.usage
    prompt_tokens = usage.input + usage.cache_read + usage.cache_write
    if prompt_tokens <= 0:
        return None
    return _PreviousRequest(
        prompt_tokens=prompt_tokens,
        model_key=f"{message.provider}/{message.model}",
        timestamp=message.timestamp,
        reported_cache=reported_cache or usage.cache_read + usage.cache_write > 0,
    )


def _scan(
    entries: list[dict[str, Any]], models: Any
) -> tuple[_PreviousRequest | None, CacheWasteTotals, dict[int, CacheMiss]]:
    prev: _PreviousRequest | None = None
    totals = CacheWasteTotals()
    misses: dict[int, CacheMiss] = {}

    for entry in entries:
        if entry.get("type") in ("compaction", "branch_summary"):
            # The context legitimately changed; the next turn's prompt is new content,
            # not re-billed content. Model switches are NOT exempt: they re-bill the
            # full prompt and should be counted.
            prev = None
            continue
        message = entry.get("message")
        if entry.get("type") == "message" and getattr(message, "role", None) == "assistant":
            miss = _detect_miss(prev, message, models)
            if miss:
                totals.missed_tokens += miss.missed_tokens
                totals.missed_cost += miss.missed_cost
                totals.miss_count += 1
                misses[id(message)] = miss
            next_prev = _as_previous_request(message, prev.reported_cache if prev else False)
            prev = next_prev if next_prev is not None else prev
    return prev, totals, misses


def compute_cache_waste(entries: list[dict[str, Any]], models: Any) -> CacheWasteTotals:
    """Cumulative cache waste across a session: prompt tokens that should have been
    cache reads (they were in the previous turn's prompt) but were re-billed."""
    return _scan(entries, models)[1]


def collect_cache_misses(entries: list[dict[str, Any]], models: Any) -> dict[int, CacheMiss]:
    """All counted cache misses across a session, keyed by id() of the assistant
    message that paid for them (pi keys a Map by object reference). Used to
    re-derive transcript notices when rebuilding the chat from entries."""
    return _scan(entries, models)[2]


def detect_cache_miss(entries: list[dict[str, Any]], message: AssistantMessage, models: Any) -> CacheMiss | None:
    """Detect a cache miss on a just-completed assistant message.
    `entries` must not yet contain `message` (message_end fires before persistence)."""
    return _detect_miss(_scan(entries, models)[0], message, models)
