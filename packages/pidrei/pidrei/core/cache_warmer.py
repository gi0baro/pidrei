"""Mirror of pi coding-agent src/core/cache-warmer.ts.

Keeps one prompt cache entry alive by re-sending its request with a one-token
output cap before the entry expires.

Runtime mapping (PORT_0.87.1.md decision 6):
- `Date.now()` / `setTimeout` go through the `clock.now_ms` and
  `timers.set_timeout` seams (module attributes, so tests swap them). The timer
  callback spawns the refresh as its own task.
- The per-run `AbortController` is a `CancelToken` carried as the warm
  request's `cancel`; the provider stream owns its work in a scope and unwinds
  when the token fires.
- pi's single thread serializes `start` (agent loop), `on_agent_settled`
  (session run), mode changes and `/session` reads (UI) and the refresh; here
  those run on different tasks, so run/inactive state transitions happen under
  one lock that is never held across an await.
"""

import dataclasses
import math
import re
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import tonio.colored as tonio

from pidrei_ai.builders import UsageBuilder
from pidrei_ai.registry import calculate_cost
from pidrei_ai.types import Model, SimpleStreamOptions, TranscriptContext
from pidrei_ai.utils import clock, timers
from pidrei_ai.utils.cancel import CancelToken
from pidrei_ai.utils.provider_env import get_provider_env_value

from .settings_manager import CacheWarmingMode


# Streaming warming never continues past this long after the real request that started it.
MAX_WARMING_AGE_MS = 60 * 60_000
# Idle warming uses a shorter horizon because continuation estimates become less reliable with age.
MAX_IDLE_WARMING_AGE_MS = 30 * 60_000
# A refresh is sent only when it is expected to save at least this many dollars.
CACHE_WARMING_MINIMUM_EXPECTED_SAVINGS = 0.05
# Chance that a real request arrives before the cache entry expires while the
# agent sits idle. Measured from pi's own usage; per-session estimates were not
# better than this constant.
IDLE_CONTINUATION_PROBABILITY = 0.15


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


def is_replayable(model: Model, options: SimpleStreamOptions | None) -> bool:
    """Whether replaying the request with a one-token output cap leaves its cache
    entry untouched. Anthropic's budget-based thinking (Claude models without
    adaptive thinking) derives `budget_tokens` from `max_tokens`; the replay
    would get a different budget, which Anthropic keys the message cache on,
    and the model could still think for thousands of tokens."""
    if options is None or not options.reasoning or model.api != "anthropic-messages":
        return True
    return getattr(model.compat, "force_adaptive_thinking", None) is True


def _last_prompt_tokens(entries: list[dict[str, Any]]) -> int:
    """Prompt size of the most recent real request on the branch, as reported by the provider."""
    for entry in reversed(entries):
        message = entry.get("message")
        if entry.get("type") == "message" and getattr(message, "role", None) == "assistant":
            usage = message.usage
            return usage.input + usage.cache_read + usage.cache_write
    return 0


def _price(model: Model, **tokens: int) -> float:
    return calculate_cost(model, UsageBuilder(**tokens)).total


type CacheWarmingAction = Literal["warm", "stop"]


@dataclass(slots=True)
class CacheWarmingDecision:
    """Inputs and outcome of one warm-or-stop decision, as shown by `/session`."""

    # "streaming" while the agent run that sent the request is still active.
    phase: Literal["streaming", "idle"]
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
    # pidrei's decision: "warm" when `expected_savings` is at least $0.05.
    action: CacheWarmingAction


@dataclass(slots=True)
class CacheWarmingStatus:
    # "scheduled": a refresh timer is armed; "refreshing": a warm request is in flight.
    state: Literal["inactive", "scheduled", "refreshing"]
    # Why nothing is scheduled.
    reason: str | None = None
    next_warm_at: int | None = None
    # The pending decision, or the decision that stopped warming.
    decision: CacheWarmingDecision | None = None
    # True when an extension changed `decision.action`.
    extension_override: bool | None = None


@dataclass(slots=True)
class CacheWarmRequest:
    """The request whose prompt cache entry should be kept warm, exactly as it was sent."""

    model: Model
    context: TranscriptContext
    options: SimpleStreamOptions


@dataclass(slots=True, eq=False)
class _ActiveRun:
    request: CacheWarmRequest
    # False once the session's model or messages no longer match the request.
    is_current: Callable[[], bool]
    ttl_ms: float
    delay_ms: int
    started_at: int
    cancel: CancelToken
    phase: Literal["streaming", "idle"] = "streaming"
    next_warm_at: int = 0
    # Latest safe time to send this refresh, leaving half the original expiry margin.
    refresh_deadline_at: int = 0
    # Set while a refresh that an extension forced is in flight.
    extension_override: bool = False
    # Cancels the armed refresh timer; None while a refresh is running.
    cancel_timer: Callable[[], None] | None = field(default=None)


async def _default_decide(event: dict[str, Any]) -> CacheWarmingAction:
    return event["action"]


class CacheWarmer:
    """Keeps one prompt cache entry alive by re-sending its request with a
    one-token output cap before the entry expires. `start` replaces any
    previous run; warm requests never extend the fixed safety windows."""

    def __init__(
        self,
        models: Any,
        session_manager: Any,
        get_mode: Callable[[], CacheWarmingMode],
        decide: Callable[[dict[str, Any]], Awaitable[CacheWarmingAction]] = _default_decide,
    ) -> None:
        # `models` needs `stream_simple`; `session_manager` needs `append_usage` and `get_branch`.
        self._models = models
        self._session_manager = session_manager
        self._get_mode = get_mode
        # Lets extensions override `event["action"]`; failures fall back to the warmer's decision.
        self._decide = decide
        # Called with the persisted usage entry after each successful refresh.
        self.on_warmed: Callable[[dict[str, Any]], None] | None = None
        self._lock = threading.Lock()
        self._run: _ActiveRun | None = None
        self._inactive = CacheWarmingStatus(state="inactive", reason="waiting for first request")

    @property
    def status(self) -> CacheWarmingStatus:
        if self._get_mode() == "off":
            return CacheWarmingStatus(state="inactive", reason="cache warming disabled")
        with self._lock:
            run = self._run
            if run is None:
                return self._inactive
            # The run's fields are written under the lock; read one consistent set.
            phase = run.phase
            refreshing = run.cancel_timer is None
            next_warm_at = run.next_warm_at
            extension_override = run.extension_override
        if not run.is_current():
            return CacheWarmingStatus(state="inactive", reason="conversation context changed")
        decision = self._evaluate(run.request.model, phase)
        if not decision.economics_available and not refreshing:
            return CacheWarmingStatus(state="inactive", reason="cache economics unavailable")
        return CacheWarmingStatus(
            state="refreshing" if refreshing else "scheduled",
            next_warm_at=next_warm_at,
            decision=decision,
            extension_override=extension_override,
        )

    def start(self, request: CacheWarmRequest, is_current: Callable[[], bool]) -> None:
        """Keep the prompt cache entry written by `request` warm while `is_current` holds."""
        with self._lock:
            self._clear_run_locked()
            if self._get_mode() == "off":
                self._stop_locked("cache warming disabled")
                return
            if not is_replayable(request.model, request.options):
                self._stop_locked("request cannot be replayed safely")
                return
            ttl_ms = get_prompt_cache_ttl_ms(request.model, request.options)
            if ttl_ms is None:
                self._stop_locked(
                    "request disabled prompt caching"
                    if request.options.cache_retention == "none"
                    else "cache lifetime unavailable"
                )
                return
            delay_ms = get_cache_warming_delay_ms(ttl_ms)
            if delay_ms is None:
                self._stop_locked("cache lifetime unavailable")
                return
            self._run = _ActiveRun(
                request=request,
                is_current=is_current,
                ttl_ms=ttl_ms,
                delay_ms=delay_ms,
                started_at=clock.now_ms(),
                cancel=CancelToken(),
            )
            self._schedule_locked(self._run)

    def on_agent_settled(self) -> None:
        with self._lock:
            run = self._run
            if run is None:
                return
            if self._get_mode() == "streaming":
                self._stop_locked("agent run settled")
                return
            run.phase = "idle"
            deadline = run.started_at + MAX_IDLE_WARMING_AGE_MS
            if run.next_warm_at > deadline or clock.now_ms() >= deadline:
                self._stop_locked("30-minute idle safety limit reached")

    def on_mode_changed(self) -> None:
        """Reconcile an active run after the persisted warming mode changes."""
        with self._lock:
            run = self._run
            if run is None:
                return
            reason = self._get_mode_stop_reason(run)
            if reason:
                self._stop_locked(reason)

    def cancel(self) -> None:
        with self._lock:
            self._stop_locked("inactive")

    def _clear_run_locked(self) -> None:
        run = self._run
        if run is None:
            return
        self._run = None
        if run.cancel_timer is not None:
            run.cancel_timer()
            run.cancel_timer = None
        run.cancel.cancel()

    def _stop_locked(
        self, reason: str, decision: CacheWarmingDecision | None = None, extension_override: bool | None = None
    ) -> None:
        self._clear_run_locked()
        self._inactive = CacheWarmingStatus(
            state="inactive", reason=reason, decision=decision, extension_override=extension_override
        )

    def _schedule_locked(self, run: _ActiveRun) -> None:
        run.extension_override = False
        now = clock.now_ms()
        run.next_warm_at = now + run.delay_ms
        # A timer can run late after sleep or a stalled runtime. Keep half of
        # the planned pre-expiry margin for that delay and request dispatch; a
        # late refresh is likely a full-price cache write, not a cache warm.
        run.refresh_deadline_at = run.next_warm_at + math.floor((run.ttl_ms - run.delay_ms) / 2)
        deadline = run.started_at + (MAX_IDLE_WARMING_AGE_MS if run.phase == "idle" else MAX_WARMING_AGE_MS)
        if run.next_warm_at > deadline or now >= deadline:
            self._stop_locked(
                "30-minute idle safety limit reached" if run.phase == "idle" else "one-hour safety limit reached"
            )
            return

        def fire() -> None:
            tonio.spawn.without_tracking(self._refresh(run))

        run.cancel_timer = timers.set_timeout(max(0, run.next_warm_at - clock.now_ms()), fire)

    async def _refresh(self, run: _ActiveRun) -> None:
        with self._lock:
            run.cancel_timer = None
            if not self._validate_run_locked(run) or self._refresh_deadline_missed_locked(run):
                return
            phase = run.phase
        decision = self._evaluate(run.request.model, phase)
        action: CacheWarmingAction = decision.action
        try:
            action = await self._decide(
                {
                    "type": "cache_warming_decision",
                    "warmCost": decision.warm_cost,
                    "missCost": decision.miss_cost,
                    "continuationProbability": decision.continuation_probability,
                    "action": action,
                }
            )
        except Exception:
            # Extension failures fall back to the warmer's own decision.
            pass
        with self._lock:
            if not self._validate_run_locked(run) or self._refresh_deadline_missed_locked(run):
                return
            extension_override = action != decision.action
            if action == "stop":
                reason = (
                    "stopped by extension"
                    if extension_override
                    else "expected savings below threshold"
                    if decision.economics_available
                    else "cache economics unavailable"
                )
                self._stop_locked(reason, decision, extension_override)
                return
            run.extension_override = extension_override

        try:
            request = run.request
            message = await self._models.stream_simple(
                request.model,
                request.context,
                dataclasses.replace(request.options, max_tokens=1, max_retries=0, cancel=run.cancel),
            ).result()
            with self._lock:
                valid = self._validate_run_locked(run)
            if valid and message.stop_reason not in ("error", "aborted"):
                entry = await self._session_manager.append_usage(
                    "cache_warm",
                    message.provider,
                    message.response_model if message.response_model is not None else message.model,
                    message.usage,
                    "extension override" if extension_override else None,
                )
                on_warmed = self.on_warmed
                if on_warmed is not None:
                    on_warmed(entry)
            if not valid:
                return
        except Exception:
            # Cache warming is best-effort and must not affect the active agent run.
            pass
        with self._lock:
            if self._run is run:
                self._schedule_locked(run)

    def _refresh_deadline_missed_locked(self, run: _ActiveRun) -> bool:
        if clock.now_ms() <= run.refresh_deadline_at:
            return False
        self._stop_locked("cache refresh deadline missed")
        return True

    def _validate_run_locked(self, run: _ActiveRun) -> bool:
        if self._run is not run:
            return False
        reason = self._get_mode_stop_reason(run) or ("conversation context changed" if not run.is_current() else None)
        if not reason:
            return True
        self._stop_locked(reason)
        return False

    def _get_mode_stop_reason(self, run: _ActiveRun) -> str | None:
        mode = self._get_mode()
        if mode == "off":
            return "cache warming disabled"
        if mode == "streaming" and run.phase == "idle":
            return "agent run settled"
        return None

    def _evaluate(self, model: Model, phase: Literal["streaming", "idle"]) -> CacheWarmingDecision:
        """`phase` is read by the caller under the lock (`on_agent_settled` writes it)."""
        prompt_tokens = _last_prompt_tokens(self._session_manager.get_branch())
        cache_hit_cost = _price(model, cache_read=prompt_tokens)
        cache_miss_cost = (
            _price(model, cache_write=prompt_tokens)
            if model.cost.cache_write > 0
            else _price(model, input=prompt_tokens)
        )
        warm_cost = _price(model, cache_read=prompt_tokens, output=1)
        miss_cost = max(0.0, cache_miss_cost - cache_hit_cost)
        continuation_probability = IDLE_CONTINUATION_PROBABILITY if phase == "idle" else 1
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


def _format_dollars(value: float) -> str:
    return f"-${abs(value):.3f}" if value < 0 else f"${value:.3f}"


def _format_cache_warming_economics(decision: CacheWarmingDecision) -> str:
    if not decision.economics_available:
        return "cache economics unavailable"
    probability = round(decision.continuation_probability * 100)
    probability_text = (
        f"{probability}% continuation probability while agent is running"
        if decision.phase == "streaming"
        else f"{probability}% continuation probability"
    )
    comparison = ">=" if decision.action == "warm" else "<"
    return (
        f"{probability_text}, expected savings {_format_dollars(decision.expected_savings)} "
        f"{comparison} ${CACHE_WARMING_MINIMUM_EXPECTED_SAVINGS:.3f}"
    )


def _format_cache_warming_decision_time(next_warm_at: int | None, now: int) -> str:
    if next_warm_at is None or next_warm_at <= now:
        return "Decision now"
    remaining_seconds = -(-(next_warm_at - now) // 1000)
    hours = remaining_seconds // 3600
    remaining_seconds %= 3600
    minutes = remaining_seconds // 60
    seconds = remaining_seconds % 60
    parts: list[str] = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")
    return f"Decision in {' '.join(parts)}"


def format_cache_warming_status(status: CacheWarmingStatus, now: int | None = None) -> str:
    """One-line status for `/session`."""
    now = now if now is not None else clock.now_ms()
    decision = status.decision
    # A decision is attached once pidrei (or an extension) acted on it; "inactive"
    # without one never got that far.
    if decision is None or (
        status.state == "inactive" and not decision.economics_available and not status.extension_override
    ):
        return f"Inactive ({status.reason if status.reason is not None else 'unknown reason'})"
    details = (
        f"extension override, {_format_cache_warming_economics(decision)}"
        if status.extension_override
        else f"{_format_cache_warming_economics(decision)} -> {decision.action}"
    )
    if status.state == "inactive":
        return f"Stopped ({details})"
    if status.state == "refreshing":
        return f"Warming cache ({details})"
    return f"{_format_cache_warming_decision_time(status.next_warm_at, now)} ({details})"


def format_cache_warming_usage(entry: dict[str, Any]) -> str:
    """One-line transcript text for persisted cache-warming usage."""
    note = f" ({entry['note']})" if entry.get("note") else ""
    cost = re.sub(r"(\.\d{3}\d*?)0+$", r"\1", f"{entry['usage'].cost.total:.6f}")
    return f"Cache warmed{note}: ${cost}"
