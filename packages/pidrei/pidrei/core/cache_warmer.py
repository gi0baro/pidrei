"""Mirror of pi coding-agent src/core/cache-warmer.ts.

Keeps one prompt cache entry alive by re-sending its request with a one-token
output cap before the entry expires. The policy (delays, lifetimes,
replayability, the refresh's options, the warm-or-stop economics) is
`pidrei_ai.cache_warming`; this module owns the scheduling.

Runtime mapping (0.87.1 delta port):
- `Date.now()` / `setTimeout` are `clock.now_ms` and a `Timeout`. The timer
  callback spawns the refresh as its own task.
- The per-run `AbortController` is a `CancelToken` carried as the warm
  request's `cancel`; the provider stream owns its work in a scope and unwinds
  when the token fires.
- pi's single thread serializes `start` (agent loop), `on_agent_settled`
  (session run), mode changes and `/session` reads (UI) and the refresh; here
  those run on different tasks, so run/inactive state transitions happen under
  one lock that is never held across an await.
"""

import math
import re
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import tonio.colored as tonio

from pidrei_ai.cache_warming import (
    CACHE_WARMING_MAX_AGE_MS,
    CACHE_WARMING_MAX_IDLE_AGE_MS,
    CACHE_WARMING_MINIMUM_EXPECTED_SAVINGS,
    CacheWarmingAction,
    CacheWarmingDecision,
    CacheWarmingPhase,
    evaluate_cache_warming,
    get_cache_warming_delay_ms,
    get_cache_warming_request_options,
    get_prompt_cache_ttl_ms,
    is_cache_warming_replayable,
)
from pidrei_ai.types import ClientIdentity, Model, SimpleStreamOptions, TranscriptContext
from pidrei_utils import clock
from pidrei_utils.cancel import CancelToken
from pidrei_utils.timers import Timeout

from .settings_manager import CacheWarmingMode


def _last_prompt_tokens(entries: list[dict[str, Any]]) -> int:
    """Prompt size of the most recent real request on the branch, as reported by the provider."""
    for entry in reversed(entries):
        message = entry.get("message")
        if entry.get("type") == "message" and getattr(message, "role", None) == "assistant":
            usage = message.usage
            return usage.input + usage.cache_read + usage.cache_write
    return 0


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
    phase: CacheWarmingPhase = "streaming"
    next_warm_at: int = 0
    # Latest safe time to send this refresh, leaving half the original expiry margin.
    refresh_deadline_at: int = 0
    # Set while a refresh that an extension forced is in flight.
    extension_override: bool = False
    # The armed refresh timer; None while a refresh is running.
    timer: Timeout | None = field(default=None)


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
        get_identity: Callable[[], ClientIdentity],
        decide: Callable[[dict[str, Any]], Awaitable[CacheWarmingAction]] = _default_decide,
    ) -> None:
        # `models` needs `stream_simple`; `session_manager` needs `append_usage` and `get_branch`.
        self._models = models
        self._session_manager = session_manager
        self._get_mode = get_mode
        # Read when each warm request is sent, so an attribution opt-out applies to warming at once.
        # pi replays the identity the warmed request was sent with.
        self._get_identity = get_identity
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
            refreshing = run.timer is None
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
            if not is_cache_warming_replayable(request.model, request.options):
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
            deadline = run.started_at + CACHE_WARMING_MAX_IDLE_AGE_MS
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
        if run.timer is not None:
            run.timer.cancel()
            run.timer = None
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
        deadline = run.started_at + (CACHE_WARMING_MAX_IDLE_AGE_MS if run.phase == "idle" else CACHE_WARMING_MAX_AGE_MS)
        if run.next_warm_at > deadline or now >= deadline:
            self._stop_locked(
                "30-minute idle safety limit reached" if run.phase == "idle" else "one-hour safety limit reached"
            )
            return

        def fire() -> None:
            tonio.spawn.without_tracking(self._refresh(run))

        run.timer = Timeout(max(0, run.next_warm_at - clock.now_ms()), fire)

    async def _refresh(self, run: _ActiveRun) -> None:
        with self._lock:
            run.timer = None
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
            # The identity is resolved again on top of the replayed options (`warm-identity`).
            message = await self._models.stream_simple(
                request.model,
                request.context,
                replace(
                    get_cache_warming_request_options(request.options, run.cancel),
                    identity=self._get_identity(),
                ),
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

    def _evaluate(self, model: Model, phase: CacheWarmingPhase) -> CacheWarmingDecision:
        """`phase` is read by the caller under the lock (`on_agent_settled` writes it)."""
        return evaluate_cache_warming(model, _last_prompt_tokens(self._session_manager.get_branch()), phase)


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


def format_cache_warming_usage(total_cost: float, note: str | None = None) -> str:
    """One-line transcript text for the total cost of one or more consecutive cache warms."""
    note_text = f" ({note})" if note else ""
    cost = re.sub(r"(\.\d{3}\d*?)0+$", r"\1", f"{total_cost:.6f}")
    return f"Cache warmed{note_text}: ${cost}"
