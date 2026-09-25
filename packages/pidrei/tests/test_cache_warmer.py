"""Mirror of pi coding-agent test/cache-warmer.test.ts.

pi drives `Date.now()`/`setTimeout` with vitest fake timers and lets
`advanceTimersByTimeAsync` flush the refresh the timer starts. pidrei swaps the
`clock.now_ms` / `timers.set_timeout` seams for the shared `FakeTimers`
(PORT_0.87.1.md decision 6): `advance_timers` takes each due refresh timer off
the queue and awaits the refresh it would have spawned (`_refresh`) instead of
racing a detached task.
"""

import dataclasses
import sys
from pathlib import Path

import pytest
import tonio.colored as tonio

from pidrei.core.auth_storage import AuthStorage
from pidrei.core.cache_warmer import (
    CacheWarmer,
    CacheWarmingDecision,
    CacheWarmingStatus,
    CacheWarmRequest,
    format_cache_warming_status,
    format_cache_warming_usage,
    get_cache_warming_delay_ms,
    get_prompt_cache_ttl_ms,
    is_replayable,
)
from pidrei.core.event_bus import create_event_bus
from pidrei.core.extensions.loader import create_extension_runtime, load_extension_from_factory
from pidrei.core.extensions.runner import ExtensionRunner
from pidrei.core.session_manager import SessionManager
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import AssistantMessage, Context, SimpleStreamOptions, Usage, UsageCost
from pidrei_ai.utils.cancel import CancelToken
from pidrei_ai.utils.transcript import normalize_context

from .model_runtime_helpers import create_in_memory_model_registry


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "agent" / "tests"))
from fake_timers import fake_timers


ADAPTIVE_MODEL = dataclasses.replace(
    get_builtin_model("anthropic", "claude-opus-4-6"), prompt_cache={"short": 300, "long": 3600}
)
BUDGET_MODEL = dataclasses.replace(
    get_builtin_model("anthropic", "claude-sonnet-4-5"), prompt_cache={"short": 300, "long": 3600}
)
OPENAI_MODEL = dataclasses.replace(get_builtin_model("openai", "gpt-5"), prompt_cache={"short": 300, "long": 86_400})
UNKNOWN_MODEL = dataclasses.replace(ADAPTIVE_MODEL, prompt_cache=None)

WARM_USAGE = Usage(
    input=0,
    output=1,
    cache_read=100,
    cache_write=0,
    total_tokens=101,
    cost=UsageCost(input=0, output=0, cache_read=0.01, cache_write=0, total=0.01),
)


def response(model, stop_reason: str = "length") -> AssistantMessage:
    return AssistantMessage(
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=WARM_USAGE,
        stop_reason=stop_reason,
        timestamp=0,
    )


def branch_with_prompt(prompt_tokens: int) -> list[dict]:
    return [
        {
            "type": "message",
            "id": "a",
            "parentId": None,
            "timestamp": "1970-01-01T00:00:00.000Z",
            "message": dataclasses.replace(
                response(ADAPTIVE_MODEL),
                usage=dataclasses.replace(
                    WARM_USAGE, output=10, cache_read=prompt_tokens, total_tokens=prompt_tokens + 10
                ),
            ),
        }
    ]


class _FakeStream:
    def __init__(self, result):
        self._result = result

    async def result(self):
        return await self._result()


class FakeRuntime:
    def __init__(self, *, result=None, decide=None, mode: str = "idle", branch: list[dict] | None = None):
        self.calls: list[tuple] = []
        self.events: list[dict] = []
        self.warmed_entries: list[dict] = []
        self.appended: list[tuple] = []
        self.appended_entries: list[dict] = []
        self.mode = mode
        self.branch = branch if branch is not None else branch_with_prompt(100_000)
        self.stream_called = tonio.Event()
        usage_manager = SessionManager.in_memory()
        runtime = self

        class _Models:
            def stream_simple(self, model, _context, stream_options):
                runtime.calls.append((model, stream_options))
                runtime.stream_called.set()

                async def resolve():
                    return await (result or _default_result)(model)

                return _FakeStream(resolve)

        class _Sessions:
            async def append_usage(self, *args):
                runtime.appended.append(args)
                entry = await usage_manager.append_usage(*args)
                runtime.appended_entries.append(entry)
                return entry

            def get_branch(self):
                return runtime.branch

        async def decide_event(event):
            runtime.events.append(event)
            override = decide(event) if decide is not None else None
            return override if override is not None else event["action"]

        self.warmer = CacheWarmer(_Models(), _Sessions(), lambda: runtime.mode, decide_event)
        self.warmer.on_warmed = self.warmed_entries.append


async def _default_result(model):
    return response(model)


def request(model=ADAPTIVE_MODEL, options: SimpleStreamOptions | None = None) -> CacheWarmRequest:
    return CacheWarmRequest(
        model=model,
        context=normalize_context(Context(messages=[])),
        options=options if options is not None else SimpleStreamOptions(),
    )


def current() -> bool:
    return True


async def advance_timers(fake, warmer: CacheWarmer, ms: int) -> None:
    """`vi.advanceTimersByTimeAsync` for the warmer: each due refresh timer is
    taken off the queue and the refresh it would spawn is awaited."""
    target = fake.now + ms
    while True:
        run = warmer._run
        if fake.pop_due(target) is None:
            break
        if run is not None:
            await warmer._refresh(run)
    fake.now = target


def test_derives_eligibility_and_timing_from_retention_and_provider_behavior():
    assert [
        get_prompt_cache_ttl_ms(ADAPTIVE_MODEL, None),
        get_prompt_cache_ttl_ms(ADAPTIVE_MODEL, SimpleStreamOptions(cache_retention="long")),
        get_prompt_cache_ttl_ms(ADAPTIVE_MODEL, SimpleStreamOptions(cache_retention="none")),
        get_prompt_cache_ttl_ms(ADAPTIVE_MODEL, SimpleStreamOptions(env={"PIDREI_CACHE_RETENTION": "long"})),
        get_prompt_cache_ttl_ms(OPENAI_MODEL, SimpleStreamOptions(cache_retention="long")),
        get_prompt_cache_ttl_ms(UNKNOWN_MODEL, None),
    ] == [300_000, 3_600_000, None, 3_600_000, 86_400_000, None]
    assert [
        get_cache_warming_delay_ms(300_000),
        get_cache_warming_delay_ms(60_000),
        get_cache_warming_delay_ms(10_000),
    ] == [270_000, 50_000, None]
    assert [
        is_replayable(BUDGET_MODEL, SimpleStreamOptions(reasoning="medium")),
        is_replayable(BUDGET_MODEL, None),
        is_replayable(ADAPTIVE_MODEL, SimpleStreamOptions(reasoning="medium")),
        is_replayable(OPENAI_MODEL, SimpleStreamOptions(reasoning="medium")),
    ] == [False, True, True, True]


@pytest.mark.tonio
async def test_replays_profitable_requests_and_preserves_options_across_repeated_refreshes():
    with fake_timers() as fake:
        runtime = FakeRuntime()
        cancel = CancelToken()

        async def transform_headers(_headers):
            return {}

        runtime.warmer.start(
            request(
                ADAPTIVE_MODEL,
                SimpleStreamOptions(
                    reasoning="high", cancel=cancel, session_id="s", transform_headers=transform_headers
                ),
            ),
            current,
        )
        await advance_timers(fake, runtime.warmer, 270_000)

        model, options = runtime.calls[0]
        assert model is ADAPTIVE_MODEL
        assert (options.reasoning, options.session_id, options.transform_headers, options.max_tokens) == (
            "high",
            "s",
            transform_headers,
            1,
        )
        assert options.max_retries == 0
        assert options.cancel is not cancel
        assert runtime.events[0]["type"] == "cache_warming_decision"
        assert runtime.events[0]["continuationProbability"] == 1
        assert runtime.events[0]["action"] == "warm"
        assert runtime.events[0]["missCost"] == pytest.approx(0.575)
        assert runtime.events[0]["warmCost"] == pytest.approx(0.050025)
        assert runtime.appended == [("cache_warm", ADAPTIVE_MODEL.provider, ADAPTIVE_MODEL.id, WARM_USAGE, None)]
        assert runtime.warmed_entries == runtime.appended_entries

        await advance_timers(fake, runtime.warmer, 270_000)
        assert len(runtime.calls) == 2
        runtime.warmer.cancel()


@pytest.mark.tonio
async def test_does_not_issue_refreshes_after_their_safe_deadline():
    with fake_timers() as fake:
        runtime = FakeRuntime()
        runtime.warmer.start(request(), current)

        # A five-minute cache is scheduled for 4m30s and retains 15 seconds of
        # the 30-second expiry margin. Simulate a timer delayed by sleep.
        fake.now = 285_001
        run = runtime.warmer._run
        assert run is not None, "expected an active cache-warming run"
        await runtime.warmer._refresh(run)

        assert runtime.calls == []
        status = runtime.warmer.status
        assert (status.state, status.reason) == ("inactive", "cache refresh deadline missed")


@pytest.mark.tonio
async def test_rechecks_the_deadline_after_an_extension_decision():
    with fake_timers() as fake:

        def decide(_event):
            fake.now = 285_001
            return "warm"

        runtime = FakeRuntime(decide=decide)
        runtime.warmer.start(request(), current)
        run = runtime.warmer._run
        assert run is not None, "expected an active cache-warming run"
        await runtime.warmer._refresh(run)

        assert runtime.calls == []
        status = runtime.warmer.status
        assert (status.state, status.reason) == ("inactive", "cache refresh deadline missed")


@pytest.mark.tonio
async def test_applies_economic_decisions_and_extension_overrides():
    with fake_timers() as fake:
        unprofitable = FakeRuntime(branch=branch_with_prompt(5_000))
        unprofitable.warmer.start(request(), current)
        await advance_timers(fake, unprofitable.warmer, 270_000)
        assert unprofitable.calls == []
        status = unprofitable.warmer.status
        assert (status.state, status.decision.action, status.decision.economics_available) == (
            "inactive",
            "stop",
            True,
        )
        assert status.extension_override is False

        forced = FakeRuntime(branch=branch_with_prompt(5_000), decide=lambda _event: "warm")
        forced.warmer.start(request(), current)
        await advance_timers(fake, forced.warmer, 270_000)
        assert len(forced.calls) == 1
        assert forced.warmed_entries[0]["note"] == "extension override"
        forced.warmer.cancel()

        vetoed = FakeRuntime(decide=lambda _event: "stop")
        vetoed.warmer.start(request(), current)
        await advance_timers(fake, vetoed.warmer, 270_000)
        assert vetoed.calls == []
        status = vetoed.warmer.status
        assert (status.state, status.extension_override) == ("inactive", True)

        unavailable = FakeRuntime(branch=branch_with_prompt(0))
        unavailable.warmer.start(request(), current)
        status = unavailable.warmer.status
        assert (status.state, status.reason) == ("inactive", "cache economics unavailable")
        await advance_timers(fake, unavailable.warmer, 270_000)
        assert unavailable.calls == []


@pytest.mark.tonio
async def test_stops_for_unsupported_requests_context_changes_and_mode_changes():
    with fake_timers() as fake:
        unsupported = FakeRuntime()
        unsupported.mode = "off"
        unsupported.warmer.start(request(), current)
        assert unsupported.warmer.status.reason == "cache warming disabled"
        unsupported.mode = "idle"
        unsupported.warmer.start(request(UNKNOWN_MODEL), current)
        assert unsupported.warmer.status.reason == "cache lifetime unavailable"
        unsupported.warmer.start(request(BUDGET_MODEL, SimpleStreamOptions(reasoning="high")), current)
        assert unsupported.warmer.status.reason == "request cannot be replayed safely"

        still_current = True
        unsupported.warmer.start(request(), lambda: still_current)
        still_current = False
        assert unsupported.warmer.status.reason == "conversation context changed"
        await advance_timers(fake, unsupported.warmer, 270_000)
        assert unsupported.calls == []

        unsupported.warmer.start(request(), current)
        unsupported.mode = "off"
        await advance_timers(fake, unsupported.warmer, 270_000)
        assert unsupported.calls == []

        streaming = FakeRuntime(mode="streaming", branch=branch_with_prompt(400_000))
        streaming.warmer.start(request(), current)
        streaming.warmer.on_agent_settled()
        assert streaming.warmer.status.reason == "agent run settled"


@pytest.mark.tonio
async def test_aborts_replaced_requests_and_does_not_record_failed_refreshes():
    with fake_timers() as fake:
        released = tonio.Event()

        async def pending_result(model):
            await released.wait()
            return response(model)

        pending = FakeRuntime(result=pending_result)
        pending.warmer.start(request(), current)
        run = pending.warmer._run
        assert fake.pop_due(fake.now + 270_000) is not None
        refresh = tonio.spawn(pending.warmer._refresh(run))
        await pending.stream_called.wait(5)
        assert pending.stream_called.is_set()
        pending.warmer.start(request(), current)
        assert pending.calls[0][1].cancel.cancelled is True
        released.set()
        pending.warmer.cancel()
        await refresh
        await advance_timers(fake, pending.warmer, 600_000)
        assert len(pending.calls) == 1

        failed = FakeRuntime(result=lambda model: _error_result(model))
        failed.warmer.start(request(), current)
        await advance_timers(fake, failed.warmer, 270_000)
        assert failed.appended == []
        failed.warmer.cancel()


async def _error_result(model):
    return response(model, "error")


@pytest.mark.tonio
async def test_formats_status_and_usage_entries():
    decision = CacheWarmingDecision(
        phase="idle",
        warm_cost=0.013,
        miss_cost=0.621,
        continuation_probability=0.6,
        expected_savings=0.36,
        economics_available=True,
        action="warm",
    )
    assert format_cache_warming_status(
        CacheWarmingStatus(state="scheduled", next_warm_at=222_000, decision=decision), 0
    ) == ("Decision in 3m 42s (60% continuation probability, expected savings $0.360 >= $0.050 -> warm)")
    usage = dataclasses.replace(
        WARM_USAGE,
        cost=UsageCost(input=0.00004, output=0.00005, cache_read=0.02940725, cache_write=0, total=0.02949725),
    )
    entry = await SessionManager.in_memory().append_usage(
        "cache_warm", ADAPTIVE_MODEL.provider, ADAPTIVE_MODEL.id, usage, "extension override"
    )
    assert format_cache_warming_usage(entry) == "Cache warmed (extension override): $0.029497"


@pytest.mark.tonio
async def test_emit_cache_warming_decision_uses_the_last_extension_override(tmp_path):
    runtime = create_extension_runtime()
    event_bus = create_event_bus()

    async def warm(pi) -> None:
        async def handler(_event, _ctx):
            return {"action": "warm"}

        pi.on("cache_warming_decision", handler)

    async def stop(pi) -> None:
        async def handler(_event, _ctx):
            return {"action": "stop"}

        pi.on("cache_warming_decision", handler)

    extensions = [
        await load_extension_from_factory(factory, str(tmp_path), event_bus, runtime) for factory in (warm, stop)
    ]
    model_registry = await create_in_memory_model_registry(AuthStorage.in_memory())
    runner = ExtensionRunner(extensions, runtime, str(tmp_path), SessionManager.in_memory(), model_registry)
    event = {
        "type": "cache_warming_decision",
        "warmCost": 0.05,
        "missCost": 0.5,
        "continuationProbability": 0.15,
        "action": "warm",
    }

    assert await runner.emit_cache_warming_decision(event) == "stop"
