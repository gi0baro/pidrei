"""Mirrors pi coding-agent test/cache-stats.test.ts."""

from dataclasses import dataclass

import pytest

from pidrei.core.cache_stats import collect_cache_misses, compute_cache_waste, detect_cache_miss
from pidrei_ai.types import AssistantMessage, Usage, UsageCost


@dataclass(slots=True)
class _PriceModelCost:
    cache_read: float


@dataclass(slots=True)
class _PriceModel:
    cost: _PriceModelCost


class _Models:
    """$/million tokens; used as cache-read price fallback on full-miss turns."""

    def get_model(self, _provider, _model_id):
        return _PriceModel(cost=_PriceModelCost(cache_read=0.3))


models = _Models()


def assistant(
    *,
    input: int = 0,
    cache_read: int = 0,
    cache_write: int = 0,
    cost: dict | None = None,
    model: str = "test-model",
    timestamp: int = 0,
) -> AssistantMessage:
    return AssistantMessage(
        content=[],
        api="anthropic-messages",
        provider="test",
        model=model,
        usage=Usage(
            input=input,
            output=10,
            cache_read=cache_read,
            cache_write=cache_write,
            total_tokens=0,
            cost=UsageCost(**(cost or {})),
        ),
        stop_reason="stop",
        timestamp=timestamp,
    )


def entry(message: AssistantMessage) -> dict:
    return {"type": "message", "id": "x", "parentId": None, "timestamp": "", "message": message}


def _turn1() -> AssistantMessage:
    # Turn 1: fresh 100k cache write at $3.75/M
    return assistant(cache_write=100_000, cost={"cache_write": 0.375}, timestamp=0)


def _turn2() -> AssistantMessage:
    # Turn 2: healthy, everything read back at $0.30/M
    return assistant(
        cache_read=100_000, cache_write=5_000, cost={"cache_read": 0.03, "cache_write": 0.019}, timestamp=60_000
    )


class TestComputeCacheWaste:
    def test_accumulates_missed_tokens_and_cost_across_turns(self):
        # Turn 3: full miss, previous 105k prompt re-billed at $3.75/M write
        turn3 = assistant(cache_write=110_000, cost={"cache_write": 0.4125}, timestamp=120_000)
        totals = compute_cache_waste([entry(_turn1()), entry(_turn2()), entry(turn3)], models)
        assert totals.missed_tokens == 105_000
        # 105k at ($3.75 - $0.30)/M
        assert totals.missed_cost == pytest.approx(0.36225, abs=1e-5)

    def test_counts_nothing_for_healthy_sessions(self):
        totals = compute_cache_waste([entry(_turn1()), entry(_turn2())], models)
        assert totals.missed_tokens == 0
        assert totals.missed_cost == 0

    def test_skips_the_turn_after_a_compaction_reset(self):
        reset = {"type": "compaction", "id": "c", "parentId": None, "timestamp": ""}
        after_reset = assistant(cache_write=20_000, cost={"cache_write": 0.075})
        totals = compute_cache_waste([entry(_turn1()), reset, entry(after_reset)], models)
        assert totals.missed_tokens == 0

    def test_counts_misses_caused_by_model_switches(self):
        other_model = assistant(cache_write=100_000, cost={"cache_write": 0.375}, model="other-model")
        totals = compute_cache_waste([entry(_turn1()), entry(other_model)], models)
        assert totals.missed_tokens == 100_000
        assert totals.miss_count == 1

    def test_skips_providers_that_report_no_cache_activity(self):
        a = assistant(input=100_000)
        b = assistant(input=110_000)
        totals = compute_cache_waste([entry(a), entry(b)], models)
        assert totals.missed_tokens == 0


class TestCollectCacheMisses:
    def test_maps_counted_misses_to_their_assistant_messages_by_reference(self):
        miss_turn = assistant(cache_write=110_000, cost={"cache_write": 0.4125}, timestamp=120_000)
        misses = collect_cache_misses([entry(_turn1()), entry(_turn2()), entry(miss_turn)], models)
        assert len(misses) == 1
        assert misses[id(miss_turn)].missed_tokens == 105_000


class TestDetectCacheMiss:
    def test_detects_miss_on_just_completed_message_with_idle_time(self):
        miss_message = assistant(cache_write=110_000, cost={"cache_write": 0.4125}, timestamp=600_000)
        miss = detect_cache_miss([entry(_turn1()), entry(_turn2())], miss_message, models)
        assert miss is not None
        assert miss.missed_tokens == 105_000
        assert miss.missed_cost == pytest.approx(0.36225, abs=1e-5)
        # 600s - 60s since the previous request
        assert miss.idle_ms == 540_000
        assert miss.model_changed is False

    def test_flags_model_switches_on_detected_misses(self):
        other_model = assistant(
            cache_write=110_000, cost={"cache_write": 0.4125}, model="other-model", timestamp=120_000
        )
        miss = detect_cache_miss([entry(_turn1()), entry(_turn2())], other_model, models)
        assert miss.missed_tokens == 105_000
        assert miss.model_changed is True

    def test_returns_none_for_healthy_turns(self):
        healthy = assistant(
            cache_read=105_000,
            cache_write=2_000,
            cost={"cache_read": 0.0315, "cache_write": 0.0075},
            timestamp=120_000,
        )
        assert detect_cache_miss([entry(_turn1()), entry(_turn2())], healthy, models) is None

    def test_returns_none_for_first_turn_of_session(self):
        assert detect_cache_miss([], _turn1(), models) is None
