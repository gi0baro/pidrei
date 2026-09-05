"""Mirror of pi agent/test/harness/adaptive-publisher.test.ts."""

import pytest

from pidrei_agent.harness.utils.adaptive_publisher import AdaptivePublisher

from .fake_timers import fake_timers


def _raise(error: Exception) -> None:
    raise error


def test_bounds_event_count_and_spaces_large_publications_by_encoded_size():
    with fake_timers() as timers:
        value = {"current": "a"}
        updates: list[str] = []
        publisher = AdaptivePublisher(
            snapshot=lambda: value["current"],
            update=lambda _previous, current: current,
            measure=len,
            publish=updates.append,
            on_error=_raise,
            min_interval_ms=100,
            target_bytes_per_second=100,
        )

        publisher.mark_dirty()
        value["current"] = "x" * 100
        publisher.mark_dirty()
        timers.advance(100)
        assert updates == ["a", "x" * 100]

        value["current"] = "held"
        publisher.mark_dirty()
        timers.advance(999)
        assert len(updates) == 2
        timers.advance(1)
        assert updates == ["a", "x" * 100, "held"]


def test_commits_its_baseline_before_a_consumer_raises():
    with fake_timers() as timers:
        value = {"current": "a"}
        updates: list[dict] = []
        throw_after_apply = {"enabled": False}

        def publish(update: dict) -> None:
            updates.append(update)
            if throw_after_apply["enabled"]:
                raise Exception("consumer failed after apply")

        publisher = AdaptivePublisher(
            snapshot=lambda: value["current"],
            update=lambda previous, current: {"previous": previous, "current": current},
            measure=lambda _update: 1,
            publish=publish,
            on_error=lambda _error: None,
            min_interval_ms=100,
            target_bytes_per_second=100,
        )

        publisher.mark_dirty()
        timers.advance(100)
        value["current"] = "ab"
        throw_after_apply["enabled"] = True
        with pytest.raises(Exception, match="consumer failed after apply"):
            publisher.mark_dirty()
        publisher.flush(True)
        assert updates == [{"previous": None, "current": "a"}, {"previous": "a", "current": "ab"}]

        throw_after_apply["enabled"] = False
        timers.advance(100)
        value["current"] = "abc"
        publisher.mark_dirty()
        assert updates[-1] == {"previous": "ab", "current": "abc"}
