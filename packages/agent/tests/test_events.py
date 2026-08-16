"""Mirror of pi agent/test/harness/events.test.ts.

pi's emit dispatches listener bodies synchronously; pidrei detaches them onto
the runtime, so assertions wait for delivery instead of reading back inline.
"""

import pytest
import tonio.colored as tonio

from pidrei_agent.harness.events import HarnessEventBus, RunEndEvent, RunStartEvent


RUN_START_EVENT = RunStartEvent(lane="main", run_id="run-1")
RUN_END_EVENT = RunEndEvent(lane="main", run_id="run-1", outcome="completed", leaf_id="entry-1")

_SETTLE = 0.05


async def _wait_until(condition, timeout=2.0):
    waited = 0.0
    while not condition():
        await tonio.time.sleep(0.005)
        waited += 0.005
        if waited >= timeout:
            raise AssertionError("condition not reached before timeout")


@pytest.mark.tonio
async def test_delivers_matching_events_to_direct_listeners_and_watchers():
    events = HarnessEventBus()
    direct = []
    watch_events = []

    async def direct_listener(event):
        direct.append(event)

    off = events.on("run_start", direct_listener)
    watch = events.watch(lambda: None)

    async def watch_listener(event):
        watch_events.append(event)

    watch.start(watch_listener)

    events.emit(RUN_START_EVENT)
    events.emit(RUN_END_EVENT)
    off()
    events.emit(RUN_START_EVENT)

    await _wait_until(lambda: len(watch_events) == 3 and len(direct) == 1)
    await tonio.time.sleep(_SETTLE)
    assert direct == [RUN_START_EVENT]
    assert watch_events == [RUN_START_EVENT, RUN_END_EVENT, RUN_START_EVENT]


@pytest.mark.tonio
async def test_captures_a_snapshot_without_an_event_gap_then_flushes_and_delivers_live_events():
    events = HarnessEventBus()
    expected_snapshot = {"leafId": None}
    received = []

    def capture_snapshot():
        snapshot = expected_snapshot
        events.emit(RUN_START_EVENT)
        return snapshot

    watch = events.watch(capture_snapshot)

    assert watch.snapshot is expected_snapshot
    assert received == []

    async def listener(event):
        received.append(event)

    watch.start(listener)
    await _wait_until(lambda: received == [RUN_START_EVENT])

    events.emit(RUN_END_EVENT)
    await _wait_until(lambda: received == [RUN_START_EVENT, RUN_END_EVENT])

    watch.unsubscribe()
    events.emit(RUN_START_EVENT)
    await tonio.time.sleep(_SETTLE)
    assert received == [RUN_START_EVENT, RUN_END_EVENT]
