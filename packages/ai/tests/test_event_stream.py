"""`EventStream` tests; mirrors pi's event-stream.test.ts (0.86.0, #9055).

pi's buffered-drain and end-with-result cases are
`test_push_after_completion_is_ignored` / `test_end_with_result_resolves_result`;
its "order after draining starts" case is mirrored below. Deviation: pi's
queue-plus-waiters stream serves several concurrent iterators ("delivers events
to waiting consumers in registration order", "wakes all waiting consumers when
ended without a result"); pidrei's stream is a single-consumer tonio channel —
every producer has exactly one consumer — so those two cases are not mirrored.
"""

import pytest
import tonio.colored as tonio

from pidrei_ai.utils.event_stream import EventStream


def make_stream() -> EventStream[dict, object]:
    return EventStream(lambda event: event["type"] == "done", lambda event: event["value"])


@pytest.mark.tonio
async def test_yields_pushed_events_in_order_and_resolves_result():
    stream = make_stream()
    events = [{"type": "delta", "i": i, "value": None} for i in range(10)]
    events.append({"type": "done", "value": "final"})
    for event in events:
        stream.push(event)

    received = [event async for event in stream]

    assert received == events
    assert await stream.result() == "final"


@pytest.mark.tonio
async def test_concurrent_producer_consumer():
    stream = make_stream()

    async def produce():
        for i in range(100):
            await tonio.yield_now()
            stream.push({"type": "delta", "i": i, "value": None})
        stream.push({"type": "done", "value": 100})

    handle = tonio.spawn(produce())
    received = [event async for event in stream]
    await handle

    assert [event["i"] for event in received[:-1]] == list(range(100))
    assert received[-1]["type"] == "done"
    assert await stream.result() == 100


@pytest.mark.tonio
async def test_push_after_completion_is_ignored():
    stream = make_stream()
    stream.push({"type": "done", "value": 1})
    stream.push({"type": "delta", "i": 0, "value": None})
    stream.push({"type": "done", "value": 2})

    received = [event async for event in stream]

    assert received == [{"type": "done", "value": 1}]
    assert await stream.result() == 1


@pytest.mark.tonio
async def test_preserves_order_when_events_arrive_after_buffered_draining_starts():
    stream: EventStream[int, int] = EventStream(lambda _event: False, lambda event: event)
    stream.push(1)
    stream.push(2)

    iterator = aiter(stream)
    assert await anext(iterator) == 1

    stream.push(3)
    assert await anext(iterator) == 2
    assert await anext(iterator) == 3

    stream.end(3)
    with pytest.raises(StopAsyncIteration):
        await anext(iterator)


@pytest.mark.tonio
async def test_end_terminates_iteration():
    stream = make_stream()
    stream.push({"type": "delta", "i": 0, "value": None})
    stream.end()
    stream.push({"type": "delta", "i": 1, "value": None})

    received = [event async for event in stream]

    assert received == [{"type": "delta", "i": 0, "value": None}]


@pytest.mark.tonio
async def test_end_with_result_resolves_result():
    stream = make_stream()
    stream.end("ended")

    assert [event async for event in stream] == []
    assert await stream.result() == "ended"


@pytest.mark.tonio
async def test_end_does_not_override_completion_result():
    stream = make_stream()
    stream.push({"type": "done", "value": "first"})
    stream.end("second")

    assert await stream.result() == "first"


@pytest.mark.tonio
async def test_result_awaited_before_completion():
    stream = make_stream()

    async def wait_result():
        return await stream.result()

    handle = tonio.spawn(wait_result())
    await tonio.yield_now()
    stream.push({"type": "done", "value": 42})

    assert await handle == 42


@pytest.mark.tonio
async def test_fail_does_not_override_a_settled_result():
    stream = make_stream()
    stream.push({"type": "done", "value": "final"})
    stream.fail(RuntimeError("late"))

    assert await stream.result() == "final"


@pytest.mark.tonio
async def test_cancel_unwinds_a_parked_producer_and_terminates_the_stream():
    from pidrei_ai.builders import AssistantMessageBuilder, TextContentBuilder, UsageBuilder
    from pidrei_ai.types import AssistantMessage
    from pidrei_ai.utils.event_stream import AssistantMessageEventStream
    from pidrei_utils.cancel import CancelToken

    stream = AssistantMessageEventStream()
    cancel = CancelToken()
    parked = tonio.Event()
    partial = AssistantMessageBuilder(
        content=[TextContentBuilder(text="so far")],
        api="a",
        provider="p",
        model="m",
        usage=UsageBuilder(),
        stop_reason="pending",
        timestamp=0,
    )

    producing = tonio.Event()

    async def produce():
        stream.partial = partial
        producing.set()
        await parked.wait(None)  # never set: only cancellation gets us out

    stream.spawn_producer(produce(), cancel)
    await producing.wait(5)
    assert producing.is_set()
    cancel.cancel()

    events = [event async for event in stream]
    result = await stream.result()
    assert [event.type for event in events] == ["error"]
    assert events[0].reason == "aborted"
    # The seam publishes a frozen snapshot of the producer-private builder
    # (spec/concurrency.md, the data plane): value equality, not identity.
    assert isinstance(result, AssistantMessage)
    assert result.stop_reason == "aborted"
    assert result.error_message == "Request was aborted"
    assert result.content[0].text == "so far"


@pytest.mark.tonio
async def test_cancel_before_the_producer_registers_a_partial_fails_the_result():
    from pidrei_ai.utils.event_stream import AssistantMessageEventStream
    from pidrei_utils.cancel import AbortError, CancelToken

    stream = AssistantMessageEventStream()
    cancel = CancelToken()
    parked = tonio.Event()
    producing = tonio.Event()

    async def produce():
        producing.set()
        await parked.wait(None)

    stream.spawn_producer(produce(), cancel)
    await producing.wait(5)
    assert producing.is_set()
    cancel.cancel()

    assert [event async for event in stream] == []
    with pytest.raises(AbortError):
        await stream.result()


def _seam_builder():
    from pidrei_ai.builders import AssistantMessageBuilder, TextContentBuilder, ToolCallBuilder, UsageBuilder

    return AssistantMessageBuilder(
        content=[TextContentBuilder(text=""), ToolCallBuilder(id="call_1", name="read", arguments={})],
        api="a",
        provider="p",
        model="m",
        usage=UsageBuilder(),
        stop_reason="pending",
        timestamp=0,
    )


@pytest.mark.tonio
async def test_push_publishes_an_independent_frozen_snapshot_per_event():
    # The freeze seam (spec/concurrency.md): every pushed event carries a
    # frozen snapshot of the producer-private builder — later builder mutation
    # must not be visible through an already-published event. Fails on the old
    # shape, where `partial` was the live shared message.
    from pidrei_ai.types import AssistantMessage, TextContent, TextDeltaEvent
    from pidrei_ai.utils.event_stream import AssistantMessageEventStream

    stream = AssistantMessageEventStream()
    builder = _seam_builder()
    stream.partial = builder

    builder.content[0].text = "hel"
    stream.push(TextDeltaEvent(content_index=0, delta="hel", partial=builder))
    builder.content[0].text = "hello"
    builder.usage.output = 5
    stream.push(TextDeltaEvent(content_index=0, delta="lo", partial=builder))
    stream.end()

    events = [event async for event in stream]
    first, second = events
    assert isinstance(first.partial, AssistantMessage)
    assert isinstance(first.partial.content[0], TextContent)
    assert first.partial is not second.partial
    assert first.partial.content[0].text == "hel"
    assert first.partial.usage.output == 0
    assert second.partial.content[0].text == "hello"
    assert second.partial.usage.output == 5


@pytest.mark.tonio
async def test_toolcall_end_carries_the_frozen_twin_of_its_partial_block():
    from pidrei_ai.types import ToolCall, ToolCallEndEvent
    from pidrei_ai.utils.event_stream import AssistantMessageEventStream

    stream = AssistantMessageEventStream()
    builder = _seam_builder()
    builder.content[1].arguments = {"path": "README.md"}
    stream.push(ToolCallEndEvent(content_index=1, tool_call=builder.content[1], partial=builder))
    stream.end()

    (event,) = [event async for event in stream]
    assert isinstance(event.tool_call, ToolCall)
    assert event.tool_call is event.partial.content[1]
    assert event.tool_call.arguments == {"path": "README.md"}


@pytest.mark.tonio
async def test_seam_passes_already_frozen_messages_through_untouched():
    # Extension-authored producers push constructed frozen messages; the seam
    # must not copy or reject them.
    from pidrei_ai.types import AssistantMessage, DoneEvent, StartEvent, TextContent, Usage
    from pidrei_ai.utils.event_stream import AssistantMessageEventStream

    message = AssistantMessage(
        content=[TextContent(text="done")],
        api="a",
        provider="p",
        model="m",
        usage=Usage(),
        stop_reason="stop",
        timestamp=0,
    )
    stream = AssistantMessageEventStream()
    stream.push(StartEvent(partial=message))
    stream.push(DoneEvent(reason="stop", message=message))

    events = [event async for event in stream]
    assert events[0].partial is message
    assert events[1].message is message
    assert await stream.result() is message


@pytest.mark.tonio
async def test_abort_with_a_frozen_partial_publishes_an_aborted_copy():
    # A custom producer may register a constructed frozen message as `partial`;
    # the abort path must publish an aborted copy instead of mutating it.
    from pidrei_ai.types import AssistantMessage, TextContent, Usage
    from pidrei_ai.utils.event_stream import AssistantMessageEventStream
    from pidrei_utils.cancel import CancelToken

    frozen = AssistantMessage(
        content=[TextContent(text="so far")],
        api="a",
        provider="p",
        model="m",
        usage=Usage(),
        stop_reason="pending",
        timestamp=0,
    )
    stream = AssistantMessageEventStream()
    cancel = CancelToken()
    parked = tonio.Event()

    producing = tonio.Event()

    async def produce():
        stream.partial = frozen
        producing.set()
        await parked.wait(None)

    stream.spawn_producer(produce(), cancel)
    await producing.wait(5)
    assert producing.is_set()
    cancel.cancel()

    result = await stream.result()
    assert result is not frozen
    assert result.stop_reason == "aborted"
    assert result.error_message == "Request was aborted"
    assert frozen.stop_reason == "pending"


# --- AssistantMessageEventStream timing (pi #10549) ---------------------------
#
# pi sleeps and asserts on the mutated message; here the clock seams are manual
# (`clock.now_ms` for the wall-clock start, `clock.monotonic` for the duration),
# so the durations are exact, and the frozen message is not mutated: the timed
# value is what the event and `result()` carry (recipe `freeze-at-seam`).


class _ManualClock:
    def __init__(self, monkeypatch) -> None:
        from pidrei_utils import clock

        self.wall_ms = 1_800_000_000_000
        self.monotonic_s = 100.0
        monkeypatch.setattr(clock, "now_ms", lambda: self.wall_ms)
        monkeypatch.setattr(clock, "monotonic", lambda: self.monotonic_s)

    def advance_ms(self, ms: int) -> None:
        self.wall_ms += ms
        self.monotonic_s += ms / 1000


def _timed_message(timestamp: int, duration_ms: int | None = None):
    from pidrei_ai.types import AssistantMessage, Usage

    return AssistantMessage(
        content=[],
        api="openai-responses",
        provider="openai",
        model="m",
        usage=Usage(),
        stop_reason="stop",
        timestamp=timestamp,
        duration_ms=duration_ms,
    )


@pytest.mark.tonio
async def test_sets_duration_ms_on_the_final_done_or_error_message_of_a_response_it_saw_start(monkeypatch):
    from dataclasses import replace

    from pidrei_ai.builders import AssistantMessageBuilder
    from pidrei_ai.types import DoneEvent, ErrorEvent
    from pidrei_ai.utils.event_stream import AssistantMessageEventStream

    clock = _ManualClock(monkeypatch)
    done = AssistantMessageEventStream()
    event = DoneEvent(reason="stop", message=_timed_message(clock.wall_ms))
    clock.advance_ms(20)
    done.push(event)
    assert event.message.duration_ms == 20
    assert (await done.result()).duration_ms == 20

    # A producer's builder gets the field before it is frozen, so both agree.
    failed = AssistantMessageEventStream()
    error = AssistantMessageBuilder.from_message(replace(_timed_message(clock.wall_ms), stop_reason="error"))
    clock.advance_ms(7)
    failed.push(ErrorEvent(reason="error", error=error))
    assert error.duration_ms == 7
    assert (await failed.result()).duration_ms == 7

    ended = AssistantMessageEventStream()
    ended.end(_timed_message(clock.wall_ms))
    assert (await ended.result()).duration_ms == 0


@pytest.mark.tonio
async def test_keeps_an_existing_duration_so_a_forwarding_stream_keeps_the_inner_measurement(monkeypatch):
    from pidrei_ai.types import DoneEvent
    from pidrei_ai.utils.event_stream import AssistantMessageEventStream

    clock = _ManualClock(monkeypatch)
    outer = AssistantMessageEventStream()
    clock.advance_ms(20)
    inner = AssistantMessageEventStream()
    inner.push(DoneEvent(reason="stop", message=_timed_message(clock.wall_ms)))
    measured = await inner.result()
    clock.advance_ms(5)
    # The outer stream forwards what the inner one published (the original is frozen).
    outer.push(DoneEvent(reason="stop", message=measured))
    assert measured.duration_ms == 0
    assert (await outer.result()).duration_ms == measured.duration_ms

    preset_stream = AssistantMessageEventStream()
    preset_stream.push(DoneEvent(reason="stop", message=_timed_message(clock.wall_ms, 1234)))
    assert (await preset_stream.result()).duration_ms == 1234


@pytest.mark.tonio
async def test_leaves_a_message_untimed_when_it_started_before_the_stream_such_as_a_fetched_deferred_result(
    monkeypatch,
):
    from pidrei_ai.types import DoneEvent
    from pidrei_ai.utils.event_stream import AssistantMessageEventStream

    clock = _ManualClock(monkeypatch)
    stream = AssistantMessageEventStream()
    clock.advance_ms(20)
    stream.push(DoneEvent(reason="stop", message=_timed_message(clock.wall_ms - 60_000)))
    assert (await stream.result()).duration_ms is None


@pytest.mark.tonio
async def test_does_not_time_a_message_pushed_after_the_stream_completed(monkeypatch):
    from pidrei_ai.builders import AssistantMessageBuilder
    from pidrei_ai.types import DoneEvent
    from pidrei_ai.utils.event_stream import AssistantMessageEventStream

    clock = _ManualClock(monkeypatch)
    stream = AssistantMessageEventStream()
    stream.push(DoneEvent(reason="stop", message=_timed_message(clock.wall_ms)))
    clock.advance_ms(20)
    # A builder shows whether the late push timed it (a frozen message is never mutated).
    late = AssistantMessageBuilder.from_message(_timed_message(clock.wall_ms))
    stream.push(DoneEvent(reason="stop", message=late))
    assert late.duration_ms is None
