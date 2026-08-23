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
async def test_producer_escape_ends_iteration_and_raises_from_result():
    stream = make_stream()

    async def produce():
        stream.push({"type": "delta", "i": 0, "value": None})
        raise KeyboardInterrupt("producer died")

    stream.spawn_producer(produce())

    received = [event async for event in stream]
    assert received == [{"type": "delta", "i": 0, "value": None}]
    with pytest.raises(KeyboardInterrupt):
        await stream.result()


@pytest.mark.tonio
async def test_fail_does_not_override_a_settled_result():
    stream = make_stream()
    stream.push({"type": "done", "value": "final"})
    stream.fail(RuntimeError("late"))

    assert await stream.result() == "final"


@pytest.mark.tonio
async def test_cancel_unwinds_a_parked_producer_and_terminates_the_stream():
    from pidrei_ai.types import AssistantMessage, TextContent, Usage
    from pidrei_ai.utils.cancel import CancelToken
    from pidrei_ai.utils.event_stream import AssistantMessageEventStream

    stream = AssistantMessageEventStream()
    cancel = CancelToken()
    parked = tonio.Event()
    partial = AssistantMessage(
        content=[TextContent(text="so far")],
        api="a",
        provider="p",
        model="m",
        usage=Usage(),
        stop_reason="pending",
        timestamp=0,
    )

    async def produce():
        stream.partial = partial
        await parked.wait(None)  # never set: only cancellation gets us out

    stream.spawn_producer(produce(), cancel)
    await tonio.sleep(0.02)
    cancel.cancel()

    events = [event async for event in stream]
    result = await stream.result()
    assert [event.type for event in events] == ["error"]
    assert events[0].reason == "aborted"
    assert result is partial
    assert result.stop_reason == "aborted"
    assert result.error_message == "Request was aborted"
    assert result.content[0].text == "so far"


@pytest.mark.tonio
async def test_cancel_before_the_producer_registers_a_partial_fails_the_result():
    from pidrei_ai.utils.cancel import AbortError, CancelToken
    from pidrei_ai.utils.event_stream import AssistantMessageEventStream

    stream = AssistantMessageEventStream()
    cancel = CancelToken()
    parked = tonio.Event()

    async def produce():
        await parked.wait(None)

    stream.spawn_producer(produce(), cancel)
    await tonio.sleep(0.02)
    cancel.cancel()

    assert [event async for event in stream] == []
    with pytest.raises(AbortError):
        await stream.result()


@pytest.mark.tonio
async def test_run_cancellable_unwinds_a_parked_operation():
    from pidrei_ai.utils.abort import run_cancellable
    from pidrei_ai.utils.cancel import AbortError, CancelToken

    cancel = CancelToken()
    parked = tonio.Event()

    async def operation():
        await parked.wait(None)
        return "never"

    async def cancel_soon():
        await tonio.sleep(0.02)
        cancel.cancel()

    tonio.spawn.without_tracking(cancel_soon())
    with pytest.raises(AbortError):
        await run_cancellable(operation(), cancel)


@pytest.mark.tonio
async def test_run_cancellable_returns_the_result_and_raises_its_errors():
    from pidrei_ai.utils.abort import run_cancellable
    from pidrei_ai.utils.cancel import CancelToken

    async def ok():
        return 7

    async def bad():
        raise ValueError("x")

    assert await run_cancellable(ok(), CancelToken()) == 7
    assert await run_cancellable(ok(), None) == 7
    with pytest.raises(ValueError):
        await run_cancellable(bad(), CancelToken())
