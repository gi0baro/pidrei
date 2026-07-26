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
