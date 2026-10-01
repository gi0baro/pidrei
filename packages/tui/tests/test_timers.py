"""pidrei-specific: `Timeout`/`Interval`, pi's `setTimeout`/`setInterval`.

A timer is a task parked on its cancel event with the delay as timeout; it
knows nothing about UI state (spec/ui-island.md, "Timers").
"""

import pytest
import tonio.colored as tonio

from pidrei_tui._timers import Interval, Timeout


@pytest.mark.tonio
async def test_a_timeout_fires_once_after_its_delay():
    fired = tonio.Event()
    calls: list[int] = []

    def fn() -> None:
        calls.append(1)
        fired.set()

    Timeout(1, fn)
    await fired.wait(5)
    assert calls == [1]


@pytest.mark.tonio
async def test_a_cancelled_timeout_never_fires():
    calls: list[int] = []
    later = tonio.Event()
    timer = Timeout(20, lambda: calls.append(1))
    timer.cancel()
    # A later timer fires only after the cancelled one's delay has passed.
    Timeout(40, later.set)
    await later.wait(5)
    assert later.is_set()
    assert calls == []


@pytest.mark.tonio
async def test_an_interval_ticks_until_cancelled():
    ticks: list[int] = []
    third = tonio.Event()
    after = tonio.Event()
    interval: Interval | None = None

    def tick() -> None:
        ticks.append(1)
        if len(ticks) == 3:
            interval.cancel()
            third.set()

    interval = Interval(1, tick)
    await third.wait(5)
    assert third.is_set()
    Timeout(20, after.set)
    await after.wait(5)
    assert ticks == [1, 1, 1]


@pytest.mark.tonio
async def test_an_error_stops_an_interval_and_reaches_on_error():
    errors: list[Exception] = []
    reported = tonio.Event()
    after = tonio.Event()
    ticks: list[int] = []

    def tick() -> None:
        ticks.append(1)
        raise RuntimeError("tick failed")

    def on_error(error: Exception) -> None:
        errors.append(error)
        reported.set()

    Interval(1, tick, on_error)
    await reported.wait(5)
    Timeout(20, after.set)
    await after.wait(5)
    assert [str(error) for error in errors] == ["tick failed"]
    assert ticks == [1]


@pytest.mark.tonio
async def test_without_on_error_an_error_only_ends_the_timer():
    ticks: list[int] = []
    after = tonio.Event()

    def tick() -> None:
        ticks.append(1)
        raise RuntimeError("tick failed")

    Interval(1, tick)
    Timeout(30, after.set)
    await after.wait(5)
    assert ticks == [1]
