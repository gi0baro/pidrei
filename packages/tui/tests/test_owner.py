import pytest
import tonio.colored as tonio

from pidrei_tui._owner import OwnerStopped, OwnerTask


@pytest.mark.tonio
async def test_a_timer_cancelled_by_owner_work_never_fires():
    # The fire is delivered as posted work. If it lands while the owner is
    # busy with work that then cancels the handle, the fire must still be
    # skipped: cancel and fire are ordered on the owner, not raced across
    # tasks (what the detached timers' identity re-checks approximated).
    fired = []
    owner = OwnerTask()
    async with tonio.scope() as scope:
        owner.start(scope)

        async def fire() -> None:
            fired.append(True)

        handle = owner.after(10, fire)

        async def busy_then_cancel() -> None:
            await tonio.sleep(0.05)  # the timer posts its fire meanwhile
            handle.cancel()

        await owner.run(busy_then_cancel)

        async def marker() -> None:
            pass

        await owner.run(marker)  # queued behind the (skipped) fire
        owner.close()
    assert fired == []


@pytest.mark.tonio
async def test_posted_work_runs_serially_in_order():
    log = []
    owner = OwnerTask()
    async with tonio.scope() as scope:
        owner.start(scope)

        def job(name: str):
            async def run() -> None:
                log.append(f"{name}:start")
                await tonio.yield_now()
                log.append(f"{name}:end")

            return run

        for name in ("a", "b", "c"):
            owner.post(job(name))
        await owner.run(job("d"))
        owner.close()
    assert log == [f"{name}:{step}" for name in "abcd" for step in ("start", "end")]


@pytest.mark.tonio
async def test_run_settles_with_owner_stopped_when_the_consumer_crashes():
    # A crashed consumer must never leave a `run()` waiter parked on a queue
    # nobody drains (the exception is held until the scope join, which a
    # parked waiter inside the scope body would never reach — a silent hang).
    owner = OwnerTask()  # on_error=None: the crash kills the consumer

    async def boom() -> None:
        raise RuntimeError("owner died")

    async def anything() -> None:
        pass

    async with tonio.scope() as scope:
        owner.start(scope)
        owner.post(boom)
        with pytest.raises(OwnerStopped) as stopped:
            await owner.run(anything)
        # The waiter gets the real cause; the crash itself is fire-and-forget
        # at the scope level (tonio reports it to stderr, nothing re-raises).
        assert isinstance(stopped.value.__cause__, RuntimeError)
        assert not owner.started


@pytest.mark.tonio
async def test_run_jobs_behind_the_close_sentinel_settle_instead_of_hanging():
    owner = OwnerTask()
    ran = []

    async def marker() -> None:
        ran.append(True)

    async def stall() -> None:
        await tonio.sleep(0.05)  # keep the consumer busy while we close

    async with tonio.scope() as scope:
        owner.start(scope)
        owner.post(stall)
        owner.close()  # sentinel queued behind `stall`
        # `started` is now False, so `run` takes the inline path by contract;
        # reach the queue directly to model a racing sender instead.
        job_done = tonio.Event()
        from pidrei_tui._owner import _Job

        job = _Job(marker, done=job_done)
        owner._pending.add(job)
        owner._sender.send(job)  # lands behind the sentinel
        await job_done.wait(None)
        assert isinstance(job.error, OwnerStopped)
        assert ran == []
