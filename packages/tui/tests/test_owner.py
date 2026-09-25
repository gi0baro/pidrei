import pytest
import tonio.colored as tonio

from pidrei_tui._owner import OwnerStopped, OwnerTask


class _PostSignal:
    """Wraps an owner's queue sender: `posted` is set once something is
    posted through it (a timer delivers its fire this way)."""

    def __init__(self, owner: OwnerTask) -> None:
        self._owner = owner
        self._sender = owner._sender
        self.posted = tonio.Event()
        owner._sender = self

    def send(self, item) -> None:
        self._sender.send(item)
        self.posted.set()

    def restore(self) -> None:
        self._owner._sender = self._sender


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

        # Never due on its own: the owner job below wakes it, so the fire is
        # posted while the owner is busy — not whenever 10ms of wall clock
        # happened to elapse relative to the job (before it, the fire simply
        # ran; after the cancel, the skip was vacuous).
        handle = owner.after(60_000, fire)

        async def busy_then_cancel() -> None:
            posts = _PostSignal(owner)
            handle._wake.set()  # the timer's deadline, now
            await posts.posted.wait(5)
            assert posts.posted.is_set(), "the timer never posted its fire"
            posts.restore()
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

    release = tonio.Event()

    async def stall() -> None:
        # Keep the consumer busy while we close and enroll the racing job:
        # were it to reach the sentinel first, its shutdown sweep would miss
        # the job (a 50ms sleep here only made that unlikely).
        await release.wait(5)

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
        release.set()
        await job_done.wait(5)
        assert job_done.is_set()
        assert isinstance(job.error, OwnerStopped)
        assert ran == []

    # The queue outlives the stop: after a restart the settled job is not
    # run behind its waiter's back.
    async with tonio.scope() as scope:
        owner.start(scope)
        await owner.run(_noop)
        owner.close()
    assert ran == []


async def _noop() -> None:
    pass


@pytest.mark.tonio
async def test_work_posted_across_a_stop_runs_in_order_after_the_restart():
    # A TUI suspend (Ctrl+Z, external editor) or renderer switch stops and
    # restarts the same owner; an agent event posted in between must not be
    # dropped with the stopped consumer's queue.
    log = []
    owner = OwnerTask()

    def job(name: str):
        async def run() -> None:
            log.append(name)

        return run

    release = tonio.Event()

    async def stall() -> None:
        await release.wait(5)

    async with tonio.scope() as scope:
        owner.start(scope)
        owner.post(stall)
        owner.close()  # sentinel queued behind `stall`
        owner.post(job("behind-sentinel"))
        release.set()
    owner.post(job("while-stopped"))

    async with tonio.scope() as scope:
        owner.start(scope)
        owner.post(job("after-restart"))
        await owner.run(_noop)
        owner.close()
    assert log == ["behind-sentinel", "while-stopped", "after-restart"]


@pytest.mark.tonio
async def test_a_sentinel_left_by_a_cancelled_consumer_does_not_stop_the_next_one():
    # ProcessTerminal.stop closes the owner, then cancels its scope: a
    # consumer unwound before reaching its sentinel leaves it queued.
    owner = OwnerTask()
    parked = tonio.Event()

    async def park() -> None:
        parked.set()
        await tonio.Event().wait(5)

    async with tonio.scope() as scope:
        owner.start(scope)
        owner.post(park)
        await parked.wait(5)
        owner.close()  # sentinel behind the parked job
        scope.cancel()

    ran = []

    async def marker() -> None:
        ran.append(True)

    async with tonio.scope() as scope:
        owner.start(scope)
        await owner.run(marker)
        owner.close()
    assert ran == [True]
