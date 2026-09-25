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


async def _noop() -> None:
    pass


@pytest.mark.tonio
async def test_a_timer_cancelled_by_owner_work_never_fires():
    # The fire is delivered as posted work. If it lands while the owner is
    # busy with work that then cancels the handle, the fire must still be
    # skipped: cancel and fire are ordered on the owner, not raced across
    # tasks (what the detached timers' identity re-checks approximated).
    fired = []
    owner = OwnerTask()
    owner.start()

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
    await owner.run(_noop)  # queued behind the (skipped) fire
    owner.close()
    await owner.join()
    assert fired == []


@pytest.mark.tonio
async def test_posted_work_runs_serially_in_order():
    log = []
    owner = OwnerTask()
    owner.start()

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
    await owner.join()
    assert log == [f"{name}:{step}" for name in "abcd" for step in ("start", "end")]


@pytest.mark.tonio
async def test_run_settles_with_owner_stopped_when_the_consumer_crashes():
    # A crashed consumer must never leave a `run()` waiter parked on a queue
    # nobody drains.
    owner = OwnerTask()  # on_error=None: the crash kills the consumer

    async def boom() -> None:
        raise RuntimeError("owner died")

    owner.start()
    owner.post(boom)
    with pytest.raises(OwnerStopped) as stopped:
        await owner.run(_noop)
    # The waiter gets the real cause; the crash itself is fire-and-forget.
    assert isinstance(stopped.value.__cause__, RuntimeError)
    assert not owner.started


@pytest.mark.tonio
async def test_run_jobs_racing_the_close_settle_instead_of_hanging():
    owner = OwnerTask()
    ran = []

    async def marker() -> None:
        ran.append(True)

    release = tonio.Event()

    async def stall() -> None:
        # Keep the consumer busy while we close and enroll the racing job:
        # were it to drain the queue first, its shutdown sweep would miss the
        # job (a 50ms sleep here only made that unlikely).
        await release.wait(5)

    owner.start()
    owner.post(stall)
    owner.close()  # the queue closed behind `stall`
    # `started` is now False, so `run` takes the inline path by contract;
    # reach the queue directly to model a sender that checked before the close.
    job_done = tonio.Event()
    from pidrei_tui._owner import _Job

    job = _Job(marker, done=job_done)
    owner._state.pending.add(job)
    owner._send(job)  # refused: the queue is closed
    release.set()
    await job_done.wait(5)
    assert job_done.is_set()
    assert isinstance(job.error, OwnerStopped)
    assert ran == []


@pytest.mark.tonio
async def test_a_restart_keeps_the_one_consumer_and_its_queue():
    # A TUI suspend (Ctrl+Z, external editor) or renderer switch restarts its
    # terminal, which calls `start()` again: the owner keeps its consumer and
    # queue, so work posted around the restart runs in order.
    log = []
    owner = OwnerTask()

    def job(name: str):
        async def run() -> None:
            log.append(name)

        return run

    owner.start()
    owner.post(job("before"))
    owner.start()  # the terminal restarting
    owner.post(job("after"))
    await owner.run(_noop)
    owner.close()
    await owner.join()
    assert log == ["before", "after"]
