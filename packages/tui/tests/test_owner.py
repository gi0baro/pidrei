import pytest
import tonio.colored as tonio

from pidrei_tui._owner import OwnerTask


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
