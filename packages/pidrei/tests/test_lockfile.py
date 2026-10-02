"""PiDrei-only: `FileLock(renew=True)`, proper-lockfile's holder renewal (no
pi test drives it; pi gets it from proper-lockfile).

The renewal `Interval` is replaced by a fake the test fires by hand, and the
wall clock by a manual one, so the lock's mtime is predictable.
"""

import os

import pytest
import tonio.colored as tonio

from pidrei.utils import lockfile
from pidrei_utils import clock


class _FakeInterval:
    created: list[_FakeInterval]

    def __init__(self, delay_ms, fn, on_error=None):
        self.delay_ms = delay_ms
        self.fn = fn
        self.cancelled = False
        _FakeInterval.created.append(self)

    def cancel(self):
        self.cancelled = True


@pytest.fixture
def renewals(monkeypatch):
    """Fires the lock's renewal timer and waits for the renewal to finish."""
    _FakeInterval.created = []
    monkeypatch.setattr(lockfile, "Interval", _FakeInterval)
    now = [1_000_000_000_000]
    monkeypatch.setattr(clock, "now_ms", lambda: now[0])
    finished = []
    original = lockfile.FileLock._renew_once

    async def renew_once(self):
        try:
            await original(self)
        finally:
            finished[-1].set()

    monkeypatch.setattr(lockfile.FileLock, "_renew_once", renew_once)

    async def fire(advance_ms: int) -> None:
        now[0] += advance_ms
        finished.append(tonio.Event())
        _FakeInterval.created[-1].fn()
        await finished[-1].wait(5)
        assert finished[-1].is_set()

    return fire


def _mtime(path) -> float:
    return os.stat(path).st_mtime


@pytest.mark.tonio
async def test_renews_the_lock_every_half_stale_window(tmp_path, renewals):
    lock = lockfile.FileLock(str(tmp_path / "refresh"), stale=20, renew=True)
    await lock.acquire()
    lock_dir = tmp_path / "refresh.lock"
    (interval,) = _FakeInterval.created
    assert interval.delay_ms == 10_000

    await renewals(10_000)
    assert _mtime(lock_dir) == pytest.approx((1_000_000_000_000 + 10_000) / 1000)
    await renewals(10_000)
    assert _mtime(lock_dir) == pytest.approx((1_000_000_000_000 + 20_000) / 1000)
    assert lock.held

    await lock.release()
    assert interval.cancelled
    assert not lock_dir.exists()


@pytest.mark.tonio
async def test_stops_renewing_and_keeps_a_lock_another_process_took_over(tmp_path, renewals):
    lock = lockfile.FileLock(str(tmp_path / "refresh"), stale=20, renew=True)
    await lock.acquire()
    lock_dir = tmp_path / "refresh.lock"
    (interval,) = _FakeInterval.created

    # Another process found the lock stale and stamped it as its own.
    os.utime(lock_dir, (123.0, 123.0))
    await renewals(10_000)

    assert interval.cancelled
    assert not lock.held
    assert _mtime(lock_dir) == 123.0
    await lock.release()
    assert lock_dir.exists()


@pytest.mark.tonio
async def test_locks_without_renew_arm_no_timer(tmp_path, renewals):
    async with lockfile.FileLock(str(tmp_path / "settings")):
        pass
    assert _FakeInterval.created == []
