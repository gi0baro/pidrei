"""Mirror of pi coding-agent test/footer-data-provider.test.ts.

pi mocks child_process (spawnSync for the first read, execFile for the
refresh). Both run on the pool here, through one module seam,
`_resolve_branch_with_git_blocking`, which is patched and call-counted
instead; the refresh tests clear the count after `prime()`. pi's fake-timer watcher-retry test
is mirrored by shortening FS_WATCH_RETRY_DELAY_MS and waiting in real time.

The two debounce tests drive the reftable watcher's listener directly and
swap the module-level `Timeout` for a hand-fired fake (pi 082f7577:
`reftableWatcher.emit` under `vi.useFakeTimers`, because native fs.watch
delivery raced watcher startup); pidrei's polling watcher has the same
timing dependency, so the shape is mirrored rather than the file writes.

Timer callbacks are synchronous and spawn their async work, as pi's `void`
calls do; pi's `advanceTimersByTimeAsync` also runs the promises a fired timer
started. Here the provider method a callback spawns is wrapped on the instance
(`finishes`), and the test waits for it after firing.
"""

from contextlib import contextmanager
from typing import ClassVar

import pytest
import tonio.colored as tonio

from pidrei.core import footer_data_provider as fdp_module
from pidrei.core.footer_data_provider import FooterDataProvider
from pidrei.utils import fs_watch

from .ui_timer_helpers import manual_ui_timers


def _patch(request, module, name, value) -> None:
    # Finalizer-based restore (predates tonio 0.9.14; `monkeypatch` works now).
    original = getattr(module, name)
    setattr(module, name, value)
    request.addfinalizer(lambda: setattr(module, name, original))


@pytest.fixture
def git_mock(request):
    state = {"resolved_branch": "main", "calls": []}

    def fake_git(repo_dir):
        state["calls"].append(repo_dir)
        return state["resolved_branch"] or None

    _patch(request, fdp_module, "_resolve_branch_with_git_blocking", fake_git)
    return state


def _create_plain_reftable_repo(temp_dir):
    repo_dir = temp_dir / "repo"
    (repo_dir / ".git" / "reftable").mkdir(parents=True)
    (repo_dir / ".git" / "HEAD").write_text("ref: refs/heads/.invalid\n")
    return repo_dir


def _create_plain_repo(temp_dir):
    repo_dir = temp_dir / "repo"
    (repo_dir / ".git").mkdir(parents=True)
    (repo_dir / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    return repo_dir


def _create_reftable_worktree(temp_dir):
    repo_dir = temp_dir / "repo"
    common_git_dir = repo_dir / ".git"
    git_dir = common_git_dir / "worktrees" / "src"
    worktree_dir = temp_dir / "worktree"
    reftable_dir = common_git_dir / "reftable"

    git_dir.mkdir(parents=True)
    reftable_dir.mkdir(parents=True)
    worktree_dir.mkdir(parents=True)

    (worktree_dir / ".git").write_text(f"gitdir: {git_dir}\n")
    (git_dir / "HEAD").write_text("ref: refs/heads/.invalid\n")
    (git_dir / "commondir").write_text("../..\n")
    (reftable_dir / "tables.list").write_text("0\n")

    return {"worktreeDir": worktree_dir, "reftableDir": reftable_dir}


async def _wait(event: tonio.Event, timeout_s: float = 3.0) -> None:
    await event.wait(timeout_s)
    assert event.is_set(), "Timed out waiting for the refresh"


class FakeTimeout:
    """Hand-fired stand-in for `_timers.Timeout` (pi: `vi.advanceTimersByTimeAsync`)."""

    instances: ClassVar[list] = []

    def __init__(self, delay_ms: float, fn) -> None:
        self.delay_ms = delay_ms
        self.fn = fn
        self.cancelled = False
        FakeTimeout.instances.append(self)

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        self.fn()


def finishes(provider: FooterDataProvider, method: str) -> tonio.Event:
    """Wrap `provider.<method>` (spawned by a timer callback) on the instance;
    the returned Event is set when a call to it finishes."""
    finished = tonio.Event()
    original = getattr(provider, method)

    async def tracked() -> None:
        try:
            await original()
        finally:
            finished.set()

    setattr(provider, method, tracked)
    return finished


@contextmanager
def fake_debounce_timers():
    original = fdp_module.Timeout
    fdp_module.Timeout = FakeTimeout
    FakeTimeout.instances = []
    try:
        yield FakeTimeout.instances
    finally:
        fdp_module.Timeout = original


def emit_reftable_change(provider: FooterDataProvider) -> None:
    watcher = provider._reftable_watcher
    assert watcher is not None
    watcher._listener("change", "tables.list")


@pytest.mark.tonio
async def test_uses_head_directly_in_a_regular_repo_from_a_nested_directory(tmp_path, git_mock):
    repo_dir = _create_plain_repo(tmp_path)
    nested_dir = repo_dir / "src" / "nested"
    nested_dir.mkdir(parents=True)

    provider = FooterDataProvider(str(nested_dir))
    await provider.prime()
    try:
        assert provider.get_git_branch() == "main"
        assert git_mock["calls"] == []
    finally:
        provider.dispose()


@pytest.mark.tonio
async def test_resolves_the_branch_via_git_when_head_is_invalid_in_a_reftable_repo(tmp_path, git_mock):
    repo_dir = _create_plain_reftable_repo(tmp_path)

    provider = FooterDataProvider(str(repo_dir))
    await provider.prime()
    try:
        assert provider.get_git_branch() == "main"
        assert git_mock["calls"] == [str(repo_dir)]
    finally:
        provider.dispose()


@pytest.mark.tonio
async def test_resolves_the_branch_via_git_in_a_reftable_backed_worktree(tmp_path, git_mock):
    fixture = _create_reftable_worktree(tmp_path)

    provider = FooterDataProvider(str(fixture["worktreeDir"]))
    await provider.prime()
    try:
        assert provider.get_git_branch() == "main"
    finally:
        provider.dispose()


@pytest.mark.tonio
async def test_treats_an_unresolved_invalid_reftable_head_as_detached(tmp_path, git_mock):
    repo_dir = _create_plain_reftable_repo(tmp_path)
    git_mock["resolved_branch"] = ""

    provider = FooterDataProvider(str(repo_dir))
    await provider.prime()
    try:
        assert provider.get_git_branch() == "detached"
    finally:
        provider.dispose()


# Drive debounce behavior explicitly; watcher delivery can race watcher startup.
@pytest.mark.tonio
async def test_does_not_notify_listeners_when_reftable_updates_keep_the_same_branch(tmp_path, git_mock):
    fixture = _create_reftable_worktree(tmp_path)

    with fake_debounce_timers() as timers:
        provider = FooterDataProvider(str(fixture["worktreeDir"]))
        await provider.prime()
        try:
            assert provider.get_git_branch() == "main"
            git_mock["calls"].clear()
            notifications = []
            provider.on_branch_change(lambda: notifications.append(True))

            emit_reftable_change(provider)
            assert len(timers) == 1
            assert timers[0].delay_ms == FooterDataProvider.WATCH_DEBOUNCE_MS
            refreshed = finishes(provider, "_refresh_git_branch_async")
            timers[0].fire()
            await _wait(refreshed)

            assert len(git_mock["calls"]) == 1
            assert provider.get_git_branch() == "main"
            assert notifications == []
        finally:
            provider.dispose()


@pytest.mark.tonio
async def test_debounces_rapid_reftable_updates_into_a_single_async_refresh(tmp_path, git_mock):
    fixture = _create_reftable_worktree(tmp_path)

    with fake_debounce_timers() as timers:
        provider = FooterDataProvider(str(fixture["worktreeDir"]))
        await provider.prime()
        try:
            assert provider.get_git_branch() == "main"
            git_mock["calls"].clear()

            emit_reftable_change(provider)
            emit_reftable_change(provider)
            emit_reftable_change(provider)
            # 499 ms in: the single debounce timer is still pending.
            assert len(timers) == 1
            assert git_mock["calls"] == []
            # 501 ms in: it fires once.
            refreshed = finishes(provider, "_refresh_git_branch_async")
            timers[0].fire()
            await _wait(refreshed)
            assert len(git_mock["calls"]) == 1
            # A further window later nothing else was armed.
            assert len(timers) == 1
            assert len(git_mock["calls"]) == 1
        finally:
            provider.dispose()


@pytest.mark.tonio
async def test_updates_the_cached_branch_when_the_reftable_directory_changes(tmp_path, git_mock):
    fixture = _create_reftable_worktree(tmp_path)

    provider = FooterDataProvider(str(fixture["worktreeDir"]))
    await provider.prime()
    try:
        assert provider.get_git_branch() == "main"
        git_mock["calls"].clear()
        git_mock["resolved_branch"] = "foo"
        notified = tonio.Event()
        provider.on_branch_change(notified.set)

        (fixture["reftableDir"] / "tables.list").write_text("1\n")
        await _wait(notified)

        assert len(git_mock["calls"]) == 1
        assert provider.get_git_branch() == "foo"
    finally:
        provider.dispose()


@pytest.mark.tonio
async def test_retries_git_watchers_after_an_async_fs_watch_error(tmp_path, git_mock):
    # pi advances fake timers across the 5s retry delay; here the retry
    # `Timeout` is recorded by manual UI timers, fired, and the watcher setup
    # it spawns waited for.
    repo_dir = _create_plain_repo(tmp_path)

    provider = FooterDataProvider(str(repo_dir))
    await provider.prime()
    try:
        original_watcher = provider._head_watcher
        assert original_watcher is not None

        with manual_ui_timers() as timers:
            provider._handle_git_watcher_error()
        assert provider._head_watcher is None

        retries = [fn for delay, _handle, fn in timers.scheduled if delay == fs_watch.FS_WATCH_RETRY_DELAY_MS]
        assert len(retries) == 1
        set_up = finishes(provider, "_setup_git_watcher")
        retries[0]()
        await _wait(set_up)
        assert provider._head_watcher is not None
        assert provider._head_watcher is not original_watcher
    finally:
        provider.dispose()


@pytest.mark.tonio
async def test_reports_no_branch_until_primed_without_touching_the_repo(tmp_path, git_mock):
    """pidrei-only: `get_git_branch()` runs from render, so it only reads the
    cache; resolving the branch is `prime()`'s job, off the runtime."""
    repo_dir = _create_plain_reftable_repo(tmp_path)

    provider = FooterDataProvider(str(repo_dir))
    try:
        assert provider.get_git_branch() is None
        assert git_mock["calls"] == []
        await provider.prime()
        assert provider.get_git_branch() == "main"
    finally:
        provider.dispose()
