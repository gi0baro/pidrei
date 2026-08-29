"""Mirror of pi coding-agent test/footer-data-provider.test.ts.

pi mocks child_process (spawnSync for the sync path, execFile for the async
refresh path); here the two module seams `_resolve_branch_with_git_sync` /
`_resolve_branch_with_git_async` are patched and call-counted instead, and
the fakes set Events the tests wait on. pi's fake-timer watcher-retry test
is mirrored by shortening FS_WATCH_RETRY_DELAY_MS and waiting in real time.
"""

import pytest
import tonio.colored as tonio

from pidrei.core import footer_data_provider as fdp_module
from pidrei.core.footer_data_provider import FooterDataProvider
from pidrei.utils import fs_watch


def _patch(request, module, name, value) -> None:
    # Finalizer-based restore (predates tonio 0.9.14; `monkeypatch` works now).
    original = getattr(module, name)
    setattr(module, name, value)
    request.addfinalizer(lambda: setattr(module, name, original))


@pytest.fixture
def git_mock(request):
    state = {"resolved_branch": "main", "sync_calls": [], "async_calls": [], "async_called": tonio.Event()}

    def fake_sync(repo_dir):
        state["sync_calls"].append(repo_dir)
        return state["resolved_branch"] or None

    def fake_async(repo_dir):
        state["async_calls"].append(repo_dir)
        state["async_called"].set()
        return state["resolved_branch"] or None

    _patch(request, fdp_module, "_resolve_branch_with_git_sync", fake_sync)
    _patch(request, fdp_module, "_resolve_branch_with_git_async", fake_async)
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


@pytest.mark.tonio
async def test_uses_head_directly_in_a_regular_repo_from_a_nested_directory(tmp_path, git_mock):
    repo_dir = _create_plain_repo(tmp_path)
    nested_dir = repo_dir / "src" / "nested"
    nested_dir.mkdir(parents=True)

    provider = FooterDataProvider(str(nested_dir))
    await provider.prime()
    try:
        assert provider.get_git_branch() == "main"
        assert git_mock["sync_calls"] == []
    finally:
        provider.dispose()


@pytest.mark.tonio
async def test_resolves_the_branch_via_git_when_head_is_invalid_in_a_reftable_repo(tmp_path, git_mock):
    repo_dir = _create_plain_reftable_repo(tmp_path)

    provider = FooterDataProvider(str(repo_dir))
    await provider.prime()
    try:
        assert provider.get_git_branch() == "main"
        assert git_mock["sync_calls"] == [str(repo_dir)]
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


@pytest.mark.tonio
async def test_does_not_notify_listeners_when_reftable_updates_keep_the_same_branch(tmp_path, git_mock):
    fixture = _create_reftable_worktree(tmp_path)

    provider = FooterDataProvider(str(fixture["worktreeDir"]))
    await provider.prime()
    try:
        assert provider.get_git_branch() == "main"
        git_mock["sync_calls"].clear()
        notifications = []
        provider.on_branch_change(lambda: notifications.append(True))

        (fixture["reftableDir"] / "tables.list").write_text("1\n")
        await _wait(git_mock["async_called"])

        assert len(git_mock["async_calls"]) == 1
        assert git_mock["sync_calls"] == []
        assert provider.get_git_branch() == "main"
        assert notifications == []
    finally:
        provider.dispose()


@pytest.mark.tonio
async def test_debounces_rapid_reftable_updates_into_a_single_async_refresh(tmp_path, git_mock):
    fixture = _create_reftable_worktree(tmp_path)

    provider = FooterDataProvider(str(fixture["worktreeDir"]))
    await provider.prime()
    try:
        assert provider.get_git_branch() == "main"
        git_mock["async_calls"].clear()

        (fixture["reftableDir"] / "tables.list").write_text("1\n")
        (fixture["reftableDir"] / "tables.list").write_text("2\n")
        (fixture["reftableDir"] / "tables.list").write_text("3\n")
        await _wait(git_mock["async_called"])
        # pi advances fake timers past a second debounce window; a second
        # refresh, if one were scheduled, would land within it.
        await tonio.sleep(FooterDataProvider.WATCH_DEBOUNCE_MS / 1000 + 0.15)

        assert len(git_mock["async_calls"]) == 1
    finally:
        provider.dispose()


@pytest.mark.tonio
async def test_updates_the_cached_branch_when_the_reftable_directory_changes(tmp_path, git_mock):
    fixture = _create_reftable_worktree(tmp_path)

    provider = FooterDataProvider(str(fixture["worktreeDir"]))
    await provider.prime()
    try:
        assert provider.get_git_branch() == "main"
        git_mock["resolved_branch"] = "foo"
        notified = tonio.Event()
        provider.on_branch_change(notified.set)

        (fixture["reftableDir"] / "tables.list").write_text("1\n")
        await _wait(notified)

        assert len(git_mock["async_calls"]) == 1
        assert provider.get_git_branch() == "foo"
    finally:
        provider.dispose()


@pytest.mark.tonio
async def test_retries_git_watchers_after_an_async_fs_watch_error(tmp_path, git_mock, request):
    # pi advances fake timers across the 5s retry delay; shorten it and
    # wait in real time instead.
    _patch(request, fs_watch, "FS_WATCH_RETRY_DELAY_MS", 100)
    repo_dir = _create_plain_repo(tmp_path)

    provider = FooterDataProvider(str(repo_dir))
    await provider.prime()
    try:
        original_watcher = provider._head_watcher
        assert original_watcher is not None

        provider._handle_git_watcher_error()
        assert provider._head_watcher is None

        await tonio.sleep(0.2)
        assert provider._head_watcher is not None
        assert provider._head_watcher is not original_watcher
    finally:
        provider.dispose()
