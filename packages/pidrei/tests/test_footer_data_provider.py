"""Mirror of pi coding-agent test/footer-data-provider.test.ts.

pi mocks child_process (spawnSync for the sync path, execFile for the async
refresh path); here the two module seams `_resolve_branch_with_git_sync` /
`_resolve_branch_with_git_async` are patched and call-counted instead. pi's
fake-timer watcher-retry test is mirrored by shortening
FS_WATCH_RETRY_DELAY_MS and waiting in real time.
"""

import time

import pytest

from pidrei.core import footer_data_provider as fdp_module
from pidrei.core.footer_data_provider import FooterDataProvider
from pidrei.utils import fs_watch


def _wait_for(condition, timeout_s=3.0):
    started_at = time.monotonic()
    while not condition():
        if time.monotonic() - started_at > timeout_s:
            raise TimeoutError("Timed out waiting for condition")
        time.sleep(0.01)


@pytest.fixture
def git_mock(monkeypatch):
    state = {"resolved_branch": "main", "sync_calls": [], "async_calls": []}

    def fake_sync(repo_dir):
        state["sync_calls"].append(repo_dir)
        return state["resolved_branch"] or None

    def fake_async(repo_dir):
        state["async_calls"].append(repo_dir)
        return state["resolved_branch"] or None

    monkeypatch.setattr(fdp_module, "_resolve_branch_with_git_sync", fake_sync)
    monkeypatch.setattr(fdp_module, "_resolve_branch_with_git_async", fake_async)
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


class TestFooterDataProviderReftableBranchDetection:
    def test_uses_head_directly_in_a_regular_repo_from_a_nested_directory(self, tmp_path, git_mock):
        repo_dir = _create_plain_repo(tmp_path)
        nested_dir = repo_dir / "src" / "nested"
        nested_dir.mkdir(parents=True)

        provider = FooterDataProvider(str(nested_dir))
        try:
            assert provider.get_git_branch() == "main"
            assert git_mock["sync_calls"] == []
        finally:
            provider.dispose()

    def test_resolves_the_branch_via_git_when_head_is_invalid_in_a_reftable_repo(self, tmp_path, git_mock):
        repo_dir = _create_plain_reftable_repo(tmp_path)

        provider = FooterDataProvider(str(repo_dir))
        try:
            assert provider.get_git_branch() == "main"
            assert git_mock["sync_calls"] == [str(repo_dir)]
        finally:
            provider.dispose()

    def test_resolves_the_branch_via_git_in_a_reftable_backed_worktree(self, tmp_path, git_mock):
        fixture = _create_reftable_worktree(tmp_path)

        provider = FooterDataProvider(str(fixture["worktreeDir"]))
        try:
            assert provider.get_git_branch() == "main"
        finally:
            provider.dispose()

    def test_treats_an_unresolved_invalid_reftable_head_as_detached(self, tmp_path, git_mock):
        repo_dir = _create_plain_reftable_repo(tmp_path)
        git_mock["resolved_branch"] = ""

        provider = FooterDataProvider(str(repo_dir))
        try:
            assert provider.get_git_branch() == "detached"
        finally:
            provider.dispose()

    def test_does_not_notify_listeners_when_reftable_updates_keep_the_same_branch(self, tmp_path, git_mock):
        fixture = _create_reftable_worktree(tmp_path)

        provider = FooterDataProvider(str(fixture["worktreeDir"]))
        try:
            assert provider.get_git_branch() == "main"
            git_mock["sync_calls"].clear()
            notifications = []
            provider.on_branch_change(lambda: notifications.append(True))

            (fixture["reftableDir"] / "tables.list").write_text("1\n")
            _wait_for(lambda: len(git_mock["async_calls"]) == 1)

            assert len(git_mock["async_calls"]) == 1
            assert git_mock["sync_calls"] == []
            assert provider.get_git_branch() == "main"
            assert notifications == []
        finally:
            provider.dispose()

    def test_debounces_rapid_reftable_updates_into_a_single_async_refresh(self, tmp_path, git_mock):
        fixture = _create_reftable_worktree(tmp_path)

        provider = FooterDataProvider(str(fixture["worktreeDir"]))
        try:
            assert provider.get_git_branch() == "main"
            git_mock["async_calls"].clear()

            (fixture["reftableDir"] / "tables.list").write_text("1\n")
            (fixture["reftableDir"] / "tables.list").write_text("2\n")
            (fixture["reftableDir"] / "tables.list").write_text("3\n")
            _wait_for(lambda: len(git_mock["async_calls"]) == 1)
            time.sleep(0.65)

            assert len(git_mock["async_calls"]) == 1
        finally:
            provider.dispose()

    def test_updates_the_cached_branch_when_the_reftable_directory_changes(self, tmp_path, git_mock):
        fixture = _create_reftable_worktree(tmp_path)

        provider = FooterDataProvider(str(fixture["worktreeDir"]))
        try:
            assert provider.get_git_branch() == "main"
            git_mock["resolved_branch"] = "foo"
            notifications = []
            provider.on_branch_change(lambda: notifications.append(True))

            (fixture["reftableDir"] / "tables.list").write_text("1\n")
            _wait_for(lambda: len(git_mock["async_calls"]) == 1)
            _wait_for(lambda: provider.get_git_branch() == "foo")

            assert len(git_mock["async_calls"]) == 1
            assert provider.get_git_branch() == "foo"
            assert len(notifications) == 1
        finally:
            provider.dispose()

    def test_retries_git_watchers_after_an_async_fs_watch_error(self, tmp_path, git_mock, monkeypatch):
        # pi advances fake timers across the 5s retry delay; shorten it and
        # wait in real time instead.
        monkeypatch.setattr(fs_watch, "FS_WATCH_RETRY_DELAY_MS", 100)
        repo_dir = _create_plain_repo(tmp_path)

        provider = FooterDataProvider(str(repo_dir))
        try:
            original_watcher = provider._head_watcher
            assert original_watcher is not None

            provider._handle_git_watcher_error()
            assert provider._head_watcher is None

            _wait_for(lambda: provider._head_watcher is not None, timeout_s=2.0)
            assert provider._head_watcher is not original_watcher
        finally:
            provider.dispose()
