"""Mirror of pi's git-update.test.ts: git package updates against real local
repositories (a "remote" directory and a clone under the agent dir).

pi's npm assertions are not mirrored: git packages here install no
dependencies (`package_manager.py`'s docstring). Where pi commits a
`package.json` only to make the update run `npm install`, the commit stays and
carries a `pyproject.toml`. `.ts` file names become `.py`.

Git is hermetic: `GIT_CONFIG_GLOBAL`/`GIT_CONFIG_SYSTEM` point at the null
device for the helpers and for the manager's own git children, and each
repository sets its identity in local config, as pi's helper does. The helpers
run git on the blocking pool (`*_blocking`, through `tonio.spawn_blocking`).
"""

import os
import shutil
import subprocess

import pytest
import tonio.colored as tonio
from tonio.colored import fs

from pidrei.core.package_manager import DefaultPackageManager
from pidrei.core.settings_manager import SettingsManager


# Must use the "git:" prefix so parse_source() treats it as a git source.
GIT_SOURCE = "git:github.com/test/extension"


def _git_blocking(args: list[str], cwd: str) -> str:
    result = subprocess.run(  # noqa: S603 - fixed binary, test-authored args
        ["git", *args],  # noqa: S607 - PATH lookup like pi's spawnSync
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise Exception(f"Command failed: git {' '.join(args)}\n{result.stderr}")
    return result.stdout.strip()


def _init_git_repo_blocking(repo_dir: str) -> None:
    os.makedirs(repo_dir, exist_ok=True)
    _git_blocking(["init", "--initial-branch=main"], repo_dir)
    _git_blocking(["config", "--local", "user.email", "test@test.com"], repo_dir)
    _git_blocking(["config", "--local", "user.name", "Test"], repo_dir)


def _create_commit_blocking(repo_dir: str, filename: str, content: str, message: str) -> str:
    with open(os.path.join(repo_dir, filename), "w", encoding="utf-8") as handle:
        handle.write(content)
    _git_blocking(["add", filename], repo_dir)
    _git_blocking(["commit", "-m", message], repo_dir)
    return _git_blocking(["rev-parse", "HEAD"], repo_dir)


def git(args: list[str], cwd: str):
    return tonio.spawn_blocking(_git_blocking, args, cwd)


def init_git_repo(repo_dir: str):
    return tonio.spawn_blocking(_init_git_repo_blocking, repo_dir)


def create_commit(repo_dir: str, filename: str, content: str, message: str):
    return tonio.spawn_blocking(_create_commit_blocking, repo_dir, filename, content, message)


def get_current_commit(repo_dir: str):
    return git(["rev-parse", "HEAD"], repo_dir)


def get_file_content(repo_dir: str, filename: str):
    return fs.Path(os.path.join(repo_dir, filename)).read_text(encoding="utf-8")


class _Context:
    def __init__(self, root: str, monkeypatch: pytest.MonkeyPatch) -> None:
        self.temp_dir = root
        # Simulates the "remote" repository.
        self.remote_dir = os.path.join(root, "remote")
        # The agent directory where extensions are installed.
        self.agent_dir = os.path.join(root, "agent")
        # Matches the install path structure: agent_dir/git/<host>/<path>.
        self.installed_dir = os.path.join(self.agent_dir, "git", "github.com", "test", "extension")
        os.makedirs(self.agent_dir)
        self.settings = SettingsManager.in_memory()
        self.manager = DefaultPackageManager(cwd=root, agent_dir=self.agent_dir, settings_manager=self.settings)
        self._monkeypatch = monkeypatch

    def _clone_installed_blocking(self) -> None:
        os.makedirs(os.path.dirname(self.installed_dir), exist_ok=True)
        _git_blocking(["clone", self.remote_dir, self.installed_dir], self.temp_dir)

    def clone_installed(self):
        return tonio.spawn_blocking(self._clone_installed_blocking)

    def _setup_remote_and_install_blocking(self) -> None:
        _init_git_repo_blocking(self.remote_dir)
        _create_commit_blocking(self.remote_dir, "extension.py", "# v1", "Initial commit")
        # Clone to the installed directory (simulating what install() does).
        self._clone_installed_blocking()
        _git_blocking(["config", "--local", "user.email", "test@test.com"], self.installed_dir)
        _git_blocking(["config", "--local", "user.name", "Test"], self.installed_dir)

    async def setup_remote_and_install(self) -> None:
        """Sets up a "remote" repository and clones it to the installed directory."""
        await tonio.spawn_blocking(self._setup_remote_and_install_blocking)
        # Add to the global packages so update() processes this source.
        self.settings.set_packages([GIT_SOURCE])

    def record_commands(self, *, fail=None) -> list[str]:
        """Records each git command line the manager runs, then runs it for
        real unless `fail` returns an error to raise instead."""
        commands: list[str] = []
        run = self.manager._run_command

        async def run_command(command: str, args: list[str], **options) -> str:
            command_line = f"{command} {' '.join(args)}"
            commands.append(command_line)
            error = fail(command_line) if fail is not None else None
            if error is not None:
                raise error
            return await run(command, args, **options)

        self._monkeypatch.setattr(self.manager, "_run_command", run_command)
        return commands

    def collect_update_results(self) -> list[str]:
        updated: list[str] = []

        def on_progress(event) -> None:
            if event.type == "result" and event.action == "update" and event.message:
                updated.append(event.message)

        self.manager.set_progress_callback(on_progress)
        return updated

    def collect_progress_messages(self) -> list[str]:
        messages: list[str] = []

        def on_progress(event) -> None:
            if event.type == "progress" and event.message:
                messages.append(event.message)

        self.manager.set_progress_callback(on_progress)
        return messages


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    # pi's allowNetwork(): the update path is the online one.
    monkeypatch.delenv("PIDREI_OFFLINE", raising=False)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    return _Context(os.path.realpath(str(tmp_path)), monkeypatch)


# -- normal updates (no force-push) ------------------------------------------------


@pytest.mark.tonio
async def test_skips_reset_clean_and_install_when_already_up_to_date(ctx):
    await init_git_repo(ctx.remote_dir)
    await create_commit(ctx.remote_dir, "extension.py", "# v1", "Initial commit")
    await ctx.clone_installed()
    ctx.settings.set_packages([GIT_SOURCE])
    commands = ctx.record_commands()
    updated = ctx.collect_update_results()

    await ctx.manager.update()

    assert updated == []
    assert "git fetch --prune --no-tags origin +refs/heads/main:refs/remotes/origin/main" in commands
    assert "git fetch --prune origin" not in commands
    # pi asserts the exact lines `git reset --hard @{upstream}` and `... origin/HEAD` are absent; the
    # reset runs on `<ref>^{commit}`, so this checks for any reset instead.
    assert not any(command.startswith("git reset") for command in commands)
    assert "git clean -fdx" not in commands


@pytest.mark.tonio
async def test_updates_to_latest_commit_when_remote_has_new_commits(ctx):
    await ctx.setup_remote_and_install()
    assert await get_file_content(ctx.installed_dir, "extension.py") == "# v1"
    old_commit = await get_current_commit(ctx.installed_dir)

    # Add a new commit to the remote.
    new_commit = await create_commit(ctx.remote_dir, "extension.py", "# v2", "Second commit")

    # Update through the package manager (no args = uses settings).
    updated = ctx.collect_update_results()
    await ctx.manager.update()

    assert updated == [f"{GIT_SOURCE} {old_commit[:7]} -> {new_commit[:7]}"]
    assert await get_current_commit(ctx.installed_dir) == new_commit
    assert await get_file_content(ctx.installed_dir, "extension.py") == "# v2"


# -- git clean failures ------------------------------------------------------------


# https://github.com/earendil-works/pi/issues/11597
# pi: "should install dependencies and retry the clean later when git clean fails"; the dependency
# install is npm's and not ported, so the case keeps the retry.
@pytest.mark.tonio
async def test_retries_the_clean_later_when_git_clean_fails(ctx):
    await ctx.setup_remote_and_install()
    new_commit = await create_commit(
        ctx.remote_dir,
        "pyproject.toml",
        '[project]\nname = "test-extension"\nversion = "1.0.0"\n',
        "Add pyproject.toml",
    )
    fail_clean = True

    def fail(command_line: str) -> Exception | None:
        if command_line == "git clean -fdx" and fail_clean:
            return Exception("git clean -fdx failed with code 1")
        return None

    commands = ctx.record_commands(fail=fail)
    progress_messages = ctx.collect_progress_messages()
    marker_path = os.path.join(ctx.agent_dir, "git", "github.com", "test", ".extension.pidrei-update-incomplete")

    await ctx.manager.update()

    assert await get_current_commit(ctx.installed_dir) == new_commit
    assert "git clean -fdx" in commands
    assert progress_messages == [
        (
            f"Could not remove all untracked files in {ctx.installed_dir} (git clean -fdx failed with code 1). "
            "Files may be in use by another pidrei session; close it and run the update again."
        )
    ]
    assert await fs.Path(marker_path).exists()

    fail_clean = False
    commands.clear()
    await ctx.manager.update()

    assert "git clean -fdx" in commands
    assert not await fs.Path(marker_path).exists()


# -- force-push scenarios ----------------------------------------------------------


@pytest.mark.tonio
async def test_handles_complete_history_rewrite(ctx):
    await ctx.setup_remote_and_install()

    # The remote gets several commits.
    await create_commit(ctx.remote_dir, "extension.py", "# v2", "v2")
    await create_commit(ctx.remote_dir, "extension.py", "# v3", "v3")

    await ctx.manager.update()
    assert await get_file_content(ctx.installed_dir, "extension.py") == "# v3"

    # The maintainer force-pushes completely different history.
    await git(["reset", "--hard", "HEAD~2"], ctx.remote_dir)
    await create_commit(ctx.remote_dir, "extension.py", "# rewrite-a", "Rewrite A")
    final_commit = await create_commit(ctx.remote_dir, "extension.py", "# rewrite-b", "Rewrite B")

    # Should handle this gracefully.
    await ctx.manager.update()

    assert await get_current_commit(ctx.installed_dir) == final_commit
    assert await get_file_content(ctx.installed_dir, "extension.py") == "# rewrite-b"


# -- pinned sources ----------------------------------------------------------------


@pytest.mark.tonio
async def test_checks_out_the_configured_pinned_git_ref_during_full_and_targeted_updates(ctx):
    await init_git_repo(ctx.remote_dir)
    v1_commit = await create_commit(ctx.remote_dir, "extension.py", "# v1", "Initial commit")
    await git(["tag", "v1"], ctx.remote_dir)
    v2_commit = await create_commit(ctx.remote_dir, "extension.py", "# v2", "Second commit")
    await git(["tag", "v2"], ctx.remote_dir)

    await ctx.clone_installed()
    await git(["checkout", "v1"], ctx.installed_dir)
    assert await get_current_commit(ctx.installed_dir) == v1_commit

    pinned_source = f"{GIT_SOURCE}@v2"
    ctx.settings.set_packages([pinned_source])

    await ctx.manager.update()

    assert await get_current_commit(ctx.installed_dir) == v2_commit
    assert await get_file_content(ctx.installed_dir, "extension.py") == "# v2"

    await git(["checkout", "v1"], ctx.installed_dir)

    await ctx.manager.update(pinned_source)

    assert await get_current_commit(ctx.installed_dir) == v2_commit
    assert await get_file_content(ctx.installed_dir, "extension.py") == "# v2"


# -- temporary git sources ---------------------------------------------------------


def _write_stale_cached_checkout_blocking(cached_dir: str, extension_file: str) -> None:
    shutil.rmtree(cached_dir, ignore_errors=True)
    os.makedirs(os.path.dirname(extension_file))
    with open(os.path.join(cached_dir, "pyproject.toml"), "w", encoding="utf-8") as handle:
        handle.write('[tool.pidrei]\nextensions = ["./pidrei-extensions"]\n')
    with open(extension_file, "w", encoding="utf-8") as handle:
        handle.write("# stale")


@pytest.mark.tonio
async def test_refreshes_cached_temporary_git_sources_when_resolving(ctx, monkeypatch):
    cached_dir = await ctx.manager._get_git_install_path(ctx.manager.parse_source(GIT_SOURCE), "temporary")
    extension_file = os.path.join(cached_dir, "pidrei-extensions", "session_breakdown.py")
    await tonio.spawn_blocking(_write_stale_cached_checkout_blocking, cached_dir, extension_file)

    executed_commands: list[str] = []

    # pi stubs runCommand (recorded) and runCommandCapture (canned reads) apart; both are
    # `_run_command` here, told apart by the bounded reads' `timeout_ms`.
    async def run_command(command: str, args: list[str], *, timeout_ms: int | None = None, **_options) -> str:
        if timeout_ms is None:
            executed_commands.append(f"{command} {' '.join(args)}")
            if command == "git" and args[0] == "reset":
                await fs.Path(extension_file).write_text("# fresh", encoding="utf-8")
            return ""
        if args[0] == "rev-parse" and args[1] == "HEAD":
            return "local-head"
        if args[0] == "rev-parse" and args[1] == "@{upstream}":
            return "remote-head"
        if args[0] == "rev-parse" and args[1] == "--abbrev-ref":
            return "origin/main"
        return ""

    monkeypatch.setattr(ctx.manager, "_run_command", run_command)

    await ctx.manager.resolve_extension_sources([GIT_SOURCE], temporary=True)

    assert "git fetch --prune --no-tags origin +refs/heads/main:refs/remotes/origin/main" in executed_commands
    assert await get_file_content(cached_dir, "pidrei-extensions/session_breakdown.py") == "# fresh"


# -- pidrei-only: the update stays on the remote's branch -----------------------------
#
# Before following pi's update target, an update checked out FETCH_HEAD, which left the checkout
# detached; the next update then fetched every branch and checked out the alphabetically first one.


async def _init_remote_with_alpha_branch(remote_dir: str) -> None:
    """A remote whose default branch `main` sorts after another branch, `alpha`."""
    await init_git_repo(remote_dir)
    await create_commit(remote_dir, "extension.py", "# main v1", "Initial commit")
    await git(["checkout", "-b", "alpha"], remote_dir)
    await create_commit(remote_dir, "extension.py", "# alpha", "Alpha commit")
    await git(["checkout", "main"], remote_dir)


@pytest.mark.tonio
async def test_consecutive_updates_stay_on_the_default_branch_when_another_branch_sorts_first(ctx):
    await _init_remote_with_alpha_branch(ctx.remote_dir)
    await ctx.clone_installed()
    ctx.settings.set_packages([GIT_SOURCE])

    await create_commit(ctx.remote_dir, "extension.py", "# main v2", "Main v2")
    await ctx.manager.update()
    main_v3 = await create_commit(ctx.remote_dir, "extension.py", "# main v3", "Main v3")
    await ctx.manager.update()

    assert await get_file_content(ctx.installed_dir, "extension.py") == "# main v3"
    assert await get_current_commit(ctx.installed_dir) == main_v3
    assert await git(["rev-parse", "--abbrev-ref", "HEAD"], ctx.installed_dir) == "main"


@pytest.mark.tonio
async def test_updates_a_checkout_left_detached_to_the_remote_default_branch(ctx):
    await _init_remote_with_alpha_branch(ctx.remote_dir)
    await ctx.clone_installed()
    # What an update used to leave behind.
    await git(["fetch", "origin", "main"], ctx.installed_dir)
    await git(["checkout", "--force", "FETCH_HEAD"], ctx.installed_dir)
    ctx.settings.set_packages([GIT_SOURCE])

    main_v2 = await create_commit(ctx.remote_dir, "extension.py", "# main v2", "Main v2")
    await ctx.manager.update()

    assert await get_file_content(ctx.installed_dir, "extension.py") == "# main v2"
    assert await get_current_commit(ctx.installed_dir) == main_v2
