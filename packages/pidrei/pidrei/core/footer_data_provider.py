"""Mirror of pi coding-agent src/core/footer-data-provider.ts.

Parallelism deltas vs single-threaded pi: watcher callbacks and refresh
timers fire on polling threads, so provider state is guarded by an RLock.
pi's async git refresh (execFile) becomes a plain subprocess call executed on
the debounce timer thread; the sync/async split is kept as two module-level
seams so mirrored tests can patch and count them separately.
"""

import os
import re
import subprocess
import sys
import threading

import tonio.colored as tonio

from ..utils import fs_watch
from ..utils.fs_watch import close_watcher, unwatch_file, watch_file, watch_with_error_handler


_UNSET = object()

_GIT_SYMBOLIC_REF_ARGS = ["--no-optional-locks", "symbolic-ref", "--quiet", "--short", "HEAD"]


def _find_git_paths(cwd: str) -> dict | None:
    """Find git metadata paths by walking up from cwd.

    Handles both regular git repos (.git is a directory) and worktrees
    (.git is a file). Returns ``{"repoDir", "commonGitDir", "headPath"}``.
    """
    directory = cwd
    while True:
        git_path = os.path.join(directory, ".git")
        if os.path.exists(git_path):
            try:
                if os.path.isfile(git_path):
                    with open(git_path, encoding="utf-8") as f:
                        content = f.read().strip()
                    if content.startswith("gitdir: "):
                        git_dir = os.path.abspath(os.path.join(directory, content[8:].strip()))
                        head_path = os.path.join(git_dir, "HEAD")
                        if not os.path.exists(head_path):
                            return None
                        common_dir_path = os.path.join(git_dir, "commondir")
                        if os.path.exists(common_dir_path):
                            with open(common_dir_path, encoding="utf-8") as f:
                                common_git_dir = os.path.abspath(os.path.join(git_dir, f.read().strip()))
                        else:
                            common_git_dir = git_dir
                        return {"repoDir": directory, "commonGitDir": common_git_dir, "headPath": head_path}
                elif os.path.isdir(git_path):
                    head_path = os.path.join(git_path, "HEAD")
                    if not os.path.exists(head_path):
                        return None
                    return {"repoDir": directory, "commonGitDir": git_path, "headPath": head_path}
            except OSError:
                return None
        parent = os.path.dirname(directory)
        if parent == directory:
            return None
        directory = parent


# Sync by design, but only safe because of priming: `prime()` resolves the
# branch through `spawn_blocking` before the first render, so the render
# path reads the cache. The other caller is the daemon refresh thread,
# which is not a runtime worker. An *unprimed* provider would still hit
# the lazy fallback in `get_git_branch()` on a worker — prime it.
def _resolve_branch_with_git_sync(repo_dir: str) -> str | None:
    """Ask git for the current branch. None on detached HEAD or git failure."""
    try:
        result = subprocess.run(  # noqa: S603
            ["git", *_GIT_SYMBOLIC_REF_ARGS],  # noqa: S607 - PATH lookup like pi's spawnSync
            cwd=repo_dir,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            encoding="utf-8",
            check=False,
        )
    except OSError:
        return None
    branch = result.stdout.strip() if result.returncode == 0 else ""
    return branch or None


def _resolve_branch_with_git_async(repo_dir: str) -> str | None:
    """The refresh-path variant (pi's execFile); runs on the debounce thread."""
    return _resolve_branch_with_git_sync(repo_dir)


def _is_wsl_environment() -> bool:

    return sys.platform == "linux" and bool(os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"))


_WINDOWS_MOUNT_RE = re.compile(r"^/mnt/[a-z](?:/|$)", re.IGNORECASE)


def _is_windows_mounted_repo_path(repo_dir: str) -> bool:
    return _WINDOWS_MOUNT_RE.match(repo_dir) is not None


def _should_poll_git_head(repo_dir: str) -> bool:
    return _is_wsl_environment() and _is_windows_mounted_repo_path(repo_dir)


class FooterDataProvider:
    """Provides git branch and extension statuses.

    Data not otherwise accessible to extensions; token stats and model info
    are available via the session manager and model.
    """

    WATCH_DEBOUNCE_MS = 500

    def __init__(self, cwd: str) -> None:
        self._lock = threading.RLock()
        self._cwd = cwd
        self._extension_statuses: dict = {}
        self._cached_branch = _UNSET
        self._head_watcher = None
        self._head_watch_file_path: str | None = None
        self._head_watch_file_listener = None
        self._reftable_watcher = None
        self._reftable_tables_list_watcher = None
        self._reftable_tables_list_path: str | None = None
        self._branch_change_callbacks: list = []
        self._available_provider_count = 0
        self._refresh_timer: threading.Timer | None = None
        self._git_watcher_retry_timer: threading.Timer | None = None
        self._refresh_in_flight = False
        self._refresh_pending = False
        self._disposed = False
        # No I/O here: `_find_git_paths` reads git metadata and the watcher
        # touches the filesystem, and a constructor cannot await. `prime()`
        # does both from an async caller; until then `get_git_branch()` falls
        # back to resolving lazily, which is the pre-prime behaviour.
        self._git_paths: dict | None = None
        self._primed = False

    async def prime(self) -> None:
        """Resolve git paths and the branch off the runtime, then start the
        watcher. Call once from an async caller before the first render:
        afterwards `get_git_branch()` is a pure cache read, and the daemon
        refresh thread keeps it current."""
        if self._primed:
            return
        self._primed = True
        self._git_paths = await tonio.spawn_blocking(_find_git_paths, self._cwd)
        branch = await tonio.spawn_blocking(self._resolve_git_branch_sync)
        with self._lock:
            self._cached_branch = branch
        self._setup_git_watcher()

    def prime_sync(self) -> None:
        """Priming for callers with no runtime running — no worker exists to
        block, the same boundary condition that exempts import-time code."""
        if self._primed:
            return
        self._primed = True
        self._git_paths = _find_git_paths(self._cwd)
        with self._lock:
            self._cached_branch = self._resolve_git_branch_sync()
        self._setup_git_watcher()

    def get_git_branch(self) -> str | None:
        """Current branch, None if not in repo, "detached" on detached HEAD."""
        with self._lock:
            if self._cached_branch is _UNSET:
                self._cached_branch = self._resolve_git_branch_sync()
            return self._cached_branch

    def get_extension_statuses(self) -> dict:
        """Extension status texts set via ctx.ui.set_status()."""
        return self._extension_statuses

    def on_branch_change(self, callback):
        """Subscribe to git branch changes. Returns unsubscribe function."""
        with self._lock:
            self._branch_change_callbacks.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._branch_change_callbacks:
                    self._branch_change_callbacks.remove(callback)

        return unsubscribe

    def set_extension_status(self, key: str, text: str | None) -> None:
        if text is None:
            self._extension_statuses.pop(key, None)
        else:
            self._extension_statuses[key] = text

    def clear_extension_statuses(self) -> None:
        self._extension_statuses.clear()

    def get_available_provider_count(self) -> int:
        """Number of unique providers with available models (footer display)."""
        return self._available_provider_count

    def set_available_provider_count(self, count: int) -> None:
        self._available_provider_count = count

    def set_cwd(self, cwd: str) -> None:
        with self._lock:
            if self._cwd == cwd:
                return

            self._cwd = cwd
            if self._refresh_timer is not None:
                self._refresh_timer.cancel()
                self._refresh_timer = None
            self._clear_git_watchers()
            self._cached_branch = _UNSET
            self._git_paths = _find_git_paths(cwd)
            self._setup_git_watcher()
            callbacks = list(self._branch_change_callbacks)
        for callback in callbacks:
            callback()

    def dispose(self) -> None:
        with self._lock:
            self._disposed = True
            if self._refresh_timer is not None:
                self._refresh_timer.cancel()
                self._refresh_timer = None
            self._clear_git_watchers()
            self._branch_change_callbacks.clear()

    def _notify_branch_change(self) -> None:
        with self._lock:
            callbacks = list(self._branch_change_callbacks)
        for callback in callbacks:
            callback()

    def _schedule_refresh(self) -> None:
        with self._lock:
            if self._disposed or self._refresh_timer is not None:
                return
            if self._refresh_in_flight:
                self._refresh_pending = True
                return

            def fire() -> None:
                with self._lock:
                    self._refresh_timer = None
                self._refresh_git_branch_async()

            self._refresh_timer = threading.Timer(FooterDataProvider.WATCH_DEBOUNCE_MS / 1000, fire)
            self._refresh_timer.daemon = True
            self._refresh_timer.start()

    def _refresh_git_branch_async(self) -> None:
        with self._lock:
            if self._disposed:
                return
            if self._refresh_in_flight:
                self._refresh_pending = True
                return
            self._refresh_in_flight = True
            git_paths = self._git_paths

        notify = False
        try:
            next_branch = self._resolve_git_branch_async(git_paths)
            with self._lock:
                if self._disposed:
                    return
                if self._cached_branch is not _UNSET and self._cached_branch != next_branch:
                    self._cached_branch = next_branch
                    notify = True
                else:
                    self._cached_branch = next_branch
        finally:
            if notify:
                self._notify_branch_change()
            with self._lock:
                self._refresh_in_flight = False
                pending = self._refresh_pending and not self._disposed
                self._refresh_pending = False
            if pending:
                self._schedule_refresh()

    def _resolve_git_branch_sync(self) -> str | None:
        try:
            if not self._git_paths:
                return None
            with open(self._git_paths["headPath"], encoding="utf-8") as f:
                content = f.read().strip()
            if content.startswith("ref: refs/heads/"):
                branch = content[16:]
                if branch == ".invalid":
                    return _resolve_branch_with_git_sync(self._git_paths["repoDir"]) or "detached"
                return branch
            return "detached"
        except OSError:
            return None

    def _resolve_git_branch_async(self, git_paths: dict | None) -> str | None:
        try:
            if not git_paths:
                return None
            with open(git_paths["headPath"], encoding="utf-8") as f:
                content = f.read().strip()
            if content.startswith("ref: refs/heads/"):
                branch = content[16:]
                if branch == ".invalid":
                    return _resolve_branch_with_git_async(git_paths["repoDir"]) or "detached"
                return branch
            return "detached"
        except OSError:
            return None

    def _clear_git_watchers(self) -> None:
        close_watcher(self._head_watcher)
        self._head_watcher = None
        if self._head_watch_file_path and self._head_watch_file_listener:
            unwatch_file(self._head_watch_file_path, self._head_watch_file_listener)
            self._head_watch_file_path = None
            self._head_watch_file_listener = None
        close_watcher(self._reftable_watcher)
        self._reftable_watcher = None
        close_watcher(self._reftable_tables_list_watcher)
        self._reftable_tables_list_watcher = None
        if self._reftable_tables_list_path:
            unwatch_file(self._reftable_tables_list_path)
            self._reftable_tables_list_path = None
        if self._git_watcher_retry_timer is not None:
            self._git_watcher_retry_timer.cancel()
            self._git_watcher_retry_timer = None

    def _schedule_git_watcher_retry(self) -> None:
        if self._disposed or self._git_watcher_retry_timer is not None:
            return

        def fire() -> None:
            with self._lock:
                self._git_watcher_retry_timer = None
                self._setup_git_watcher()

        # Read the delay via the module so tests can shorten it
        self._git_watcher_retry_timer = threading.Timer(fs_watch.FS_WATCH_RETRY_DELAY_MS / 1000, fire)
        self._git_watcher_retry_timer.daemon = True
        self._git_watcher_retry_timer.start()

    def _handle_git_watcher_error(self) -> None:
        with self._lock:
            self._clear_git_watchers()
            self._schedule_git_watcher_retry()

    def _setup_git_watcher(self) -> None:
        self._clear_git_watchers()
        if not self._git_paths:
            return

        poll_git_head = _should_poll_git_head(self._git_paths["repoDir"])

        def on_head_dir_event(_event_type: str, filename: str | None) -> None:
            if not filename or filename == "HEAD":
                self._schedule_refresh()

        # Watch the directory containing HEAD, not HEAD itself. Git uses
        # atomic writes (write temp, rename over HEAD), which changes the
        # inode, and a file watch stops working after the inode changes.
        self._head_watcher = watch_with_error_handler(
            os.path.dirname(self._git_paths["headPath"]),
            on_head_dir_event,
            self._handle_git_watcher_error,
        )
        if poll_git_head:
            self._head_watch_file_path = self._git_paths["headPath"]

            def on_head_stat(current: dict, previous: dict) -> None:
                if (
                    current["mtimeMs"] != previous["mtimeMs"]
                    or current["ctimeMs"] != previous["ctimeMs"]
                    or current["size"] != previous["size"]
                ):
                    self._schedule_refresh()

            self._head_watch_file_listener = on_head_stat
            watch_file(self._head_watch_file_path, 1000, on_head_stat)
        if self._head_watcher is None and not poll_git_head:
            return

        # In reftable repos, branch switches update files in the reftable
        # directory instead of HEAD. Watch it separately so the footer picks
        # up those changes.
        reftable_dir = os.path.join(self._git_paths["commonGitDir"], "reftable")
        if os.path.exists(reftable_dir):
            self._reftable_watcher = watch_with_error_handler(
                reftable_dir,
                lambda _event_type, _filename: self._schedule_refresh(),
                self._handle_git_watcher_error,
            )
            if self._reftable_watcher is None:
                return

            tables_list_path = os.path.join(reftable_dir, "tables.list")
            if os.path.exists(tables_list_path):
                self._reftable_tables_list_path = tables_list_path
                self._reftable_tables_list_watcher = watch_with_error_handler(
                    tables_list_path,
                    lambda _event_type, _filename: self._schedule_refresh(),
                    self._handle_git_watcher_error,
                )
                if self._reftable_tables_list_watcher is None:
                    return

                def on_tables_list_stat(current: dict, previous: dict) -> None:
                    if (
                        current["mtimeMs"] != previous["mtimeMs"]
                        or current["ctimeMs"] != previous["ctimeMs"]
                        or current["size"] != previous["size"]
                    ):
                        self._schedule_refresh()

                watch_file(tables_list_path, 250, on_tables_list_stat)
