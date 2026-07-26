"""Mirror of pi coding-agent src/core/package-manager.ts.

Resolves local resources (settings-configured entries with include/exclude
patterns, and auto-discovery from the user/project resource directories) and
`packages` sources.

**Package sources are git and local only** (decided 2026-07-26). pi also
supports `npm:`, and roughly a third of its package-manager is the npm/pnpm/bun
side of that: install roots, version ranges, `npm view` update checks, legacy
global-install migration. A pidrei extension is a directory of `.py` modules,
which is exactly what a git checkout hands you, so `npm:` has no analogue worth
inventing — a PyPI story would buy a package-manager dependency and its own
update path for a distribution channel we have not committed to yet (Phase 7).
`npm:` sources are refused by name rather than silently ignored.

Two further consequences of that decision, both deliberate:

- pi runs `npm install --omit=dev` in a freshly cloned git package. The Python
  equivalent would be installing a cloned package's dependencies into the host
  interpreter, which is not something a config file should be able to do
  silently; a git package is expected to be self-contained or to declare its
  needs in its README.
- pi's manifest is `package.json`'s `pi` key; here it is `pyproject.toml`'s
  `[tool.pidrei]` table, the same file the extension loader reads.
"""

import glob
import hashlib
import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Any

import tonio.colored as tonio

from ..config import CONFIG_DIR_NAME
from ..utils.git import parse_git_url
from ..utils.paths import canonicalize_path, is_local_path, resolve_path
from .exec import exec_command
from .extensions.loader import is_extension_file, read_pidrei_manifest, resolve_extension_entries
from .settings_manager import SettingsManager
from .skills import IgnoreMatcher, add_ignore_rules
from .source_info import PathMetadata


RESOURCE_TYPES = ("extensions", "skills", "prompts", "themes")

_FILE_PATTERNS: dict[str, re.Pattern] = {
    # pi matches `.ts`/`.js`; a pidrei extension is a Python module.
    "extensions": re.compile(r"\.py$"),
    "skills": re.compile(r"\.md$"),
    "prompts": re.compile(r"\.md$"),
    "themes": re.compile(r"\.json$"),
}


@dataclass(slots=True)
class ResolvedResource:
    path: str
    enabled: bool
    metadata: PathMetadata


@dataclass(slots=True)
class ResolvedPaths:
    extensions: list[ResolvedResource] = field(default_factory=list)
    skills: list[ResolvedResource] = field(default_factory=list)
    prompts: list[ResolvedResource] = field(default_factory=list)
    themes: list[ResolvedResource] = field(default_factory=list)


@dataclass(slots=True)
class ProgressEvent:
    """pi's ProgressEvent: `type` is start|complete|error, `action` is
    install|pull|remove|update."""

    type: str
    action: str
    source: str
    message: str | None = None


@dataclass(slots=True)
class ConfiguredPackage:
    source: str
    scope: str  # "user" | "project"
    installed_path: str | None = None


@dataclass(slots=True)
class PackageUpdate:
    source: str
    display_name: str
    scope: str  # "user" | "project"
    #: pi carries "npm" | "git"; only git sources exist here.
    type: str = "git"


@dataclass(slots=True)
class LocalSource:
    path: str
    type: str = "local"


@dataclass(slots=True)
class GitSource:
    repo: str
    host: str
    path: str
    ref: str | None = None
    type: str = "git"

    @property
    def pinned(self) -> bool:
        """A configured ref is a checkout target, not a floating branch."""
        return self.ref is not None


class UnsupportedSourceError(Exception):
    """Raised for a source pidrei deliberately does not support (`npm:`)."""


def is_offline_mode_enabled() -> bool:
    return bool(os.environ.get("PIDREI_OFFLINE"))


def get_extension_temp_folder(agent_dir: str) -> str:
    temp_folder = os.path.join(agent_dir, "tmp", "extensions")
    os.makedirs(temp_folder, mode=0o700, exist_ok=True)
    os.chmod(temp_folder, 0o700)
    return temp_folder


def _get_home_dir() -> str:
    return os.environ.get("HOME") or os.path.expanduser("~")


def _resource_precedence_rank(metadata: PathMetadata) -> int:
    if metadata.origin == "package":
        return 4
    scope_base = 0 if metadata.scope == "project" else 2
    return scope_base + (0 if metadata.source == "local" else 1)


def _is_pattern(value: str) -> bool:
    return value.startswith(("!", "+", "-")) or "*" in value or "?" in value


def _split_patterns(entries: list[str]) -> tuple[list[str], list[str]]:
    plain: list[str] = []
    patterns: list[str] = []
    for entry in entries:
        if _is_pattern(entry):
            patterns.append(entry)
        else:
            plain.append(entry)
    return plain, patterns


def _minimatch(value: str, pattern: str) -> bool:
    """minimatch subset used by resource patterns (`*`/`?` don't cross `/`)."""
    regex = ""
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "*":
            if pattern[index : index + 2] == "**":
                regex += ".*"
                index += 2
                continue
            regex += "[^/]*"
        elif char == "?":
            regex += "[^/]"
        elif char == "[":
            end = pattern.find("]", index + 1)
            if end == -1:
                regex += re.escape(char)
            else:
                regex += pattern[index : end + 1]
                index = end + 1
                continue
        else:
            regex += re.escape(char)
        index += 1
    return re.fullmatch(regex, value) is not None


def _matches_any_pattern(file_path: str, patterns: list[str], base_dir: str) -> bool:
    rel = os.path.relpath(file_path, base_dir)
    name = os.path.basename(file_path)
    is_skill_file = name == "SKILL.md"
    parent_dir = os.path.dirname(file_path) if is_skill_file else None
    parent_rel = os.path.relpath(parent_dir, base_dir) if is_skill_file else None
    parent_name = os.path.basename(parent_dir) if is_skill_file else None

    for pattern in patterns:
        if _minimatch(rel, pattern) or _minimatch(name, pattern) or _minimatch(file_path, pattern):
            return True
        if not is_skill_file:
            continue
        if _minimatch(parent_rel, pattern) or _minimatch(parent_name, pattern) or _minimatch(parent_dir, pattern):
            return True
    return False


def _normalize_exact_pattern(pattern: str) -> str:
    return pattern.removeprefix("./")


def _matches_any_exact_pattern(file_path: str, patterns: list[str], base_dir: str) -> bool:
    if not patterns:
        return False
    rel = os.path.relpath(file_path, base_dir)
    name = os.path.basename(file_path)
    is_skill_file = name == "SKILL.md"
    parent_dir = os.path.dirname(file_path) if is_skill_file else None
    parent_rel = os.path.relpath(parent_dir, base_dir) if is_skill_file else None

    for pattern in patterns:
        normalized = _normalize_exact_pattern(pattern)
        if normalized in (rel, file_path):
            return True
        if not is_skill_file:
            continue
        if normalized in (parent_rel, parent_dir):
            return True
    return False


def _is_override_pattern(value: str) -> bool:
    return value.startswith(("!", "+", "-"))


def _has_glob_pattern(value: str) -> bool:
    return "*" in value or "?" in value


def _get_override_patterns(entries: list[str]) -> list[str]:
    return [pattern for pattern in entries if _is_override_pattern(pattern)]


def apply_autoload_disabled_patterns(all_paths: list[str], patterns: list[str], base_dir: str) -> dict[str, bool]:
    """Patterns for an `autoload: false` package: only what a pattern names is
    decided at all, so the package contributes nothing by default."""
    result: dict[str, bool] = {}
    for pattern in patterns:
        target = pattern[1:] if pattern.startswith(("+", "-", "!")) else pattern
        enabled = not pattern.startswith(("-", "!"))
        exact = pattern.startswith(("+", "-"))
        for file_path in all_paths:
            matched = (
                _matches_any_exact_pattern(file_path, [target], base_dir)
                if exact
                else _matches_any_pattern(file_path, [target], base_dir)
            )
            if matched:
                result[file_path] = enabled
    return result


def is_enabled_by_overrides(file_path: str, patterns: list[str], base_dir: str) -> bool:
    overrides = _get_override_patterns(patterns)
    excludes = [pattern[1:] for pattern in overrides if pattern.startswith("!")]
    force_includes = [pattern[1:] for pattern in overrides if pattern.startswith("+")]
    force_excludes = [pattern[1:] for pattern in overrides if pattern.startswith("-")]

    enabled = True
    if excludes and _matches_any_pattern(file_path, excludes, base_dir):
        enabled = False
    if force_includes and _matches_any_exact_pattern(file_path, force_includes, base_dir):
        enabled = True
    if force_excludes and _matches_any_exact_pattern(file_path, force_excludes, base_dir):
        enabled = False
    return enabled


def apply_patterns(all_paths: list[str], patterns: list[str], base_dir: str) -> set[str]:
    """Apply include/exclude/force patterns to paths, returning the enabled set."""
    includes: list[str] = []
    excludes: list[str] = []
    force_includes: list[str] = []
    force_excludes: list[str] = []

    for pattern in patterns:
        if pattern.startswith("+"):
            force_includes.append(pattern[1:])
        elif pattern.startswith("-"):
            force_excludes.append(pattern[1:])
        elif pattern.startswith("!"):
            excludes.append(pattern[1:])
        else:
            includes.append(pattern)

    # Step 1: Apply includes (or all if no includes)
    if not includes:
        result = list(all_paths)
    else:
        result = [path for path in all_paths if _matches_any_pattern(path, includes, base_dir)]

    # Step 2: Apply excludes
    if excludes:
        result = [path for path in result if not _matches_any_pattern(path, excludes, base_dir)]

    # Step 3: Force-include (add back from all_paths, overriding exclusions)
    if force_includes:
        for path in all_paths:
            if path not in result and _matches_any_exact_pattern(path, force_includes, base_dir):
                result.append(path)

    # Step 4: Force-exclude (remove even if included or force-included)
    if force_excludes:
        result = [path for path in result if not _matches_any_exact_pattern(path, force_excludes, base_dir)]

    return set(result)


def _stat_kind(full_path: str) -> tuple[bool, bool] | None:
    try:
        is_dir = os.path.isdir(full_path)
        is_file = os.path.isfile(full_path)
    except OSError:
        return None
    if not is_dir and not is_file and not os.path.exists(full_path):
        return None  # Broken symlink
    return is_dir, is_file


def _collect_files(
    dir: str,
    file_pattern: re.Pattern,
    skip_node_modules: bool = True,
    ignore_matcher: IgnoreMatcher | None = None,
    root_dir: str | None = None,
) -> list[str]:
    files: list[str] = []
    if not os.path.exists(dir):
        return files

    root = root_dir if root_dir is not None else dir
    ig = ignore_matcher if ignore_matcher is not None else IgnoreMatcher()
    add_ignore_rules(ig, dir, root)

    try:
        entries = sorted(os.scandir(dir), key=lambda entry: entry.name)
    except OSError:
        return files

    for entry in entries:
        if entry.name.startswith("."):
            continue
        if skip_node_modules and entry.name == "node_modules":
            continue

        full_path = os.path.join(dir, entry.name)
        kind = _stat_kind(full_path)
        if kind is None:
            continue
        is_dir, is_file = kind

        rel_path = os.path.relpath(full_path, root)
        ignore_path = f"{rel_path}/" if is_dir else rel_path
        if ig.ignores(ignore_path):
            continue

        if is_dir:
            files.extend(_collect_files(full_path, file_pattern, skip_node_modules, ig, root))
        elif is_file and file_pattern.search(entry.name):
            files.append(full_path)

    return files


def _collect_skill_entries(
    dir: str,
    mode: str,  # "pi" | "agents"
    ignore_matcher: IgnoreMatcher | None = None,
    root_dir: str | None = None,
) -> list[str]:
    entries: list[str] = []
    if not os.path.exists(dir):
        return entries

    root = root_dir if root_dir is not None else dir
    ig = ignore_matcher if ignore_matcher is not None else IgnoreMatcher()
    add_ignore_rules(ig, dir, root)

    try:
        dir_entries = sorted(os.scandir(dir), key=lambda entry: entry.name)
    except OSError:
        return entries

    for entry in dir_entries:
        if entry.name != "SKILL.md":
            continue

        full_path = os.path.join(dir, entry.name)
        kind = _stat_kind(full_path)
        if kind is None:
            continue
        _is_dir, is_file = kind

        rel_path = os.path.relpath(full_path, root)
        if is_file and not ig.ignores(rel_path):
            entries.append(full_path)
            return entries

    for entry in dir_entries:
        if entry.name.startswith("."):
            continue
        if entry.name == "node_modules":
            continue

        full_path = os.path.join(dir, entry.name)
        kind = _stat_kind(full_path)
        if kind is None:
            continue
        is_dir, is_file = kind

        rel_path = os.path.relpath(full_path, root)
        if mode == "pi" and dir == root and is_file and entry.name.endswith(".md") and not ig.ignores(rel_path):
            entries.append(full_path)
            continue

        if not is_dir:
            continue
        if ig.ignores(f"{rel_path}/"):
            continue

        entries.extend(_collect_skill_entries(full_path, mode, ig, root))

    return entries


def _collect_flat_entries(dir: str, suffix: str) -> list[str]:
    entries: list[str] = []
    if not os.path.exists(dir):
        return entries

    ig = IgnoreMatcher()
    add_ignore_rules(ig, dir, dir)

    try:
        dir_entries = sorted(os.scandir(dir), key=lambda entry: entry.name)
    except OSError:
        return entries

    for entry in dir_entries:
        if entry.name.startswith("."):
            continue
        if entry.name == "node_modules":
            continue

        full_path = os.path.join(dir, entry.name)
        kind = _stat_kind(full_path)
        if kind is None:
            continue
        _is_dir, is_file = kind

        rel_path = os.path.relpath(full_path, dir)
        if ig.ignores(rel_path):
            continue

        if is_file and entry.name.endswith(suffix):
            entries.append(full_path)

    return entries


def _collect_auto_extension_entries(dir: str) -> list[str]:
    """Auto-discovery of extension entry files (pi's collectAutoExtensionEntries)."""
    entries: list[str] = []
    if not os.path.exists(dir):
        return entries

    # A directory that declares its own entry points is one extension, not a
    # folder of them.
    root_entries = resolve_extension_entries(dir)
    if root_entries:
        return root_entries

    ig = IgnoreMatcher()
    add_ignore_rules(ig, dir, dir)

    try:
        dir_entries = sorted(os.scandir(dir), key=lambda entry: entry.name)
    except OSError:
        return entries

    for entry in dir_entries:
        if entry.name.startswith("."):
            continue
        if entry.name == "node_modules":
            continue

        full_path = os.path.join(dir, entry.name)
        kind = _stat_kind(full_path)
        if kind is None:
            continue
        is_dir, is_file = kind

        rel_path = os.path.relpath(full_path, dir)
        ignore_path = f"{rel_path}/" if is_dir else rel_path
        if ig.ignores(ignore_path):
            continue

        if is_file and is_extension_file(entry.name):
            entries.append(full_path)
        elif is_dir:
            resolved_entries = resolve_extension_entries(full_path)
            if resolved_entries:
                entries.extend(resolved_entries)

    return entries


def _collect_resource_files(dir: str, resource_type: str) -> list[str]:
    if resource_type == "skills":
        return _collect_skill_entries(dir, "pi")
    if resource_type == "extensions":
        return _collect_auto_extension_entries(dir)
    return _collect_files(dir, _FILE_PATTERNS[resource_type])


def _find_git_repo_root(start_dir: str) -> str | None:
    dir = os.path.abspath(start_dir)
    while True:
        if os.path.exists(os.path.join(dir, ".git")):
            return dir
        parent = os.path.dirname(dir)
        if parent == dir:
            return None
        dir = parent


def collect_ancestor_agents_skill_dirs(start_dir: str) -> list[str]:
    skill_dirs: list[str] = []
    resolved_start_dir = os.path.abspath(start_dir)
    git_repo_root = _find_git_repo_root(resolved_start_dir)

    dir = resolved_start_dir
    while True:
        skill_dirs.append(os.path.join(dir, ".agents", "skills"))
        if git_repo_root and dir == git_repo_root:
            break
        parent = os.path.dirname(dir)
        if parent == dir:
            break
        dir = parent

    return skill_dirs


class DefaultPackageManager:
    """Resource and package-source resolver (see the module docstring for what
    the git-only decision drops)."""

    def __init__(self, *, cwd: str, agent_dir: str, settings_manager: SettingsManager):
        self._cwd = resolve_path(cwd)
        self._agent_dir = resolve_path(agent_dir)
        self._settings_manager = settings_manager
        self._progress_callback = None

    # -- progress ----------------------------------------------------------------

    def set_progress_callback(self, callback) -> None:
        self._progress_callback = callback

    def _emit_progress(self, event: ProgressEvent) -> None:
        if self._progress_callback is not None:
            self._progress_callback(event)

    async def _with_progress(self, action: str, source: str, message: str, operation) -> None:
        self._emit_progress(ProgressEvent(type="start", action=action, source=source, message=message))
        try:
            await operation()
        except Exception as error:
            self._emit_progress(ProgressEvent(type="error", action=action, source=source, message=str(error)))
            raise
        self._emit_progress(ProgressEvent(type="complete", action=action, source=source))

    # -- source parsing ----------------------------------------------------------

    def parse_source(self, source: str) -> LocalSource | GitSource:
        """pi returns an npm|git|local union; here `npm:` is refused by name so
        a stale pi config fails loudly instead of resolving to nothing."""
        if source.startswith("npm:"):
            raise UnsupportedSourceError(
                f"npm package sources are not supported: {source}. "
                "Use a git source (git:… or an https:// URL) or a local path."
            )

        if is_local_path(source):
            return LocalSource(path=source)

        parsed = parse_git_url(source)
        if parsed is not None:
            return GitSource(repo=parsed["repo"], host=parsed["host"], path=parsed["path"], ref=parsed.get("ref"))

        return LocalSource(path=source)

    def _get_package_source_string(self, package: Any) -> str:
        return package if isinstance(package, str) else package["source"]

    def _get_package_filter(self, package: Any) -> dict | None:
        return None if isinstance(package, str) else package

    def _get_package_identity(self, source: str, scope: str | None = None) -> str:
        parsed = self.parse_source(source)
        if isinstance(parsed, GitSource):
            # host/path, so ssh and https forms of one repo are one package.
            return f"git:{parsed.host}/{parsed.path}"
        if scope:
            return f"local:{self._resolve_path_from_base(parsed.path, self._get_base_dir_for_scope(scope))}"
        return f"local:{self._resolve_resource_path(parsed.path)}"

    def _resolve_resource_path(self, path: str) -> str:
        return resolve_path(path, self._cwd, home_dir=_get_home_dir(), trim=True)

    def _resolve_path_from_base(self, input: str, base_dir: str) -> str:
        return resolve_path(input, base_dir, home_dir=_get_home_dir(), trim=True)

    # -- scopes and managed paths -------------------------------------------------

    def _assert_project_trusted_for_scope(self, scope: str) -> None:
        if scope == "project" and not self._settings_manager.is_project_trusted():
            raise Exception("Project is not trusted; refusing to access project package storage")

    def _get_base_dir_for_scope(self, scope: str) -> str:
        if scope == "project":
            self._assert_project_trusted_for_scope(scope)
            return os.path.join(self._cwd, CONFIG_DIR_NAME)
        if scope == "user":
            return self._agent_dir
        return self._cwd

    def _resolve_managed_path(self, root: str, *parts: str) -> str:
        resolved_root = os.path.abspath(root)
        resolved_path = os.path.abspath(os.path.join(resolved_root, *parts))
        if resolved_path != resolved_root and not resolved_path.startswith(f"{resolved_root}{os.sep}"):
            raise Exception(f"Refusing to use path outside package install root: {resolved_path}")
        return resolved_path

    def _get_temporary_dir(self, prefix: str, suffix: str | None = None) -> str:
        root = self._resolve_managed_path(get_extension_temp_folder(self._agent_dir), prefix)
        digest = hashlib.sha256(f"{prefix}-{suffix or ''}".encode()).hexdigest()[:8]
        return self._resolve_managed_path(root, digest, suffix or "")

    def _get_git_install_root(self, scope: str) -> str | None:
        if scope == "temporary":
            return None
        if scope == "project":
            self._assert_project_trusted_for_scope(scope)
            return os.path.join(self._cwd, CONFIG_DIR_NAME, "git")
        return os.path.join(self._agent_dir, "git")

    def _get_git_install_path(self, source: GitSource, scope: str) -> str:
        if scope == "temporary":
            return self._get_temporary_dir(f"git-{source.host}", source.path)
        install_root = self._get_git_install_root(scope)
        if not install_root:
            raise Exception("Missing git install root")
        return self._resolve_managed_path(install_root, source.host, source.path)

    def get_installed_path(self, source: str, scope: str) -> str | None:
        parsed = self.parse_source(source)
        if isinstance(parsed, GitSource):
            return self._get_git_install_path(parsed, scope)
        return None

    # -- git ---------------------------------------------------------------------

    @staticmethod
    def _ensure_git_ignore(git_root: str) -> None:
        os.makedirs(git_root, exist_ok=True)
        ignore_path = os.path.join(git_root, ".gitignore")
        if not os.path.exists(ignore_path):
            with open(ignore_path, "w", encoding="utf-8") as handle:
                handle.write("*\n")

    async def _run_command(self, command: str, args: list[str], *, cwd: str | None = None) -> str:
        result = await exec_command(command, args, cwd or self._cwd)
        if result.code != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise Exception(f"{command} {' '.join(args)} failed: {detail}")
        return result.stdout

    async def _ensure_git_ref(self, target_dir: str, fetch_args: list[str], ref: str) -> None:
        await self._run_command("git", fetch_args, cwd=target_dir)
        await self._run_command("git", ["checkout", "--force", ref], cwd=target_dir)

    async def _get_local_git_update_target(self, target_dir: str) -> tuple[list[str], str]:
        """Track the checked-out branch when the source pins no ref."""
        try:
            branch = (await self._run_command("git", ["rev-parse", "--abbrev-ref", "HEAD"], cwd=target_dir)).strip()
        except Exception:
            branch = ""
        if not branch or branch == "HEAD":
            return ["fetch", "origin"], "FETCH_HEAD"
        return ["fetch", "origin", branch], "FETCH_HEAD"

    async def _install_git(self, source: GitSource, scope: str) -> None:
        target_dir = self._get_git_install_path(source, scope)
        if os.path.exists(target_dir):
            # Reconcile an existing checkout rather than re-cloning.
            if source.ref:
                await self._ensure_git_ref(target_dir, ["fetch", "origin", source.ref], "FETCH_HEAD")
                return
            fetch_args, ref = await self._get_local_git_update_target(target_dir)
            await self._ensure_git_ref(target_dir, fetch_args, ref)
            return

        git_root = self._get_git_install_root(scope)
        if git_root:
            self._ensure_git_ignore(git_root)
        os.makedirs(os.path.dirname(target_dir), exist_ok=True)

        await self._run_command("git", ["clone", source.repo, target_dir])
        if source.ref:
            await self._run_command("git", ["checkout", source.ref], cwd=target_dir)

    async def _update_git(self, source: GitSource, scope: str) -> None:
        target_dir = self._get_git_install_path(source, scope)
        if not os.path.exists(target_dir):
            await self._install_git(source, scope)
            return
        if source.ref:
            await self._ensure_git_ref(target_dir, ["fetch", "origin", source.ref], "FETCH_HEAD")
            return
        fetch_args, ref = await self._get_local_git_update_target(target_dir)
        await self._ensure_git_ref(target_dir, fetch_args, ref)

    async def _refresh_temporary_git_source(self, source: GitSource, source_str: str) -> None:
        if is_offline_mode_enabled():
            return
        try:
            await self._with_progress(
                "pull", source_str, f"Refreshing {source_str}...", lambda: self._update_git(source, "temporary")
            )
        except Exception:
            pass  # Keep the cached temporary checkout if the refresh fails.

    async def _install_parsed_source(self, parsed: LocalSource | GitSource, scope: str) -> None:
        if isinstance(parsed, GitSource):
            await self._install_git(parsed, scope)

    async def _remove_git(self, source: GitSource, scope: str) -> None:
        target_dir = self._get_git_install_path(source, scope)
        if not os.path.exists(target_dir):
            return
        shutil.rmtree(target_dir, ignore_errors=True)
        self._prune_empty_git_parents(target_dir, self._get_git_install_root(scope))

    @staticmethod
    def _prune_empty_git_parents(target_dir: str, install_root: str | None) -> None:
        if not install_root:
            return
        resolved_root = os.path.abspath(install_root)
        current = os.path.dirname(target_dir)
        while current.startswith(resolved_root) and current != resolved_root:
            if not os.path.exists(current):
                current = os.path.dirname(current)
                continue
            if os.listdir(current):
                break
            try:
                shutil.rmtree(current, ignore_errors=True)
            except OSError:
                break
            current = os.path.dirname(current)

    # -- install / remove / update ------------------------------------------------

    async def install(self, source: str, *, local: bool = False) -> None:
        parsed = self.parse_source(source)
        scope = "project" if local else "user"
        self._assert_project_trusted_for_scope(scope)

        async def run() -> None:
            if isinstance(parsed, GitSource):
                await self._install_git(parsed, scope)
                return
            resolved = self._resolve_resource_path(parsed.path)
            if not os.path.exists(resolved):
                raise Exception(f"Path does not exist: {resolved}")

        await self._with_progress("install", source, f"Installing {source}...", run)

    async def install_and_persist(self, source: str, *, local: bool = False) -> None:
        await self.install(source, local=local)
        self.add_source_to_settings(source, local=local)

    async def remove(self, source: str, *, local: bool = False) -> None:
        parsed = self.parse_source(source)
        scope = "project" if local else "user"
        self._assert_project_trusted_for_scope(scope)

        async def run() -> None:
            if isinstance(parsed, GitSource):
                await self._remove_git(parsed, scope)

        await self._with_progress("remove", source, f"Removing {source}...", run)

    async def remove_and_persist(self, source: str, *, local: bool = False) -> bool:
        await self.remove(source, local=local)
        return self.remove_source_from_settings(source, local=local)

    async def _get_git_upstream_ref(self, target_dir: str) -> str | None:
        try:
            ref = (
                await self._run_command(
                    "git", ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], cwd=target_dir
                )
            ).strip()
        except Exception:
            return None
        if not ref or "/" not in ref:
            return None
        # "origin/main" -> "main": ls-remote wants the ref name, not the
        # remote-tracking form.
        return ref.split("/", 1)[1] or None

    async def _get_remote_git_head(self, target_dir: str) -> str:
        upstream_ref = await self._get_git_upstream_ref(target_dir)
        if upstream_ref:
            output = await self._run_command("git", ["ls-remote", "origin", upstream_ref], cwd=target_dir)
            match = re.search(r"^([0-9a-f]{40})\s+", output, re.MULTILINE)
            if match:
                return match.group(1)

        output = await self._run_command("git", ["ls-remote", "origin", "HEAD"], cwd=target_dir)
        match = re.search(r"^([0-9a-f]{40})\s+HEAD$", output, re.MULTILINE)
        if not match:
            raise Exception("Failed to determine remote HEAD")
        return match.group(1)

    async def _git_has_available_update(self, installed_path: str) -> bool:
        if is_offline_mode_enabled():
            return False
        try:
            local_head = (await self._run_command("git", ["rev-parse", "HEAD"], cwd=installed_path)).strip()
            return local_head != (await self._get_remote_git_head(installed_path)).strip()
        except Exception:
            return False

    async def check_for_available_updates(self) -> list[PackageUpdate]:
        """Configured git packages whose remote has moved.

        Every check is a network round-trip, so they run concurrently and any
        failure is swallowed into "no update" — this feeds a startup notice, and
        a flaky remote must not delay or break a session.
        """
        if is_offline_mode_enabled():
            return []

        candidates: list[tuple[GitSource, str, str]] = []
        for package in self.list_configured_packages():
            if package.scope == "temporary":
                continue
            parsed = self.parse_source(package.source)
            # A pinned ref is a checkout target, not a moving branch.
            if not isinstance(parsed, GitSource) or parsed.pinned:
                continue
            installed_path = self._get_git_install_path(parsed, package.scope)
            if not os.path.exists(installed_path):
                continue
            candidates.append((parsed, package.scope, package.source))

        if not candidates:
            return []

        async def check_one(candidate: tuple[GitSource, str, str]) -> PackageUpdate | None:
            parsed, scope, source = candidate
            installed_path = self._get_git_install_path(parsed, scope)
            if not await self._git_has_available_update(installed_path):
                return None
            return PackageUpdate(
                source=source,
                display_name=f"{parsed.host}/{parsed.path}",
                scope=scope,
                type="git",
            )

        results = await tonio.map(check_one, candidates)
        return [update for update in results if update is not None]

    async def update(self, source: str | None = None) -> None:
        """Update configured git packages. Sources with a pinned ref are
        checkout targets, so they are re-reconciled rather than skipped."""
        targets: list[tuple[GitSource, str, str]] = []
        for package in self.list_configured_packages():
            if source is not None and self._source_match_key_for_input(
                package.source
            ) != self._source_match_key_for_input(source):
                continue
            parsed = self.parse_source(package.source)
            if isinstance(parsed, GitSource):
                targets.append((parsed, package.scope, package.source))

        if source is not None and not targets:
            raise Exception(f"No matching package found for {source}")

        # After the no-match check, not before: pi validates the requested
        # source and only then skips the network work, so `update <unknown>`
        # reports the bad source offline too.
        if is_offline_mode_enabled():
            return

        for parsed, scope, source_str in targets:
            await self._with_progress(
                "update",
                source_str,
                f"Updating {source_str}...",
                lambda parsed=parsed, scope=scope: self._update_git(parsed, scope),
            )

    # -- settings -----------------------------------------------------------------

    def _source_match_key_for_input(self, source: str) -> str:
        parsed = self.parse_source(source)
        if isinstance(parsed, GitSource):
            return f"git:{parsed.host}/{parsed.path}"
        return f"local:{self._resolve_resource_path(parsed.path)}"

    def _source_match_key_for_settings(self, source: str, scope: str) -> str:
        parsed = self.parse_source(source)
        if isinstance(parsed, GitSource):
            return f"git:{parsed.host}/{parsed.path}"
        return f"local:{self._resolve_path_from_base(parsed.path, self._get_base_dir_for_scope(scope))}"

    def _package_sources_match(self, existing: Any, input_source: str, scope: str) -> bool:
        left = self._source_match_key_for_settings(self._get_package_source_string(existing), scope)
        return left == self._source_match_key_for_input(input_source)

    def _normalize_package_source_for_settings(self, source: str, scope: str) -> str:
        """Local sources are stored relative to their scope's settings base."""
        parsed = self.parse_source(source)
        if not isinstance(parsed, LocalSource):
            return source
        base_dir = self._get_base_dir_for_scope(scope)
        return os.path.relpath(self._resolve_resource_path(parsed.path), base_dir) or "."

    def _set_packages(self, packages: list[Any], scope: str) -> None:
        if scope == "project":
            self._settings_manager.set_project_packages(packages)
        else:
            self._settings_manager.set_packages(packages)

    def _current_packages(self, scope: str) -> list[Any]:
        settings = (
            self._settings_manager.get_project_settings()
            if scope == "project"
            else self._settings_manager.get_global_settings()
        )
        return list(settings.get("packages") or [])

    def add_source_to_settings(self, source: str, *, local: bool = False) -> bool:
        scope = "project" if local else "user"
        current_packages = self._current_packages(scope)
        normalized = self._normalize_package_source_for_settings(source, scope)

        for index, existing in enumerate(current_packages):
            if not self._package_sources_match(existing, source, scope):
                continue
            if self._get_package_source_string(existing) == normalized:
                return False
            next_packages = list(current_packages)
            # Replacing a ref must not drop the entry's resource filters.
            next_packages[index] = normalized if isinstance(existing, str) else {**existing, "source": normalized}
            self._set_packages(next_packages, scope)
            return True

        self._set_packages([*current_packages, normalized], scope)
        return True

    def remove_source_from_settings(self, source: str, *, local: bool = False) -> bool:
        scope = "project" if local else "user"
        current_packages = self._current_packages(scope)
        next_packages = [
            existing for existing in current_packages if not self._package_sources_match(existing, source, scope)
        ]
        if len(next_packages) == len(current_packages):
            return False
        self._set_packages(next_packages, scope)
        return True

    def list_configured_packages(self) -> list[ConfiguredPackage]:
        configured: list[ConfiguredPackage] = []
        for scope, settings in (
            ("user", self._settings_manager.get_global_settings()),
            ("project", self._settings_manager.get_project_settings()),
        ):
            for package in settings.get("packages") or []:
                source = self._get_package_source_string(package)
                installed = self.get_installed_path(source, scope)
                configured.append(
                    ConfiguredPackage(
                        source=source,
                        scope=scope,
                        installed_path=installed if installed and os.path.exists(installed) else None,
                    )
                )
        return configured

    def _collect_files_from_paths(self, paths: list[str], resource_type: str) -> list[str]:
        files: list[str] = []
        for path in paths:
            if not os.path.exists(path):
                continue
            try:
                if os.path.isfile(path):
                    files.append(path)
                elif os.path.isdir(path):
                    files.extend(_collect_resource_files(path, resource_type))
            except OSError:
                pass
        return files

    def _add_resource(
        self,
        target: dict[str, tuple[PathMetadata, bool]],
        path: str,
        metadata: PathMetadata,
        enabled: bool,
    ) -> None:
        if not path:
            return
        if path not in target:
            target[path] = (metadata, enabled)

    def _resolve_local_entries(
        self,
        entries: list[str],
        resource_type: str,
        target: dict[str, tuple[PathMetadata, bool]],
        metadata: PathMetadata,
        base_dir: str,
    ) -> None:
        if not entries:
            return

        # Collect all files from plain entries (non-pattern entries)
        plain, patterns = _split_patterns(entries)
        resolved_plain = [self._resolve_path_from_base(path, base_dir) for path in plain]
        all_files = self._collect_files_from_paths(resolved_plain, resource_type)

        # Determine which files are enabled based on patterns
        enabled_paths = apply_patterns(all_files, patterns, base_dir)

        for file in all_files:
            self._add_resource(target, file, metadata, file in enabled_paths)

    def _add_auto_discovered_resources(
        self,
        accumulator: dict[str, dict[str, tuple[PathMetadata, bool]]],
        global_settings: dict,
        project_settings: dict,
        global_base_dir: str,
        project_base_dir: str,
    ) -> None:
        user_metadata = PathMetadata(source="auto", scope="user", origin="top-level", base_dir=global_base_dir)
        project_metadata = PathMetadata(source="auto", scope="project", origin="top-level", base_dir=project_base_dir)

        user_overrides = {
            resource_type: list(global_settings.get(resource_type) or []) for resource_type in RESOURCE_TYPES
        }
        project_overrides = {
            resource_type: list(project_settings.get(resource_type) or []) for resource_type in RESOURCE_TYPES
        }

        user_dirs = {resource_type: os.path.join(global_base_dir, resource_type) for resource_type in RESOURCE_TYPES}
        project_dirs = {
            resource_type: os.path.join(project_base_dir, resource_type) for resource_type in RESOURCE_TYPES
        }
        user_agents_skills_dir = os.path.join(_get_home_dir(), ".agents", "skills")
        project_trusted = self._settings_manager.is_project_trusted()
        project_agents_skill_dirs = (
            [
                dir
                for dir in collect_ancestor_agents_skill_dirs(self._cwd)
                if os.path.abspath(dir) != os.path.abspath(user_agents_skills_dir)
            ]
            if project_trusted
            else []
        )

        def add_resources(
            resource_type: str,
            paths: list[str],
            metadata: PathMetadata,
            overrides: list[str],
            base_dir: str,
        ) -> None:
            target = accumulator[resource_type]
            for path in paths:
                enabled = is_enabled_by_overrides(path, overrides, base_dir)
                self._add_resource(target, path, metadata, enabled)

        if project_trusted:
            add_resources(
                "extensions",
                _collect_auto_extension_entries(project_dirs["extensions"]),
                project_metadata,
                project_overrides["extensions"],
                project_base_dir,
            )
            add_resources(
                "skills",
                _collect_skill_entries(project_dirs["skills"], "pi"),
                project_metadata,
                project_overrides["skills"],
                project_base_dir,
            )

        # Project skills from .agents/ (each with its own baseDir)
        for agents_skills_dir in project_agents_skill_dirs:
            agents_base_dir = os.path.dirname(agents_skills_dir)  # the .agents directory
            agents_metadata = PathMetadata(source="auto", scope="project", origin="top-level", base_dir=agents_base_dir)
            add_resources(
                "skills",
                _collect_skill_entries(agents_skills_dir, "agents"),
                agents_metadata,
                project_overrides["skills"],
                agents_base_dir,
            )

        if project_trusted:
            add_resources(
                "prompts",
                _collect_flat_entries(project_dirs["prompts"], ".md"),
                project_metadata,
                project_overrides["prompts"],
                project_base_dir,
            )
            add_resources(
                "themes",
                _collect_flat_entries(project_dirs["themes"], ".json"),
                project_metadata,
                project_overrides["themes"],
                project_base_dir,
            )

        add_resources(
            "extensions",
            _collect_auto_extension_entries(user_dirs["extensions"]),
            user_metadata,
            user_overrides["extensions"],
            global_base_dir,
        )
        add_resources(
            "skills",
            _collect_skill_entries(user_dirs["skills"], "pi"),
            user_metadata,
            user_overrides["skills"],
            global_base_dir,
        )

        # User skills from ~/.agents/ (with its own baseDir)
        user_agents_base_dir = os.path.dirname(user_agents_skills_dir)
        user_agents_metadata = PathMetadata(
            source="auto", scope="user", origin="top-level", base_dir=user_agents_base_dir
        )
        add_resources(
            "skills",
            _collect_skill_entries(user_agents_skills_dir, "agents"),
            user_agents_metadata,
            user_overrides["skills"],
            user_agents_base_dir,
        )

        add_resources(
            "prompts",
            _collect_flat_entries(user_dirs["prompts"], ".md"),
            user_metadata,
            user_overrides["prompts"],
            global_base_dir,
        )
        add_resources(
            "themes",
            _collect_flat_entries(user_dirs["themes"], ".json"),
            user_metadata,
            user_overrides["themes"],
            global_base_dir,
        )

    def _to_resolved_paths(self, accumulator: dict[str, dict[str, tuple[PathMetadata, bool]]]) -> ResolvedPaths:
        def map_to_resolved(entries: dict[str, tuple[PathMetadata, bool]]) -> list[ResolvedResource]:
            resolved = [
                ResolvedResource(path=path, enabled=enabled, metadata=metadata)
                for path, (metadata, enabled) in entries.items()
            ]
            resolved.sort(key=lambda entry: _resource_precedence_rank(entry.metadata))

            seen: set[str] = set()
            deduped: list[ResolvedResource] = []
            for entry in resolved:
                canonical_path = canonicalize_path(entry.path)
                if canonical_path in seen:
                    continue
                seen.add(canonical_path)
                deduped.append(entry)
            return deduped

        return ResolvedPaths(
            extensions=map_to_resolved(accumulator["extensions"]),
            skills=map_to_resolved(accumulator["skills"]),
            prompts=map_to_resolved(accumulator["prompts"]),
            themes=map_to_resolved(accumulator["themes"]),
        )

    def _create_accumulator(self) -> dict[str, dict[str, tuple[PathMetadata, bool]]]:
        return {resource_type: {} for resource_type in RESOURCE_TYPES}

    # -- package resources -------------------------------------------------------

    def _collect_files_from_manifest_entries(self, entries: list[str], root: str, resource_type: str) -> list[str]:
        source_entries = [entry for entry in entries if not _is_override_pattern(entry)]
        resolved: list[str] = []
        for entry in source_entries:
            if not _has_glob_pattern(entry):
                resolved.append(os.path.abspath(os.path.join(root, entry)))
                continue
            resolved.extend(
                os.path.abspath(os.path.join(root, match)) for match in glob.glob(entry, root_dir=root, recursive=True)
            )
        return self._collect_files_from_paths(resolved, resource_type)

    def _collect_manifest_files(self, package_root: str, resource_type: str) -> list[str]:
        manifest = read_pidrei_manifest(os.path.join(package_root, "pyproject.toml"))
        entries = (manifest or {}).get(resource_type)
        if entries:
            all_files = self._collect_files_from_manifest_entries(entries, package_root, resource_type)
            manifest_patterns = _get_override_patterns(entries)
            if manifest_patterns:
                enabled = apply_patterns(all_files, manifest_patterns, package_root)
                return [file for file in all_files if file in enabled]
            return all_files

        convention_dir = os.path.join(package_root, resource_type)
        if not os.path.exists(convention_dir):
            return []
        return _collect_resource_files(convention_dir, resource_type)

    def _add_manifest_entries(
        self,
        entries: list[str] | None,
        root: str,
        resource_type: str,
        target: dict[str, tuple[PathMetadata, bool]],
        metadata: PathMetadata,
    ) -> None:
        if not entries:
            return
        all_files = self._collect_files_from_manifest_entries(entries, root, resource_type)
        enabled_paths = apply_patterns(all_files, _get_override_patterns(entries), root)
        for file in all_files:
            if file in enabled_paths:
                self._add_resource(target, file, metadata, True)

    def _collect_default_resources(
        self,
        package_root: str,
        resource_type: str,
        target: dict[str, tuple[PathMetadata, bool]],
        metadata: PathMetadata,
    ) -> None:
        manifest = read_pidrei_manifest(os.path.join(package_root, "pyproject.toml"))
        entries = (manifest or {}).get(resource_type)
        if entries:
            self._add_manifest_entries(entries, package_root, resource_type, target, metadata)
            return
        directory = os.path.join(package_root, resource_type)
        if os.path.exists(directory):
            for file in _collect_resource_files(directory, resource_type):
                self._add_resource(target, file, metadata, True)

    def _apply_package_filter(
        self,
        package_root: str,
        user_patterns: list[str],
        resource_type: str,
        target: dict[str, tuple[PathMetadata, bool]],
        metadata: PathMetadata,
    ) -> None:
        all_files = self._collect_manifest_files(package_root, resource_type)
        if not user_patterns:
            # An explicitly empty list disables every resource of this type.
            for file in all_files:
                self._add_resource(target, file, metadata, False)
            return

        enabled_by_user = apply_patterns(all_files, user_patterns, package_root)
        for file in all_files:
            self._add_resource(target, file, metadata, file in enabled_by_user)

    def _apply_package_delta_filter(
        self,
        package_root: str,
        user_patterns: list[str],
        resource_type: str,
        target: dict[str, tuple[PathMetadata, bool]],
        metadata: PathMetadata,
    ) -> None:
        if not user_patterns:
            return
        all_files = self._collect_manifest_files(package_root, resource_type)
        for file_path, enabled in apply_autoload_disabled_patterns(all_files, user_patterns, package_root).items():
            self._add_resource(target, file_path, metadata, enabled)

    def _collect_package_resources(
        self,
        package_root: str,
        accumulator: dict[str, dict[str, tuple[PathMetadata, bool]]],
        filter: dict | None,
        metadata: PathMetadata,
    ) -> bool:
        if filter is not None:
            for resource_type in RESOURCE_TYPES:
                patterns = filter.get(resource_type)
                target = accumulator[resource_type]
                if filter.get("autoload") is False:
                    self._apply_package_delta_filter(package_root, patterns or [], resource_type, target, metadata)
                elif patterns is not None:
                    self._apply_package_filter(package_root, patterns, resource_type, target, metadata)
                else:
                    self._collect_default_resources(package_root, resource_type, target, metadata)
            return True

        manifest = read_pidrei_manifest(os.path.join(package_root, "pyproject.toml"))
        if manifest:
            for resource_type in RESOURCE_TYPES:
                self._add_manifest_entries(
                    manifest.get(resource_type), package_root, resource_type, accumulator[resource_type], metadata
                )
            return True

        has_any_dir = False
        for resource_type in RESOURCE_TYPES:
            directory = os.path.join(package_root, resource_type)
            if os.path.exists(directory):
                for file in _collect_resource_files(directory, resource_type):
                    self._add_resource(accumulator[resource_type], file, metadata, True)
                has_any_dir = True
        return has_any_dir

    def _resolve_local_extension_source(
        self,
        source: LocalSource,
        accumulator: dict[str, dict[str, tuple[PathMetadata, bool]]],
        filter: dict | None,
        metadata: PathMetadata,
        base_dir: str,
    ) -> None:
        resolved = self._resolve_path_from_base(source.path, base_dir)
        if not os.path.exists(resolved):
            return
        try:
            if os.path.isfile(resolved):
                metadata.base_dir = os.path.dirname(resolved)
                self._add_resource(accumulator["extensions"], resolved, metadata, True)
                return
            if os.path.isdir(resolved):
                metadata.base_dir = resolved
                if not self._collect_package_resources(resolved, accumulator, filter, metadata):
                    self._add_resource(accumulator["extensions"], resolved, metadata, True)
        except OSError:
            return

    # -- package sources ---------------------------------------------------------

    def _dedupe_packages(self, packages: list[tuple[Any, str]]) -> list[tuple[Any, str]]:
        result: list[tuple[Any, str]] = []
        seen: dict[str, int] = {}
        for package, scope in packages:
            identity = self._get_package_identity(self._get_package_source_string(package), scope)
            index = seen.get(identity)
            if index is None:
                seen[identity] = len(result)
                result.append((package, scope))
                continue
            existing_package, existing_scope = result[index]
            if existing_scope == "project" and scope == "user":
                # A project delta entry layers over the global package, so both stay.
                if isinstance(existing_package, dict) and existing_package.get("autoload") is False:
                    result.append((package, scope))
            elif scope == "project":
                result[index] = (package, scope)
        return result

    def _find_autoload_delta_base(
        self, package: Any, scope: str, sources: list[tuple[Any, str]]
    ) -> tuple[str, str] | None:
        if scope != "project" or not isinstance(package, dict) or package.get("autoload") is not False:
            return None
        identity = self._get_package_identity(package["source"], scope)
        for entry_package, entry_scope in sources:
            if entry_scope != "user":
                continue
            if self._get_package_identity(self._get_package_source_string(entry_package), "user") == identity:
                return self._get_package_source_string(entry_package), "user"
        return None

    async def _resolve_package_sources(
        self,
        sources: list[tuple[Any, str]],
        accumulator: dict[str, dict[str, tuple[PathMetadata, bool]]],
        on_missing=None,
    ) -> None:
        for package, scope in sources:
            source_str = self._get_package_source_string(package)
            filter = self._get_package_filter(package)
            delta_base = self._find_autoload_delta_base(package, scope, sources)
            resolved_source = delta_base[0] if delta_base else source_str
            resolved_scope = delta_base[1] if delta_base else scope
            parsed = self.parse_source(resolved_source)
            metadata = PathMetadata(source=source_str, scope=scope, origin="package")

            if isinstance(parsed, LocalSource):
                base_dir = self._get_base_dir_for_scope(resolved_scope)
                self._resolve_local_extension_source(parsed, accumulator, filter, metadata, base_dir)
                continue

            async def install_missing(parsed=parsed, resolved_source=resolved_source, scope=resolved_scope) -> bool:
                if is_offline_mode_enabled():
                    return False
                if on_missing is None:
                    await self._install_parsed_source(parsed, scope)
                    return True
                action = await on_missing(resolved_source)
                if action == "skip":
                    return False
                if action == "error":
                    raise Exception(f"Missing source: {resolved_source}")
                await self._install_parsed_source(parsed, scope)
                return True

            installed_path = self._get_git_install_path(parsed, resolved_scope)
            if not os.path.exists(installed_path):
                if not await install_missing():
                    continue
            elif resolved_scope == "temporary" and not parsed.pinned and not is_offline_mode_enabled():
                await self._refresh_temporary_git_source(parsed, resolved_source)
            metadata.base_dir = installed_path
            self._collect_package_resources(installed_path, accumulator, filter, metadata)

    async def resolve(self, on_missing=None) -> ResolvedPaths:
        accumulator = self._create_accumulator()
        global_settings = self._settings_manager.get_global_settings()
        project_settings = self._settings_manager.get_project_settings()

        # Project first, so cwd resources win collisions.
        all_packages: list[tuple[Any, str]] = [
            *[(package, "project") for package in (project_settings.get("packages") or [])],
            *[(package, "user") for package in (global_settings.get("packages") or [])],
        ]
        await self._resolve_package_sources(self._dedupe_packages(all_packages), accumulator, on_missing)

        global_base_dir = self._agent_dir
        project_base_dir = os.path.join(self._cwd, CONFIG_DIR_NAME)

        for resource_type in RESOURCE_TYPES:
            target = accumulator[resource_type]
            global_entries = list(global_settings.get(resource_type) or [])
            project_entries = list(project_settings.get(resource_type) or [])
            self._resolve_local_entries(
                project_entries,
                resource_type,
                target,
                PathMetadata(source="local", scope="project", origin="top-level"),
                project_base_dir,
            )
            self._resolve_local_entries(
                global_entries,
                resource_type,
                target,
                PathMetadata(source="local", scope="user", origin="top-level"),
                global_base_dir,
            )

        self._add_auto_discovered_resources(
            accumulator, global_settings, project_settings, global_base_dir, project_base_dir
        )

        return self._to_resolved_paths(accumulator)

    async def resolve_extension_sources(
        self, sources: list[str], *, local: bool = False, temporary: bool = False
    ) -> ResolvedPaths:
        """Resolve CLI-passed sources through the same path as configured
        packages, so a `--extension` pointing at a package directory picks up
        its skills and themes too."""
        accumulator = self._create_accumulator()
        scope = "temporary" if temporary else ("project" if local else "user")
        await self._resolve_package_sources([(source, scope) for source in sources], accumulator)
        return self._to_resolved_paths(accumulator)
