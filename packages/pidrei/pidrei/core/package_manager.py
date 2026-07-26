"""Local-resource subset of pi coding-agent src/core/package-manager.ts.

Ports the resolution of *local* resources: settings-configured entries with
include/exclude patterns, and auto-discovery from the user/project resource
directories (extensions/, skills/, prompts/, themes/, plus ancestor
.agents/skills dirs). npm/git package installation, pi-manifest handling, and
the extension temp folder are Phase 5; configured `packages` entries are
ignored here (documented deviation) and `resolve()` covers everything pi's
DefaultPackageManager.resolve() does for a package-free configuration.
"""

import os
import re
from dataclasses import dataclass, field

from ..config import CONFIG_DIR_NAME
from ..utils.paths import canonicalize_path, resolve_path
from .settings_manager import SettingsManager
from .skills import IgnoreMatcher, add_ignore_rules
from .source_info import PathMetadata


RESOURCE_TYPES = ("extensions", "skills", "prompts", "themes")

_FILE_PATTERNS: dict[str, re.Pattern] = {
    "extensions": re.compile(r"\.(ts|js|py)$"),
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


def _get_override_patterns(entries: list[str]) -> list[str]:
    return [pattern for pattern in entries if pattern.startswith(("!", "+", "-"))]


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
    """Auto-discovery of extension entry files (extension loading is Phase 5;
    discovery keeps enable/disable bookkeeping consistent)."""
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
        is_dir, is_file = kind

        rel_path = os.path.relpath(full_path, dir)
        ignore_path = f"{rel_path}/" if is_dir else rel_path
        if ig.ignores(ignore_path):
            continue

        if is_file and _FILE_PATTERNS["extensions"].search(entry.name):
            entries.append(full_path)

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
    """Local-resource resolver (see module docstring for the Phase 5 gaps)."""

    def __init__(self, *, cwd: str, agent_dir: str, settings_manager: SettingsManager):
        self._cwd = resolve_path(cwd)
        self._agent_dir = resolve_path(agent_dir)
        self._settings_manager = settings_manager

    def _resolve_path_from_base(self, input: str, base_dir: str) -> str:
        return resolve_path(input, base_dir, home_dir=_get_home_dir(), trim=True)

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

    async def resolve(self) -> ResolvedPaths:
        accumulator = self._create_accumulator()
        global_settings = self._settings_manager.get_global_settings()
        project_settings = self._settings_manager.get_project_settings()

        # Configured `packages` sources are Phase 5 (npm/git installation).

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
        """Resolve CLI-passed sources. Only local paths are supported for now
        (npm/git sources are Phase 5)."""
        accumulator = self._create_accumulator()
        scope = "temporary" if temporary else ("project" if local else "user")
        metadata = PathMetadata(source="cli" if temporary else "local", scope=scope, origin="top-level")

        for source in sources:
            resolved = self._resolve_path_from_base(source, self._cwd)
            if not os.path.exists(resolved):
                continue
            if os.path.isdir(resolved):
                for file in _collect_auto_extension_entries(resolved):
                    self._add_resource(accumulator["extensions"], file, metadata, True)
            elif _FILE_PATTERNS["extensions"].search(resolved):
                self._add_resource(accumulator["extensions"], resolved, metadata, True)

        return self._to_resolved_paths(accumulator)
