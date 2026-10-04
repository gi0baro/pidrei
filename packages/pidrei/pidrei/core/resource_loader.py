"""Mirror of pi coding-agent src/core/resource-loader.ts.

Loads extensions, skills, prompt templates, AGENTS.md context files, and
SYSTEM.md / APPEND_SYSTEM.md, with the project-trust bootstrap flow.
"""

import os
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, replace
from typing import Any

import tonio.colored as tonio
from tonio.colored import fs

from pidrei_ai.utils.tasks import gather
from pidrei_tui import detect_capabilities, get_terminal_color_mode

from ..config import APP_NAME, CONFIG_DIR_NAME
from ..utils.paths import canonicalize_path_blocking, is_local_path, resolve_path
from ..utils.text import strip_bom
from .diagnostics import ResourceCollision, ResourceDiagnostic
from .event_bus import EventBus
from .extensions.loader import (
    clear_extension_cache,
    create_extension_runtime,
    load_extension_from_factory,
    load_extensions_cached,
)
from .extensions.types import (
    Extension,
    ExtensionLoadError,
    ExtensionLoadWarning,
    ExtensionRuntime,
    LoadExtensionsResult,
)
from .footer_data_provider import _find_git_paths_blocking
from .output_guard import write_stderr
from .package_manager import DefaultPackageManager, ResolvedResource
from .prompt_templates import PromptTemplate, load_prompt_templates
from .settings_manager import SettingsManager
from .skills import LoadSkillsResult, Skill, load_skills
from .source_info import (
    BUILTIN_PATH_PREFIX,
    PathMetadata,
    SourceInfo,
    create_source_info,
    get_synthetic_path_source,
    is_synthetic_path,
)
from .timings import reset_timings


@dataclass(slots=True)
class LoadPromptsResult:
    prompts: list[PromptTemplate]
    diagnostics: list[ResourceDiagnostic]


@dataclass(slots=True)
class AgentsFile:
    path: str
    content: str


@dataclass(slots=True)
class SourcedPath:
    path: str
    metadata: PathMetadata


def _warn(message: str) -> None:
    write_stderr(f"\x1b[33mWarning: {message}\x1b[0m\n")


def _directories_among_blocking(paths: list[str]) -> frozenset[str]:
    return frozenset(path for path in paths if os.path.isdir(path))


def _skill_directories_among_blocking(paths: list[str]) -> frozenset[str]:
    """The paths that are a directory holding a `SKILL.md`."""
    return frozenset(path for path in paths if os.path.isdir(path) and os.path.exists(os.path.join(path, "SKILL.md")))


def _resolve_prompt_input(input: str | None, description: str) -> Awaitable[str | None]:
    """The value is either a literal prompt or a path to read — deciding which
    means touching the filesystem, so the whole check goes to the pool."""
    return tonio.spawn_blocking(_resolve_prompt_input_blocking, input, description)


def _resolve_prompt_input_blocking(input: str | None, description: str) -> str | None:
    if not input:
        return None

    if os.path.exists(input):
        try:
            with open(input, encoding="utf-8") as f:
                return strip_bom(f.read())
        except OSError as error:
            _warn(f"Could not read {description} file {input}: {error}")
            return input

    return input


def _load_context_file_from_dir_blocking(dir: str) -> AgentsFile | None:
    for filename in ("AGENTS.override.md", "AGENTS.md", "AGENTS.MD", "CLAUDE.md", "CLAUDE.MD"):
        file_path = os.path.join(dir, filename)
        if os.path.exists(file_path):
            try:
                if not os.path.isfile(file_path):
                    continue
                with open(file_path, encoding="utf-8") as f:
                    return AgentsFile(path=file_path, content=strip_bom(f.read()))
            except OSError as error:
                _warn(f"Could not read {file_path}: {error}")
    return None


def _find_shadowed_context_file_blocking(cwd: str) -> str | None:
    """The main repo's context file that a nested linked worktree's own copy
    shadows: both are the same tracked AGENTS.md/CLAUDE.md, so loading both
    loads it twice. Returns None when nothing is shadowed, leaving normal
    ancestor inheritance alone.

    Returned canonicalized (realpath), because `git worktree add` writes the
    `.git` file's `gitdir:` target in realpath form while cwd may still be
    symlinked (macOS `/tmp` -> `/private/tmp`).
    """
    git_paths = _find_git_paths_blocking(cwd)
    if git_paths is None:
        return None
    common_git_dir = canonicalize_path_blocking(git_paths["commonGitDir"])
    worktree_root = canonicalize_path_blocking(git_paths["repoDir"])
    main_repo_root = os.path.dirname(common_git_dir)
    # False for an ordinary repo, where the two are the same dir, and for a sibling
    # worktree (`git worktree add ../feat`), whose main repo is not an ancestor.
    if not worktree_root.startswith(f"{main_repo_root}{os.sep}"):
        return None
    # dirname of the common git dir is the main worktree root only when that dir is
    # itself checked out from the same repo. In a bare layout (`proj/.bare` +
    # `proj/main`) it is just the directory holding `.bare`, which tracks nothing; a
    # submodule's gitdir has no `commondir`, so it lands under `.git/modules`.
    if canonicalize_path_blocking(os.path.join(main_repo_root, ".git")) != common_git_dir:
        return None
    worktree_context_file = _load_context_file_from_dir_blocking(worktree_root)
    if worktree_context_file is None:
        return None
    return os.path.join(main_repo_root, os.path.basename(worktree_context_file.path))


def load_project_context_files(*, cwd: str, agent_dir: str) -> Awaitable[list[AgentsFile]]:
    """Walking cwd's ancestors for AGENTS.md is one blocking unit, so it goes
    to the pool whole rather than as a probe-and-read per directory."""
    return tonio.spawn_blocking(_load_project_context_files_blocking, cwd=cwd, agent_dir=agent_dir)


def _load_project_context_files_blocking(*, cwd: str, agent_dir: str) -> list[AgentsFile]:
    resolved_cwd = resolve_path(cwd)
    resolved_agent_dir = resolve_path(agent_dir)

    context_files: list[AgentsFile] = []
    seen_paths: set[str] = set()

    global_context = _load_context_file_from_dir_blocking(resolved_agent_dir)
    if global_context is not None:
        context_files.append(global_context)
        seen_paths.add(global_context.path)

    ancestor_context_files: list[AgentsFile] = []

    shadowed_context_file = _find_shadowed_context_file_blocking(resolved_cwd)
    current_dir = resolved_cwd
    while True:
        context_file = _load_context_file_from_dir_blocking(current_dir)
        is_shadowed = (
            shadowed_context_file is not None
            and context_file is not None
            and canonicalize_path_blocking(context_file.path) == shadowed_context_file
        )
        if context_file is not None and not is_shadowed and context_file.path not in seen_paths:
            ancestor_context_files.insert(0, context_file)
            seen_paths.add(context_file.path)

        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            break
        current_dir = parent_dir

    context_files.extend(ancestor_context_files)

    return context_files


def is_builtin_extension(entry: Any) -> bool:
    """Built-in extensions supply the code of `builtin:<name>` extension paths
    (an `InlineExtension` with `builtin=True`)."""
    return not callable(entry) and getattr(entry, "builtin", False) is True


def _merge_extension_warnings(result: LoadExtensionsResult, warnings: list[ExtensionLoadWarning]) -> None:
    """Append warnings, one per path (a later warning for a path replaces an earlier one)."""
    by_path = {warning.path: warning for warning in [*result.warnings, *warnings]}
    result.warnings = list(by_path.values())


def _omit_replaced_extensions(
    extensions: list[Extension], warnings: list[ExtensionLoadWarning] | None = None
) -> list[Extension]:
    """Leave out replaceable extensions (see `InlineExtension`) that share a
    tool, command, or flag name with another extension. For example, a
    third-party MCP extension that registers `/mcp` replaces the built-in MCP
    extension instead of both connecting the same servers."""

    def names(extension: Extension) -> list[str]:
        return [
            *(f"tool:{name}" for name in extension.tools),
            *(f"command:{name}" for name in extension.commands),
            *(f"flag:{name}" for name in extension.flags),
        ]

    taken: dict[str, Extension] = {}
    for extension in extensions:
        if not extension.replaceable:
            for name in names(extension):
                taken[name] = extension

    kept: list[Extension] = []
    for extension in extensions:
        replacement = (
            next(((name, taken[name]) for name in names(extension) if name in taken), None)
            if extension.replaceable
            else None
        )
        if replacement is None:
            kept.append(extension)
            continue
        if warnings is not None and extension.path.startswith(BUILTIN_PATH_PREFIX):
            builtin_name = extension.path[len(BUILTIN_PATH_PREFIX) :]
            name, owner = replacement
            kind, raw_name = name.split(":", 1)
            registered_name = f"/{raw_name}" if kind == "command" else f"--{raw_name}" if kind == "flag" else raw_name
            warnings.append(
                ExtensionLoadWarning(
                    path=extension.path,
                    warning=(
                        f"Extension {owner.path} registers {kind} `{registered_name}`, so built-in extension "
                        f"`{builtin_name}` was not loaded. To use `{builtin_name}`, run `{APP_NAME} config` and "
                        "make sure it is enabled under Built-in extensions, then disable or remove the existing "
                        "extension. We recommend only having one or the other loaded at a time."
                    ),
                )
            )
    return kept


class DefaultResourceLoader:
    def __init__(
        self,
        *,
        cwd: str,
        agent_dir: str,
        settings_manager: SettingsManager | None = None,
        event_bus: EventBus | None = None,
        extension_factories: list[Any] | None = None,
        additional_extension_paths: list[str] | None = None,
        additional_skill_paths: list[str] | None = None,
        additional_prompt_template_paths: list[str] | None = None,
        additional_theme_paths: list[str] | None = None,
        no_extensions: bool = False,
        no_skills: bool = False,
        no_prompt_templates: bool = False,
        no_themes: bool = False,
        no_context_files: bool = False,
        system_prompt: str | None = None,
        append_system_prompt: list[str] | None = None,
        extensions_override: Callable[[LoadExtensionsResult], LoadExtensionsResult] | None = None,
        skills_override: Callable[[LoadSkillsResult], LoadSkillsResult] | None = None,
        prompts_override: Callable[[LoadPromptsResult], LoadPromptsResult] | None = None,
        agents_files_override: Callable[[list[AgentsFile]], list[AgentsFile]] | None = None,
        system_prompt_override: Callable[[str | None], str | None] | None = None,
        append_system_prompt_override: Callable[[list[str]], list[str]] | None = None,
    ):
        self._cwd = resolve_path(cwd)
        self._agent_dir = resolve_path(agent_dir)
        # Both completed by `__await__` (`await DefaultResourceLoader(...)`):
        # without a settings manager passed in, loading one awaits, and the
        # package manager is built on it.
        self._settings_manager = settings_manager
        self._package_manager: DefaultPackageManager | None = None
        self._event_bus = event_bus if event_bus is not None else EventBus()
        factories = extension_factories or []
        self._extension_factories = [entry for entry in factories if not is_builtin_extension(entry)]
        # Built-in extensions supply the code of `builtin:<name>` extension paths.
        self._builtin_extensions: dict[str, Any] = {
            entry.name: entry for entry in factories if is_builtin_extension(entry)
        }
        self._additional_extension_paths = additional_extension_paths or []
        self._additional_skill_paths = additional_skill_paths or []
        self._additional_prompt_template_paths = additional_prompt_template_paths or []
        self._additional_theme_paths = additional_theme_paths or []
        self._no_extensions = no_extensions
        self._no_skills = no_skills
        self._no_prompt_templates = no_prompt_templates
        self._no_themes = no_themes
        self._no_context_files = no_context_files
        self._system_prompt_source = system_prompt
        self._append_system_prompt_source = append_system_prompt
        self._extensions_override = extensions_override
        self._skills_override = skills_override
        self._prompts_override = prompts_override
        self._agents_files_override = agents_files_override
        self._system_prompt_override = system_prompt_override
        self._append_system_prompt_override = append_system_prompt_override

        self._extensions_result = LoadExtensionsResult(runtime=create_extension_runtime())
        self._loaded = False
        self._skills: list[Skill] = []
        self._skill_diagnostics: list[ResourceDiagnostic] = []
        self._prompts: list[PromptTemplate] = []
        self._prompt_diagnostics: list[ResourceDiagnostic] = []
        self._agents_files: list[AgentsFile] = []
        self._system_prompt: str | None = None
        self._system_prompt_source_path: str | None = None
        self._append_system_prompt: list[str] = []
        self._append_system_prompt_source_paths: list[str] = []
        self._last_skill_paths: list[str] = []
        self._extension_skill_source_infos: dict[str, SourceInfo] = {}
        self._extension_prompt_source_infos: dict[str, SourceInfo] = {}
        self._last_prompt_paths: list[str] = []
        self._resource_metadata_by_path: dict[str, PathMetadata] = {}
        self._themes: list = []
        self._theme_diagnostics: list[ResourceDiagnostic] = []

    def __await__(self):
        return self._start().__await__()

    async def _start(self) -> DefaultResourceLoader:
        if self._settings_manager is None:
            self._settings_manager = await SettingsManager(self._cwd, self._agent_dir)
        if self._package_manager is None:
            self._package_manager = DefaultPackageManager(
                cwd=self._cwd,
                agent_dir=self._agent_dir,
                settings_manager=self._settings_manager,
                builtin_extensions=list(self._builtin_extensions),
            )
        return self

    # -- getters ---------------------------------------------------------------

    def get_extensions(self) -> LoadExtensionsResult:
        return self._extensions_result

    def get_skills(self) -> LoadSkillsResult:
        return LoadSkillsResult(skills=self._skills, diagnostics=self._skill_diagnostics)

    def get_prompts(self) -> LoadPromptsResult:
        return LoadPromptsResult(prompts=self._prompts, diagnostics=self._prompt_diagnostics)

    def get_themes(self) -> dict:
        """Loaded themes as a ``{"themes", "diagnostics"}`` record."""
        return {"themes": self._themes, "diagnostics": self._theme_diagnostics}

    def get_agents_files(self) -> list[AgentsFile]:
        return self._agents_files

    def get_system_prompt(self) -> str | None:
        return self._system_prompt

    def get_system_prompt_source(self) -> AgentsFile | None:
        """File-backed SYSTEM.md source, path-only (pi returns `{path}`)."""
        if self._system_prompt_source_path is None:
            return None
        return AgentsFile(path=self._system_prompt_source_path, content="")

    def get_append_system_prompt(self) -> list[str]:
        return self._append_system_prompt

    def get_append_system_prompt_sources(self) -> list[AgentsFile]:
        """File-backed APPEND_SYSTEM.md sources, path-only (pi returns `{path}`)."""
        return [AgentsFile(path=path, content="") for path in self._append_system_prompt_source_paths]

    # -- extension-provided resources -------------------------------------------

    async def extend_resources(
        self,
        *,
        skill_paths: list[SourcedPath] | None = None,
        prompt_paths: list[SourcedPath] | None = None,
    ) -> None:
        normalized_skills = self._normalize_extension_paths(skill_paths or [])
        normalized_prompts = self._normalize_extension_paths(prompt_paths or [])

        for entry in normalized_skills:
            self._extension_skill_source_infos[entry.path] = create_source_info(entry.path, entry.metadata)
        for entry in normalized_prompts:
            self._extension_prompt_source_infos[entry.path] = create_source_info(entry.path, entry.metadata)

        if normalized_skills:
            self._last_skill_paths = await self._merge_paths(
                self._last_skill_paths, [entry.path for entry in normalized_skills]
            )
            await self._update_skills_from_paths(self._last_skill_paths, self._resource_metadata_by_path)

        if normalized_prompts:
            self._last_prompt_paths = await self._merge_paths(
                self._last_prompt_paths, [entry.path for entry in normalized_prompts]
            )
            await self._update_prompts_from_paths(self._last_prompt_paths, self._resource_metadata_by_path)

    # -- reload ------------------------------------------------------------------

    async def load_project_trust_extensions(self) -> LoadExtensionsResult:
        # Force untrusted project settings for the bootstrap pass. This keeps project-local
        # extensions/packages out while still loading user/global and temporary CLI extensions.
        await self._settings_manager.set_project_trusted(False)
        await self._settings_manager.reload()
        return await self._load_current_extension_set(include_inline_factories=True)

    async def reload(
        self,
        *,
        resolve_project_trust: Callable[[LoadExtensionsResult], Awaitable[bool]] | None = None,
    ) -> None:
        reset_timings("extensions")

        if self._loaded:
            clear_extension_cache()

        pre_trust_extensions: LoadExtensionsResult | None = None
        if resolve_project_trust is not None:
            pre_trust_extensions = await self.load_project_trust_extensions()
            project_trusted = await resolve_project_trust(pre_trust_extensions)
            await self._settings_manager.set_project_trusted(project_trusted)

        # reload() preserves SettingsManager.project_trusted and reloads settings for that trust state.
        await self._settings_manager.reload()
        resolved_paths = await self._package_manager.resolve()
        cli_extension_paths = await self._package_manager.resolve_extension_sources(
            self._additional_extension_paths, temporary=True
        )
        # Kept on the instance so post-reload passes (extend_resources) can
        # still resolve package metadata.
        self._resource_metadata_by_path = {}
        metadata_by_path = self._resource_metadata_by_path

        self._extension_skill_source_infos = {}
        self._extension_prompt_source_infos = {}

        def get_enabled_resources(resources: list[ResolvedResource]) -> list[ResolvedResource]:
            for resource in resources:
                if resource.path not in metadata_by_path:
                    metadata_by_path[resource.path] = resource.metadata
            return [resource for resource in resources if resource.enabled]

        def get_enabled_paths(resources: list[ResolvedResource]) -> list[str]:
            return [resource.path for resource in get_enabled_resources(resources)]

        enabled_extensions = get_enabled_paths(resolved_paths.extensions)
        enabled_skill_resources = get_enabled_resources(resolved_paths.skills)
        enabled_prompts = get_enabled_paths(resolved_paths.prompts)

        enabled_skills = await self._map_skill_paths(enabled_skill_resources, metadata_by_path)

        for resource in [*cli_extension_paths.extensions, *cli_extension_paths.skills]:
            if resource.path not in metadata_by_path:
                metadata_by_path[resource.path] = PathMetadata(source="cli", scope="temporary", origin="top-level")

        cli_enabled_extensions = get_enabled_paths(cli_extension_paths.extensions)
        cli_enabled_skills = get_enabled_paths(cli_extension_paths.skills)
        cli_enabled_prompts = get_enabled_paths(cli_extension_paths.prompts)

        if self._no_extensions:
            extension_paths = cli_enabled_extensions
        else:
            extension_paths = await self._merge_paths(cli_enabled_extensions, enabled_extensions)

        extensions_result = await self._load_final_extension_set(extension_paths, pre_trust_extensions)
        for resolved in await self._missing_local_paths(self._additional_extension_paths):
            extensions_result.errors.append(
                ExtensionLoadError(path=resolved, error=f"Extension path does not exist: {resolved}")
            )
        self._extensions_result = (
            self._extensions_override(extensions_result) if self._extensions_override else extensions_result
        )
        await self._apply_extension_source_info(self._extensions_result.extensions, metadata_by_path)

        if self._no_skills:
            skill_paths = await self._merge_paths(cli_enabled_skills, self._additional_skill_paths)
        else:
            skill_paths = await self._merge_paths([*cli_enabled_skills, *enabled_skills], self._additional_skill_paths)

        self._last_skill_paths = skill_paths

        if self._no_prompt_templates:
            prompt_paths = await self._merge_paths(cli_enabled_prompts, self._additional_prompt_template_paths)
        else:
            prompt_paths = await self._merge_paths(
                [*cli_enabled_prompts, *enabled_prompts], self._additional_prompt_template_paths
            )
        self._last_prompt_paths = prompt_paths

        enabled_themes = get_enabled_paths(resolved_paths.themes)
        cli_enabled_themes = get_enabled_paths(cli_extension_paths.themes)
        if self._no_themes:
            theme_paths = await self._merge_paths(cli_enabled_themes, self._additional_theme_paths)
        else:
            theme_paths = await self._merge_paths([*cli_enabled_themes, *enabled_themes], self._additional_theme_paths)

        # Only extensions feed the stages below (skill/prompt source infos);
        # the stages themselves are independent and each writes its own
        # fields, so they run concurrently. `metadata_by_path` is complete by
        # now and only read from here on.
        await gather(
            self._load_skills_stage(skill_paths, metadata_by_path),
            self._load_prompts_stage(prompt_paths, metadata_by_path),
            self._update_themes_from_paths(theme_paths, metadata_by_path),
            self._load_agents_files_stage(),
            self._load_system_prompt_stage(),
            self._load_append_system_prompt_stage(),
        )
        self._loaded = True

    async def _load_skills_stage(self, skill_paths: list[str], metadata_by_path: dict[str, PathMetadata]) -> None:
        await self._update_skills_from_paths(skill_paths, metadata_by_path)
        for resolved in await self._missing_local_paths(self._additional_skill_paths):
            if not any(d.path == resolved for d in self._skill_diagnostics):
                self._skill_diagnostics.append(
                    ResourceDiagnostic(type="error", message="Skill path does not exist", path=resolved)
                )

    async def _load_prompts_stage(self, prompt_paths: list[str], metadata_by_path: dict[str, PathMetadata]) -> None:
        await self._update_prompts_from_paths(prompt_paths, metadata_by_path)
        for resolved in await self._missing_local_paths(self._additional_prompt_template_paths):
            if not any(d.path == resolved for d in self._prompt_diagnostics):
                self._prompt_diagnostics.append(
                    ResourceDiagnostic(type="error", message="Prompt template path does not exist", path=resolved)
                )

    async def _load_agents_files_stage(self) -> None:
        agents_files = (
            [] if self._no_context_files else await load_project_context_files(cwd=self._cwd, agent_dir=self._agent_dir)
        )
        self._agents_files = (
            self._agents_files_override(agents_files) if self._agents_files_override is not None else agents_files
        )

    async def _load_system_prompt_stage(self) -> None:
        system_prompt_source = (
            self._system_prompt_source
            if self._system_prompt_source is not None
            else await tonio.spawn_blocking(self._discover_system_prompt_file_blocking)
        )
        base_system_prompt = await _resolve_prompt_input(system_prompt_source, "system prompt")
        self._system_prompt = (
            self._system_prompt_override(base_system_prompt)
            if self._system_prompt_override is not None
            else base_system_prompt
        )
        self._system_prompt_source_path = (
            resolve_path(system_prompt_source)
            if system_prompt_source is not None and await fs.Path(system_prompt_source).exists()
            else None
        )

    async def _load_append_system_prompt_stage(self) -> None:
        if self._append_system_prompt_source is not None:
            append_sources = self._append_system_prompt_source
        else:
            discovered = await tonio.spawn_blocking(self._discover_append_system_prompt_file_blocking)
            append_sources = [discovered] if discovered is not None else []
        resolved = await gather(*(_resolve_prompt_input(source, "append system prompt") for source in append_sources))
        base_append = [text for text in resolved if text is not None]
        self._append_system_prompt = (
            self._append_system_prompt_override(base_append)
            if self._append_system_prompt_override is not None
            else base_append
        )
        found = await gather(*(fs.Path(source).exists() for source in append_sources))
        self._append_system_prompt_source_paths = [
            resolve_path(source) for source, exists in zip(append_sources, found, strict=True) if exists
        ]

    # -- extension loading -------------------------------------------------------

    async def _load_current_extension_set(self, *, include_inline_factories: bool) -> LoadExtensionsResult:
        resolved_paths = await self._package_manager.resolve()
        cli_extension_paths = await self._package_manager.resolve_extension_sources(
            self._additional_extension_paths, temporary=True
        )
        enabled = [resource.path for resource in resolved_paths.extensions if resource.enabled]
        cli_enabled = [resource.path for resource in cli_extension_paths.extensions if resource.enabled]
        # Built-in extensions wait for the final pass: project settings can disable them, and a loaded
        # extension cannot be unloaded.
        extension_paths = [
            path
            for path in (cli_enabled if self._no_extensions else await self._merge_paths(cli_enabled, enabled))
            if not path.startswith(BUILTIN_PATH_PREFIX)
        ]

        extensions_result = await load_extensions_cached(extension_paths, self._cwd, self._event_bus)
        if not include_inline_factories:
            return extensions_result

        inline_extensions, inline_errors = await self._load_extension_factories(extensions_result.runtime)
        extensions_result.extensions.extend(inline_extensions)
        extensions_result.errors.extend(inline_errors)
        replacement_warnings: list[ExtensionLoadWarning] = []
        extensions_result.extensions = _omit_replaced_extensions(extensions_result.extensions, replacement_warnings)
        _merge_extension_warnings(extensions_result, replacement_warnings)
        return extensions_result

    def _resolve_extension_load_path(self, path: str) -> str:
        return path if is_synthetic_path(path) else resolve_path(path, self._cwd, normalize_unicode_spaces=True)

    async def _load_extension_paths(
        self, paths: list[str], runtime: ExtensionRuntime | None = None
    ) -> LoadExtensionsResult:
        """Load extension paths: files from disk and `builtin:<name>` paths from the built-in extensions."""
        result = await load_extensions_cached(
            [path for path in paths if not path.startswith(BUILTIN_PATH_PREFIX)], self._cwd, self._event_bus, runtime
        )
        for path in paths:
            if not path.startswith(BUILTIN_PATH_PREFIX):
                continue
            builtin = self._builtin_extensions.get(path[len(BUILTIN_PATH_PREFIX) :])
            if builtin is None:
                result.errors.append(ExtensionLoadError(path=path, error=f"Unknown built-in extension: {path}"))
                continue
            try:
                extension = await load_extension_from_factory(
                    builtin.factory, self._cwd, self._event_bus, result.runtime, path
                )
            except Exception as error:
                result.errors.append(ExtensionLoadError(path=path, error=str(error)))
                continue
            extension.hidden = True
            extension.replaceable = getattr(builtin, "replaceable", False) is True
            result.extensions.append(extension)
        return result

    async def _load_final_extension_set(
        self, extension_paths: list[str], pre_trust_extensions: LoadExtensionsResult | None
    ) -> LoadExtensionsResult:
        # Without a pre-trust pass nothing is preloaded, and inline extensions load here.
        # The bootstrap pass already ran its factories; re-running them would
        # double every registration, so only the paths it did not reach load now.
        preloaded = pre_trust_extensions.extensions if pre_trust_extensions is not None else []
        preloaded_by_path = {
            extension.resolved_path: extension for extension in preloaded if not extension.path.startswith("<inline:")
        }
        failed_preload_paths = {
            self._resolve_extension_load_path(error.path)
            for error in (pre_trust_extensions.errors if pre_trust_extensions is not None else [])
        }
        remaining_paths = [
            path
            for path in extension_paths
            if self._resolve_extension_load_path(path) not in preloaded_by_path
            and self._resolve_extension_load_path(path) not in failed_preload_paths
        ]
        remaining = await self._load_extension_paths(
            remaining_paths, pre_trust_extensions.runtime if pre_trust_extensions is not None else None
        )
        loaded_by_path = dict(preloaded_by_path)
        for extension in remaining.extensions:
            loaded_by_path[extension.resolved_path] = extension

        if pre_trust_extensions is not None:
            inline_extensions = [extension for extension in preloaded if extension.path.startswith("<inline:")]
            inline_errors: list[ExtensionLoadError] = []
        else:
            inline_extensions, inline_errors = await self._load_extension_factories(remaining.runtime)
        ordered = [
            extension
            for path in extension_paths
            if (extension := loaded_by_path.get(self._resolve_extension_load_path(path))) is not None
        ]
        ordered.extend(inline_extensions)

        replacement_warnings: list[ExtensionLoadWarning] = []
        extensions_result = LoadExtensionsResult(
            extensions=_omit_replaced_extensions(ordered, replacement_warnings),
            errors=[
                *(pre_trust_extensions.errors if pre_trust_extensions is not None else []),
                *remaining.errors,
                *inline_errors,
            ],
            warnings=[
                *(pre_trust_extensions.warnings if pre_trust_extensions is not None else []),
                *remaining.warnings,
            ],
            runtime=remaining.runtime,
        )
        _merge_extension_warnings(extensions_result, replacement_warnings)
        self._add_extension_conflict_diagnostics(extensions_result)
        return extensions_result

    async def _load_extension_factories(
        self, runtime: ExtensionRuntime
    ) -> tuple[list[Extension], list[ExtensionLoadError]]:
        extensions: list[Extension] = []
        errors: list[ExtensionLoadError] = []

        for index, entry in enumerate(self._extension_factories):
            named = not callable(entry)
            factory = entry.factory if named else entry
            extension_path = f"<inline:{entry.name if named else index + 1}>"
            try:
                extension = await load_extension_from_factory(
                    factory, self._cwd, self._event_bus, runtime, extension_path
                )
                extension.hidden = bool(named and getattr(entry, "hidden", False))
                extension.replaceable = named and getattr(entry, "replaceable", False) is True
                extensions.append(extension)
            except Exception as error:
                errors.append(ExtensionLoadError(path=extension_path, error=str(error)))

        return extensions, errors

    def _add_extension_conflict_diagnostics(self, extensions_result: LoadExtensionsResult) -> None:
        """Conflicts are reported, never resolved: every extension stays
        loaded and load order decides precedence."""
        for path, message in self._detect_extension_conflicts(extensions_result.extensions):
            extensions_result.errors.append(ExtensionLoadError(path=path, error=message))

    def _detect_extension_conflicts(self, extensions: list[Extension]) -> list[tuple[str, str]]:
        conflicts: list[tuple[str, str]] = []
        tool_owners: dict[str, str] = {}
        flag_owners: dict[str, str] = {}

        for extension in extensions:
            for tool_name in extension.tools:
                owner = tool_owners.get(tool_name)
                if owner is not None and owner != extension.path:
                    conflicts.append((extension.path, f'Tool "{tool_name}" conflicts with {owner}'))
                else:
                    tool_owners[tool_name] = extension.path

            for flag_name in extension.flags:
                owner = flag_owners.get(flag_name)
                if owner is not None and owner != extension.path:
                    conflicts.append((extension.path, f'Flag "--{flag_name}" conflicts with {owner}'))
                else:
                    flag_owners[flag_name] = extension.path

        return conflicts

    async def _apply_extension_source_info(
        self, extensions: list[Extension], metadata_by_path: dict[str, PathMetadata]
    ) -> None:
        directories = await self._directories_among(extension.path for extension in extensions)
        for extension in extensions:
            source_info = self._find_source_info_for_path(
                extension.path, None, metadata_by_path, directories
            ) or self._default_source_info_for_path(extension.path, directories)
            extension.source_info = source_info
            for command in extension.commands.values():
                command.source_info = source_info
            for tool in extension.tools.values():
                tool.source_info = source_info

    # -- helpers ----------------------------------------------------------------

    async def _map_skill_paths(
        self, resources: list[ResolvedResource], metadata_by_path: dict[str, PathMetadata]
    ) -> list[str]:
        """An auto-discovered or package skill that is a directory with a
        `SKILL.md` is named by that file. The filesystem is asked once, for
        all of them."""

        def is_candidate(resource: ResolvedResource) -> bool:
            return resource.metadata.source == "auto" or resource.metadata.origin == "package"

        candidates = [resource.path for resource in resources if is_candidate(resource)]
        skill_dirs = (
            await tonio.spawn_blocking(_skill_directories_among_blocking, candidates) if candidates else frozenset()
        )
        paths: list[str] = []
        for resource in resources:
            if not is_candidate(resource) or resource.path not in skill_dirs:
                paths.append(resource.path)
                continue
            skill_file = os.path.join(resource.path, "SKILL.md")
            if skill_file not in metadata_by_path:
                metadata_by_path[skill_file] = resource.metadata
            paths.append(skill_file)
        return paths

    async def _directories_among(self, paths: Iterable[str]) -> frozenset[str]:
        """Which of `paths` are directories, asked once: what the default
        source info of a path outside the known roots depends on."""
        candidates = [os.path.abspath(path) for path in paths if path]
        if not candidates:
            return frozenset()
        return await tonio.spawn_blocking(_directories_among_blocking, candidates)

    async def _missing_local_paths(self, paths: list[str]) -> list[str]:
        """The local paths among `paths` that do not exist, resolved, in their order."""
        resolved = [self._resolve_resource_path(path) for path in paths if is_local_path(path)]
        found = await gather(*(fs.Path(path).exists() for path in resolved))
        return [path for path, exists in zip(resolved, found, strict=True) if not exists]

    def _normalize_extension_paths(self, entries: list[SourcedPath]) -> list[SourcedPath]:
        normalized: list[SourcedPath] = []
        for entry in entries:
            metadata = entry.metadata
            if metadata.base_dir:
                metadata = replace(metadata, base_dir=self._resolve_resource_path(metadata.base_dir))
            normalized.append(SourcedPath(path=self._resolve_resource_path(entry.path), metadata=metadata))
        return normalized

    async def _update_skills_from_paths(
        self, skill_paths: list[str], metadata_by_path: dict[str, PathMetadata] | None = None
    ) -> None:
        if self._no_skills and not skill_paths:
            skills_result = LoadSkillsResult(skills=[], diagnostics=[])
        else:
            skills_result = await load_skills(
                cwd=self._cwd,
                agent_dir=self._agent_dir,
                skill_paths=skill_paths,
                include_defaults=False,
            )
        resolved_skills = self._skills_override(skills_result) if self._skills_override is not None else skills_result
        directories = await self._directories_among(skill.file_path for skill in resolved_skills.skills)
        self._skills = [
            replace(
                skill,
                source_info=(
                    self._find_source_info_for_path(
                        skill.file_path, self._extension_skill_source_infos, metadata_by_path, directories
                    )
                    or skill.source_info
                    or self._default_source_info_for_path(skill.file_path, directories)
                ),
            )
            for skill in resolved_skills.skills
        ]
        self._skill_diagnostics = resolved_skills.diagnostics

    async def _update_prompts_from_paths(
        self, prompt_paths: list[str], metadata_by_path: dict[str, PathMetadata] | None = None
    ) -> None:
        if self._no_prompt_templates and not prompt_paths:
            prompts_result = LoadPromptsResult(prompts=[], diagnostics=[])
        else:
            loaded = await load_prompt_templates(
                cwd=self._cwd,
                agent_dir=self._agent_dir,
                prompt_paths=prompt_paths,
                include_defaults=False,
            )
            deduped = self._dedupe_prompts(loaded.templates)
            prompts_result = LoadPromptsResult(
                prompts=deduped.prompts, diagnostics=[*loaded.diagnostics, *deduped.diagnostics]
            )
        resolved_prompts = (
            self._prompts_override(prompts_result) if self._prompts_override is not None else prompts_result
        )
        directories = await self._directories_among(prompt.file_path for prompt in resolved_prompts.prompts)
        self._prompts = [
            replace(
                prompt,
                source_info=(
                    self._find_source_info_for_path(
                        prompt.file_path, self._extension_prompt_source_infos, metadata_by_path, directories
                    )
                    or prompt.source_info
                    or self._default_source_info_for_path(prompt.file_path, directories)
                ),
            )
            for prompt in resolved_prompts.prompts
        ]
        self._prompt_diagnostics = resolved_prompts.diagnostics

    async def _update_themes_from_paths(
        self, theme_paths: list[str], metadata_by_path: dict[str, PathMetadata] | None = None
    ) -> None:
        themes, diagnostics = await tonio.spawn_blocking(self._load_themes_from_paths_blocking, theme_paths)
        directories = await self._directories_among(loaded_theme.source_path for loaded_theme in themes)
        for loaded_theme in themes:
            source_path = loaded_theme.source_path
            if source_path:
                loaded_theme.source_info = self._find_source_info_for_path(
                    source_path, None, metadata_by_path, directories
                ) or self._default_source_info_for_path(source_path, directories)
        self._themes = themes
        self._theme_diagnostics = diagnostics

    def _load_themes_from_paths_blocking(self, theme_paths: list[str]) -> tuple[list, list[ResourceDiagnostic]]:
        # lazy: core <-> modes import cycle (see modes/__init__.py)
        # This whole method runs pool-side, so it uses the blocking loader
        # directly rather than the awaitable one.
        from ..modes.interactive.theme import _load_theme_from_path_blocking as load_theme_from_path

        if self._no_themes and not theme_paths:
            themes: list = []
            diagnostics: list[ResourceDiagnostic] = []
        else:
            themes = []
            diagnostics = []
            # Theme construction only needs trueColor, so skip the unrelated tmux hyperlink probe.
            color_mode = get_terminal_color_mode(
                {
                    **detect_capabilities(lambda: False),
                    **self._settings_manager.get_terminal_capability_overrides(),
                }
            )

            def load_theme(file_path: str):
                return load_theme_from_path(file_path, color_mode)

            # pi's loadThemes(themePaths, false): the default directories arrive through
            # the package manager's auto-discovery, which gates project themes on trust
            # and applies settings overrides; scanning them here would bypass both.
            for path in theme_paths:
                resolved = self._resolve_resource_path(path)
                if not os.path.exists(resolved):
                    diagnostics.append(
                        ResourceDiagnostic(type="warning", message="theme path does not exist", path=resolved)
                    )
                    continue
                if os.path.isdir(resolved):
                    self._load_themes_from_dir_blocking(resolved, themes, diagnostics, load_theme)
                else:
                    self._load_theme_from_file(resolved, themes, diagnostics, load_theme)

            deduped_themes, dedupe_diagnostics = self._dedupe_themes(themes)
            themes = deduped_themes
            diagnostics = [*diagnostics, *dedupe_diagnostics]

        return themes, diagnostics

    def _load_themes_from_dir_blocking(
        self, theme_dir: str, themes: list, diagnostics: list, load_theme_from_path
    ) -> None:
        if not os.path.exists(theme_dir):
            return

        try:
            for entry in sorted(os.listdir(theme_dir)):
                full_path = os.path.join(theme_dir, entry)
                if not os.path.isfile(full_path):
                    continue
                if not entry.endswith(".json"):
                    continue
                self._load_theme_from_file(full_path, themes, diagnostics, load_theme_from_path)
        except OSError as error:
            diagnostics.append(ResourceDiagnostic(type="warning", message=str(error), path=theme_dir))

    def _load_theme_from_file(self, file_path: str, themes: list, diagnostics: list, load_theme_from_path) -> None:
        try:
            themes.append(load_theme_from_path(file_path))
        except Exception as error:
            diagnostics.append(ResourceDiagnostic(type="warning", message=str(error), path=file_path))

    def _dedupe_themes(self, themes: list) -> tuple[list, list]:
        seen: dict = {}
        diagnostics: list[ResourceDiagnostic] = []

        for t in themes:
            name = t.name if t.name is not None else "unnamed"
            existing = seen.get(name)
            if existing is not None:
                diagnostics.append(
                    ResourceDiagnostic(
                        type="collision",
                        message=f'name "{name}" collision',
                        path=t.source_path,
                    )
                )
            else:
                seen[name] = t

        return list(seen.values()), diagnostics

    def _find_source_info_for_path(
        self,
        resource_path: str,
        extra_source_infos: dict[str, SourceInfo] | None,
        metadata_by_path: dict[str, PathMetadata] | None,
        directories: frozenset[str],
    ) -> SourceInfo | None:
        """`directories` is what `_directories_among` found for the caller's paths."""
        if not resource_path:
            return None

        if is_synthetic_path(resource_path):
            return self._default_source_info_for_path(resource_path, directories)

        normalized_resource_path = os.path.abspath(resource_path)
        if extra_source_infos:
            for source_path, source_info in extra_source_infos.items():
                normalized_source_path = os.path.abspath(source_path)
                if normalized_resource_path == normalized_source_path or normalized_resource_path.startswith(
                    f"{normalized_source_path}{os.sep}"
                ):
                    return replace(source_info, path=resource_path)

        if metadata_by_path:
            exact = metadata_by_path.get(normalized_resource_path) or metadata_by_path.get(resource_path)
            if exact is not None:
                return create_source_info(resource_path, exact)

            for source_path, metadata in metadata_by_path.items():
                normalized_source_path = os.path.abspath(source_path)
                if normalized_resource_path == normalized_source_path or normalized_resource_path.startswith(
                    f"{normalized_source_path}{os.sep}"
                ):
                    return create_source_info(resource_path, metadata)

        return None

    def _default_source_info_for_path(self, file_path: str, directories: frozenset[str]) -> SourceInfo:
        synthetic_source = get_synthetic_path_source(file_path)
        if synthetic_source:
            return SourceInfo(path=file_path, source=synthetic_source, scope="temporary", origin="top-level")

        normalized_path = os.path.abspath(file_path)
        agent_roots = [os.path.join(self._agent_dir, name) for name in ("skills", "prompts", "themes", "extensions")]
        project_roots = [
            os.path.join(self._cwd, CONFIG_DIR_NAME, name) for name in ("skills", "prompts", "themes", "extensions")
        ]

        for root in agent_roots:
            if self._is_under_path(normalized_path, root):
                return SourceInfo(path=file_path, source="local", scope="user", origin="top-level", base_dir=root)

        for root in project_roots:
            if self._is_under_path(normalized_path, root):
                return SourceInfo(path=file_path, source="local", scope="project", origin="top-level", base_dir=root)

        return SourceInfo(
            path=file_path,
            source="local",
            scope="temporary",
            origin="top-level",
            base_dir=normalized_path if normalized_path in directories else os.path.dirname(normalized_path),
        )

    async def _merge_paths(self, primary: list[str], additional: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()

        resolved_paths = [self._resolve_resource_path(path) for path in [*primary, *additional]]
        canonical_paths = await tonio.spawn_blocking(
            lambda: [canonicalize_path_blocking(path) for path in resolved_paths]
        )
        for resolved, canonical_path in zip(resolved_paths, canonical_paths, strict=True):
            if canonical_path in seen:
                continue
            seen.add(canonical_path)
            merged.append(resolved)

        return merged

    def _resolve_resource_path(self, path: str) -> str:
        return path if is_synthetic_path(path) else resolve_path(path, self._cwd, trim=True)

    def _dedupe_prompts(self, prompts: list[PromptTemplate]) -> LoadPromptsResult:
        seen: dict[str, PromptTemplate] = {}
        diagnostics: list[ResourceDiagnostic] = []

        for prompt in prompts:
            existing = seen.get(prompt.name)
            if existing is not None:
                diagnostics.append(
                    ResourceDiagnostic(
                        type="collision",
                        message=f'name "/{prompt.name}" collision',
                        path=prompt.file_path,
                        collision=ResourceCollision(
                            resource_type="prompt",
                            name=prompt.name,
                            winner_path=existing.file_path,
                            loser_path=prompt.file_path,
                        ),
                    )
                )
            else:
                seen[prompt.name] = prompt

        return LoadPromptsResult(prompts=list(seen.values()), diagnostics=diagnostics)

    def _discover_system_prompt_file_blocking(self) -> str | None:
        project_path = os.path.join(self._cwd, CONFIG_DIR_NAME, "SYSTEM.md")
        if self._settings_manager.is_project_trusted() and os.path.exists(project_path):
            return project_path

        global_path = os.path.join(self._agent_dir, "SYSTEM.md")
        if os.path.exists(global_path):
            return global_path

        return None

    def _discover_append_system_prompt_file_blocking(self) -> str | None:
        project_path = os.path.join(self._cwd, CONFIG_DIR_NAME, "APPEND_SYSTEM.md")
        if self._settings_manager.is_project_trusted() and os.path.exists(project_path):
            return project_path

        global_path = os.path.join(self._agent_dir, "APPEND_SYSTEM.md")
        if os.path.exists(global_path):
            return global_path

        return None

    def _is_under_path(self, target: str, root: str) -> bool:
        normalized_root = os.path.abspath(root)
        if target == normalized_root:
            return True
        prefix = normalized_root if normalized_root.endswith(os.sep) else f"{normalized_root}{os.sep}"
        return target.startswith(prefix)
