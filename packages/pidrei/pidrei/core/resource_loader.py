"""Mirror of pi coding-agent src/core/resource-loader.ts.

Loads extensions, skills, prompt templates, AGENTS.md context files, and
SYSTEM.md / APPEND_SYSTEM.md, with the project-trust bootstrap flow.
"""

import os
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any

import tonio.colored as tonio
from tonio.colored import fs

from ..config import CONFIG_DIR_NAME
from ..utils.paths import canonicalize_path, is_local_path, resolve_path
from .diagnostics import ResourceCollision, ResourceDiagnostic
from .event_bus import EventBus
from .extensions.loader import (
    clear_extension_cache,
    create_extension_runtime,
    load_extension_from_factory,
    load_extensions_cached,
)
from .extensions.types import Extension, ExtensionLoadError, ExtensionRuntime, LoadExtensionsResult
from .footer_data_provider import _find_git_paths
from .package_manager import DefaultPackageManager, ResolvedResource
from .prompt_templates import PromptTemplate, load_prompt_templates
from .settings_manager import SettingsManager
from .skills import LoadSkillsResult, Skill, load_skills
from .source_info import PathMetadata, SourceInfo, create_source_info
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
    print(f"\x1b[33mWarning: {message}\x1b[0m", file=sys.stderr)


def _resolve_prompt_input(input: str | None, description: str) -> Awaitable[str | None]:
    """The value is either a literal prompt or a path to read — deciding which
    means touching the filesystem, so the whole check goes to the pool."""
    return tonio.spawn_blocking(_resolve_prompt_input_sync, input, description)


def _resolve_prompt_input_sync(input: str | None, description: str) -> str | None:
    if not input:
        return None

    if os.path.exists(input):
        try:
            with open(input, encoding="utf-8") as f:
                return f.read()
        except OSError as error:
            _warn(f"Could not read {description} file {input}: {error}")
            return input

    return input


def _load_context_file_from_dir(dir: str) -> AgentsFile | None:
    for filename in ("AGENTS.override.md", "AGENTS.md", "AGENTS.MD", "CLAUDE.md", "CLAUDE.MD"):
        file_path = os.path.join(dir, filename)
        if os.path.exists(file_path):
            try:
                if not os.path.isfile(file_path):
                    continue
                with open(file_path, encoding="utf-8") as f:
                    return AgentsFile(path=file_path, content=f.read())
            except OSError as error:
                _warn(f"Could not read {file_path}: {error}")
    return None


def _find_shadowed_context_file(cwd: str) -> str | None:
    """The main repo's context file that a nested linked worktree's own copy
    shadows: both are the same tracked AGENTS.md/CLAUDE.md, so loading both
    loads it twice. Returns None when nothing is shadowed, leaving normal
    ancestor inheritance alone.

    Returned canonicalized (realpath), because `git worktree add` writes the
    `.git` file's `gitdir:` target in realpath form while cwd may still be
    symlinked (macOS `/tmp` -> `/private/tmp`).
    """
    git_paths = _find_git_paths(cwd)
    if git_paths is None:
        return None
    common_git_dir = canonicalize_path(git_paths["commonGitDir"])
    worktree_root = canonicalize_path(git_paths["repoDir"])
    main_repo_root = os.path.dirname(common_git_dir)
    # False for an ordinary repo, where the two are the same dir, and for a sibling
    # worktree (`git worktree add ../feat`), whose main repo is not an ancestor.
    if not worktree_root.startswith(f"{main_repo_root}{os.sep}"):
        return None
    # dirname of the common git dir is the main worktree root only when that dir is
    # itself checked out from the same repo. In a bare layout (`proj/.bare` +
    # `proj/main`) it is just the directory holding `.bare`, which tracks nothing; a
    # submodule's gitdir has no `commondir`, so it lands under `.git/modules`.
    if canonicalize_path(os.path.join(main_repo_root, ".git")) != common_git_dir:
        return None
    worktree_context_file = _load_context_file_from_dir(worktree_root)
    if worktree_context_file is None:
        return None
    return os.path.join(main_repo_root, os.path.basename(worktree_context_file.path))


def load_project_context_files(*, cwd: str, agent_dir: str) -> Awaitable[list[AgentsFile]]:
    """Walking cwd's ancestors for AGENTS.md is one blocking unit, so it goes
    to the pool whole rather than as a probe-and-read per directory."""
    return tonio.spawn_blocking(_load_project_context_files_sync, cwd=cwd, agent_dir=agent_dir)


def _load_project_context_files_sync(*, cwd: str, agent_dir: str) -> list[AgentsFile]:
    resolved_cwd = resolve_path(cwd)
    resolved_agent_dir = resolve_path(agent_dir)

    context_files: list[AgentsFile] = []
    seen_paths: set[str] = set()

    global_context = _load_context_file_from_dir(resolved_agent_dir)
    if global_context is not None:
        context_files.append(global_context)
        seen_paths.add(global_context.path)

    ancestor_context_files: list[AgentsFile] = []

    shadowed_context_file = _find_shadowed_context_file(resolved_cwd)
    current_dir = resolved_cwd
    while True:
        context_file = _load_context_file_from_dir(current_dir)
        is_shadowed = (
            shadowed_context_file is not None
            and context_file is not None
            and canonicalize_path(context_file.path) == shadowed_context_file
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
        self._settings_manager = (
            settings_manager
            if settings_manager is not None
            # A constructor cannot await. Every production caller passes one in
            # (`agent_session_services`, `package_commands`, `sdk`); this
            # fallback exists for tests and direct SDK use.
            else SettingsManager.create_sync(self._cwd, self._agent_dir)
        )
        self._package_manager = DefaultPackageManager(
            cwd=self._cwd, agent_dir=self._agent_dir, settings_manager=self._settings_manager
        )
        self._event_bus = event_bus if event_bus is not None else EventBus()
        self._extension_factories = extension_factories or []
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
            self._last_skill_paths = self._merge_paths(
                self._last_skill_paths, [entry.path for entry in normalized_skills]
            )
            await self._update_skills_from_paths(self._last_skill_paths, self._resource_metadata_by_path)

        if normalized_prompts:
            self._last_prompt_paths = self._merge_paths(
                self._last_prompt_paths, [entry.path for entry in normalized_prompts]
            )
            await self._update_prompts_from_paths(self._last_prompt_paths, self._resource_metadata_by_path)

    # -- reload ------------------------------------------------------------------

    async def load_project_trust_extensions(self) -> LoadExtensionsResult:
        # Force untrusted project settings for the bootstrap pass. This keeps project-local
        # extensions/packages out while still loading user/global and temporary CLI extensions.
        self._settings_manager.set_project_trusted(False)
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
            self._settings_manager.set_project_trusted(project_trusted)

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

        enabled_skills = [self._map_skill_path(resource, metadata_by_path) for resource in enabled_skill_resources]

        for resource in [*cli_extension_paths.extensions, *cli_extension_paths.skills]:
            if resource.path not in metadata_by_path:
                metadata_by_path[resource.path] = PathMetadata(source="cli", scope="temporary", origin="top-level")

        cli_enabled_extensions = get_enabled_paths(cli_extension_paths.extensions)
        cli_enabled_skills = get_enabled_paths(cli_extension_paths.skills)
        cli_enabled_prompts = get_enabled_paths(cli_extension_paths.prompts)

        if self._no_extensions:
            extension_paths = cli_enabled_extensions
        else:
            extension_paths = self._merge_paths(cli_enabled_extensions, enabled_extensions)

        extensions_result = await self._load_final_extension_set(extension_paths, pre_trust_extensions)
        for path in self._additional_extension_paths:
            if is_local_path(path):
                resolved = self._resolve_resource_path(path)
                if not await fs.Path(resolved).exists():
                    extensions_result.errors.append(
                        ExtensionLoadError(path=resolved, error=f"Extension path does not exist: {resolved}")
                    )
        self._extensions_result = (
            self._extensions_override(extensions_result) if self._extensions_override else extensions_result
        )
        self._apply_extension_source_info(self._extensions_result.extensions, metadata_by_path)

        if self._no_skills:
            skill_paths = self._merge_paths(cli_enabled_skills, self._additional_skill_paths)
        else:
            skill_paths = self._merge_paths([*cli_enabled_skills, *enabled_skills], self._additional_skill_paths)

        self._last_skill_paths = skill_paths
        await self._update_skills_from_paths(skill_paths, metadata_by_path)
        for path in self._additional_skill_paths:
            if is_local_path(path):
                resolved = self._resolve_resource_path(path)
                if not await fs.Path(resolved).exists() and not any(
                    d.path == resolved for d in self._skill_diagnostics
                ):
                    self._skill_diagnostics.append(
                        ResourceDiagnostic(type="error", message="Skill path does not exist", path=resolved)
                    )

        if self._no_prompt_templates:
            prompt_paths = self._merge_paths(cli_enabled_prompts, self._additional_prompt_template_paths)
        else:
            prompt_paths = self._merge_paths(
                [*cli_enabled_prompts, *enabled_prompts], self._additional_prompt_template_paths
            )

        self._last_prompt_paths = prompt_paths
        await self._update_prompts_from_paths(prompt_paths, metadata_by_path)
        for path in self._additional_prompt_template_paths:
            if is_local_path(path):
                resolved = self._resolve_resource_path(path)
                if not await fs.Path(resolved).exists() and not any(
                    d.path == resolved for d in self._prompt_diagnostics
                ):
                    self._prompt_diagnostics.append(
                        ResourceDiagnostic(type="error", message="Prompt template path does not exist", path=resolved)
                    )

        enabled_themes = get_enabled_paths(resolved_paths.themes)
        cli_enabled_themes = get_enabled_paths(cli_extension_paths.themes)
        if self._no_themes:
            theme_paths = self._merge_paths(cli_enabled_themes, self._additional_theme_paths)
        else:
            theme_paths = self._merge_paths([*cli_enabled_themes, *enabled_themes], self._additional_theme_paths)
        await tonio.spawn_blocking(self._update_themes_from_paths, theme_paths, metadata_by_path)

        agents_files = (
            [] if self._no_context_files else await load_project_context_files(cwd=self._cwd, agent_dir=self._agent_dir)
        )
        self._agents_files = (
            self._agents_files_override(agents_files) if self._agents_files_override is not None else agents_files
        )

        system_prompt_source = (
            self._system_prompt_source
            if self._system_prompt_source is not None
            else self._discover_system_prompt_file()
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

        if self._append_system_prompt_source is not None:
            append_sources = self._append_system_prompt_source
        else:
            discovered = self._discover_append_system_prompt_file()
            append_sources = [discovered] if discovered is not None else []
        base_append = [
            resolved
            for source in append_sources
            if (resolved := await _resolve_prompt_input(source, "append system prompt")) is not None
        ]
        self._append_system_prompt = (
            self._append_system_prompt_override(base_append)
            if self._append_system_prompt_override is not None
            else base_append
        )
        self._append_system_prompt_source_paths = [
            resolve_path(source) for source in append_sources if await fs.Path(source).exists()
        ]
        self._loaded = True

    # -- extension loading -------------------------------------------------------

    async def _load_current_extension_set(self, *, include_inline_factories: bool) -> LoadExtensionsResult:
        resolved_paths = await self._package_manager.resolve()
        cli_extension_paths = await self._package_manager.resolve_extension_sources(
            self._additional_extension_paths, temporary=True
        )
        enabled = [resource.path for resource in resolved_paths.extensions if resource.enabled]
        cli_enabled = [resource.path for resource in cli_extension_paths.extensions if resource.enabled]
        extension_paths = cli_enabled if self._no_extensions else self._merge_paths(cli_enabled, enabled)

        extensions_result = await load_extensions_cached(extension_paths, self._cwd, self._event_bus)
        if not include_inline_factories:
            return extensions_result

        inline_extensions, inline_errors = await self._load_extension_factories(extensions_result.runtime)
        extensions_result.extensions.extend(inline_extensions)
        extensions_result.errors.extend(inline_errors)
        return extensions_result

    def _resolve_extension_load_path(self, path: str) -> str:
        return resolve_path(path, self._cwd, normalize_unicode_spaces=True)

    async def _load_final_extension_set(
        self, extension_paths: list[str], pre_trust_extensions: LoadExtensionsResult | None
    ) -> LoadExtensionsResult:
        if pre_trust_extensions is None:
            extensions_result = await load_extensions_cached(extension_paths, self._cwd, self._event_bus)
            inline_extensions, inline_errors = await self._load_extension_factories(extensions_result.runtime)
            extensions_result.extensions.extend(inline_extensions)
            extensions_result.errors.extend(inline_errors)
            self._add_extension_conflict_diagnostics(extensions_result)
            return extensions_result

        # The bootstrap pass already ran these factories; re-running them would
        # double every registration, so only the paths it did not reach load now.
        preloaded_by_path = {
            extension.resolved_path: extension
            for extension in pre_trust_extensions.extensions
            if not extension.path.startswith("<inline:")
        }
        failed_preload_paths = {self._resolve_extension_load_path(error.path) for error in pre_trust_extensions.errors}
        remaining_paths = [
            path
            for path in extension_paths
            if self._resolve_extension_load_path(path) not in preloaded_by_path
            and self._resolve_extension_load_path(path) not in failed_preload_paths
        ]
        remaining = await load_extensions_cached(
            remaining_paths, self._cwd, self._event_bus, pre_trust_extensions.runtime
        )
        loaded_by_path = dict(preloaded_by_path)
        for extension in remaining.extensions:
            loaded_by_path[extension.resolved_path] = extension

        inline_extensions = [
            extension for extension in pre_trust_extensions.extensions if extension.path.startswith("<inline:")
        ]
        ordered = [
            extension
            for path in extension_paths
            if (extension := loaded_by_path.get(self._resolve_extension_load_path(path))) is not None
        ]
        ordered.extend(inline_extensions)

        extensions_result = LoadExtensionsResult(
            extensions=ordered,
            errors=[*pre_trust_extensions.errors, *remaining.errors],
            runtime=pre_trust_extensions.runtime,
        )
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

    def _apply_extension_source_info(
        self, extensions: list[Extension], metadata_by_path: dict[str, PathMetadata]
    ) -> None:
        for extension in extensions:
            source_info = self._find_source_info_for_path(
                extension.path, None, metadata_by_path
            ) or self._get_default_source_info_for_path(extension.path)
            extension.source_info = source_info
            for command in extension.commands.values():
                command.source_info = source_info
            for tool in extension.tools.values():
                tool.source_info = source_info

    # -- helpers ----------------------------------------------------------------

    def _map_skill_path(self, resource: ResolvedResource, metadata_by_path: dict[str, PathMetadata]) -> str:
        if resource.metadata.source != "auto" and resource.metadata.origin != "package":
            return resource.path
        try:
            if not os.path.isdir(resource.path):
                return resource.path
        except OSError:
            return resource.path
        skill_file = os.path.join(resource.path, "SKILL.md")
        if os.path.exists(skill_file):
            if skill_file not in metadata_by_path:
                metadata_by_path[skill_file] = resource.metadata
            return skill_file
        return resource.path

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
        self._skills = [
            replace(
                skill,
                source_info=(
                    self._find_source_info_for_path(
                        skill.file_path, self._extension_skill_source_infos, metadata_by_path
                    )
                    or skill.source_info
                    or self._get_default_source_info_for_path(skill.file_path)
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
            all_prompts = await load_prompt_templates(
                cwd=self._cwd,
                agent_dir=self._agent_dir,
                prompt_paths=prompt_paths,
                include_defaults=False,
            )
            prompts_result = self._dedupe_prompts(all_prompts)
        resolved_prompts = (
            self._prompts_override(prompts_result) if self._prompts_override is not None else prompts_result
        )
        self._prompts = [
            replace(
                prompt,
                source_info=(
                    self._find_source_info_for_path(
                        prompt.file_path, self._extension_prompt_source_infos, metadata_by_path
                    )
                    or prompt.source_info
                    or self._get_default_source_info_for_path(prompt.file_path)
                ),
            )
            for prompt in resolved_prompts.prompts
        ]
        self._prompt_diagnostics = resolved_prompts.diagnostics

    def _update_themes_from_paths(
        self, theme_paths: list[str], metadata_by_path: dict[str, PathMetadata] | None = None
    ) -> None:
        # lazy: core <-> modes import cycle (see modes/__init__.py)
        # This whole method runs pool-side, so it uses the blocking loader
        # directly rather than the awaitable one.
        from ..modes.interactive.theme import _load_theme_from_path_sync as load_theme_from_path

        if self._no_themes and not theme_paths:
            themes: list = []
            diagnostics: list[ResourceDiagnostic] = []
        else:
            themes = []
            diagnostics = []
            # Default theme directories (agent-level and project-level)
            if not self._no_themes:
                default_dirs = [
                    os.path.join(self._agent_dir, "themes"),
                    os.path.join(self._cwd, CONFIG_DIR_NAME, "themes"),
                ]
                for theme_dir in default_dirs:
                    self._load_themes_from_dir(theme_dir, themes, diagnostics, load_theme_from_path)

            for path in theme_paths:
                resolved = self._resolve_resource_path(path)
                if not os.path.exists(resolved):
                    diagnostics.append(
                        ResourceDiagnostic(type="warning", message="theme path does not exist", path=resolved)
                    )
                    continue
                if os.path.isdir(resolved):
                    self._load_themes_from_dir(resolved, themes, diagnostics, load_theme_from_path)
                else:
                    self._load_theme_from_file(resolved, themes, diagnostics, load_theme_from_path)

            deduped_themes, dedupe_diagnostics = self._dedupe_themes(themes)
            themes = deduped_themes
            diagnostics = [*diagnostics, *dedupe_diagnostics]

        for loaded_theme in themes:
            source_path = loaded_theme.source_path
            if source_path:
                loaded_theme.source_info = self._find_source_info_for_path(
                    source_path, None, metadata_by_path
                ) or self._get_default_source_info_for_path(source_path)
        self._themes = themes
        self._theme_diagnostics = diagnostics

    def _load_themes_from_dir(self, theme_dir: str, themes: list, diagnostics: list, load_theme_from_path) -> None:
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
        extra_source_infos: dict[str, SourceInfo] | None = None,
        metadata_by_path: dict[str, PathMetadata] | None = None,
    ) -> SourceInfo | None:
        if not resource_path:
            return None

        if resource_path.startswith("<"):
            return self._get_default_source_info_for_path(resource_path)

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

    def _get_default_source_info_for_path(self, file_path: str) -> SourceInfo:
        if file_path.startswith("<") and file_path.endswith(">"):
            return SourceInfo(
                path=file_path,
                source=file_path[1:-1].split(":")[0] or "temporary",
                scope="temporary",
                origin="top-level",
            )

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
            base_dir=normalized_path if os.path.isdir(normalized_path) else os.path.dirname(normalized_path),
        )

    def _merge_paths(self, primary: list[str], additional: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()

        for path in [*primary, *additional]:
            resolved = self._resolve_resource_path(path)
            canonical_path = canonicalize_path(resolved)
            if canonical_path in seen:
                continue
            seen.add(canonical_path)
            merged.append(resolved)

        return merged

    def _resolve_resource_path(self, path: str) -> str:
        return resolve_path(path, self._cwd, trim=True)

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

    def _discover_system_prompt_file(self) -> str | None:
        project_path = os.path.join(self._cwd, CONFIG_DIR_NAME, "SYSTEM.md")
        if self._settings_manager.is_project_trusted() and os.path.exists(project_path):
            return project_path

        global_path = os.path.join(self._agent_dir, "SYSTEM.md")
        if os.path.exists(global_path):
            return global_path

        return None

    def _discover_append_system_prompt_file(self) -> str | None:
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
