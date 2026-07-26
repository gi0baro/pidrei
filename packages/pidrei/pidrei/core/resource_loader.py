"""Mirror of pi coding-agent src/core/resource-loader.ts (Phase 3 subset).

Loads skills, prompt templates, AGENTS.md context files, and SYSTEM.md /
APPEND_SYSTEM.md, with the project-trust bootstrap flow. Extensions load as
an empty result (the extension system is Phase 5); both keep their
option/override seams so the
call sites port unchanged.
"""

import os
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace

from ..config import CONFIG_DIR_NAME
from ..utils.paths import canonicalize_path, is_local_path, resolve_path
from .diagnostics import ResourceCollision, ResourceDiagnostic
from .extensions.types import ExtensionLoadError, LoadExtensionsResult
from .package_manager import DefaultPackageManager, ResolvedResource
from .prompt_templates import PromptTemplate, load_prompt_templates
from .settings_manager import SettingsManager
from .skills import LoadSkillsResult, Skill, load_skills
from .source_info import PathMetadata, SourceInfo, create_source_info


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


def _resolve_prompt_input(input: str | None, description: str) -> str | None:
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
    for filename in ("AGENTS.md", "AGENTS.MD", "CLAUDE.md", "CLAUDE.MD"):
        file_path = os.path.join(dir, filename)
        if os.path.exists(file_path):
            try:
                with open(file_path, encoding="utf-8") as f:
                    return AgentsFile(path=file_path, content=f.read())
            except OSError as error:
                _warn(f"Could not read {file_path}: {error}")
    return None


def load_project_context_files(*, cwd: str, agent_dir: str) -> list[AgentsFile]:
    resolved_cwd = resolve_path(cwd)
    resolved_agent_dir = resolve_path(agent_dir)

    context_files: list[AgentsFile] = []
    seen_paths: set[str] = set()

    global_context = _load_context_file_from_dir(resolved_agent_dir)
    if global_context is not None:
        context_files.append(global_context)
        seen_paths.add(global_context.path)

    ancestor_context_files: list[AgentsFile] = []

    current_dir = resolved_cwd
    while True:
        context_file = _load_context_file_from_dir(current_dir)
        if context_file is not None and context_file.path not in seen_paths:
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
        skills_override: Callable[[LoadSkillsResult], LoadSkillsResult] | None = None,
        prompts_override: Callable[[LoadPromptsResult], LoadPromptsResult] | None = None,
        agents_files_override: Callable[[list[AgentsFile]], list[AgentsFile]] | None = None,
        system_prompt_override: Callable[[str | None], str | None] | None = None,
        append_system_prompt_override: Callable[[list[str]], list[str]] | None = None,
    ):
        self._cwd = resolve_path(cwd)
        self._agent_dir = resolve_path(agent_dir)
        self._settings_manager = (
            settings_manager if settings_manager is not None else SettingsManager.create(self._cwd, self._agent_dir)
        )
        self._package_manager = DefaultPackageManager(
            cwd=self._cwd, agent_dir=self._agent_dir, settings_manager=self._settings_manager
        )
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
        self._skills_override = skills_override
        self._prompts_override = prompts_override
        self._agents_files_override = agents_files_override
        self._system_prompt_override = system_prompt_override
        self._append_system_prompt_override = append_system_prompt_override

        self._extensions_result = LoadExtensionsResult()
        self._skills: list[Skill] = []
        self._skill_diagnostics: list[ResourceDiagnostic] = []
        self._prompts: list[PromptTemplate] = []
        self._prompt_diagnostics: list[ResourceDiagnostic] = []
        self._agents_files: list[AgentsFile] = []
        self._system_prompt: str | None = None
        self._append_system_prompt: list[str] = []
        self._last_skill_paths: list[str] = []
        self._extension_skill_source_infos: dict[str, SourceInfo] = {}
        self._extension_prompt_source_infos: dict[str, SourceInfo] = {}
        self._last_prompt_paths: list[str] = []
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

    def get_append_system_prompt(self) -> list[str]:
        return self._append_system_prompt

    # -- extension-provided resources -------------------------------------------

    def extend_resources(
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
            self._update_skills_from_paths(self._last_skill_paths)

        if normalized_prompts:
            self._last_prompt_paths = self._merge_paths(
                self._last_prompt_paths, [entry.path for entry in normalized_prompts]
            )
            self._update_prompts_from_paths(self._last_prompt_paths)

    # -- reload ------------------------------------------------------------------

    async def load_project_trust_extensions(self) -> LoadExtensionsResult:
        # Force untrusted project settings for the bootstrap pass. This keeps project-local
        # extensions/packages out while still loading user/global and temporary CLI extensions.
        self._settings_manager.set_project_trusted(False)
        self._settings_manager.reload()
        # Extension loading is Phase 5; the bootstrap pass yields no extensions.
        return LoadExtensionsResult()

    async def reload(
        self,
        *,
        resolve_project_trust: Callable[[LoadExtensionsResult], Awaitable[bool]] | None = None,
    ) -> None:
        if resolve_project_trust is not None:
            pre_trust_extensions = await self.load_project_trust_extensions()
            project_trusted = await resolve_project_trust(pre_trust_extensions)
            self._settings_manager.set_project_trusted(project_trusted)

        # reload() preserves SettingsManager.project_trusted and reloads settings for that trust state.
        self._settings_manager.reload()
        resolved_paths = await self._package_manager.resolve()
        cli_extension_paths = await self._package_manager.resolve_extension_sources(
            self._additional_extension_paths, temporary=True
        )
        metadata_by_path: dict[str, PathMetadata] = {}

        self._extension_skill_source_infos = {}
        self._extension_prompt_source_infos = {}

        def get_enabled_resources(resources: list[ResolvedResource]) -> list[ResolvedResource]:
            for resource in resources:
                if resource.path not in metadata_by_path:
                    metadata_by_path[resource.path] = resource.metadata
            return [resource for resource in resources if resource.enabled]

        def get_enabled_paths(resources: list[ResolvedResource]) -> list[str]:
            return [resource.path for resource in get_enabled_resources(resources)]

        enabled_skill_resources = get_enabled_resources(resolved_paths.skills)
        enabled_prompts = get_enabled_paths(resolved_paths.prompts)

        enabled_skills = [self._map_skill_path(resource, metadata_by_path) for resource in enabled_skill_resources]

        for resource in cli_extension_paths.skills:
            if resource.path not in metadata_by_path:
                metadata_by_path[resource.path] = PathMetadata(source="cli", scope="temporary", origin="top-level")

        cli_enabled_skills = get_enabled_paths(cli_extension_paths.skills)
        cli_enabled_prompts = get_enabled_paths(cli_extension_paths.prompts)

        # Extension loading is Phase 5: report missing explicit extension paths only.
        extensions_result = LoadExtensionsResult()
        for path in self._additional_extension_paths:
            if is_local_path(path):
                resolved = self._resolve_resource_path(path)
                if not os.path.exists(resolved):
                    extensions_result.errors.append(
                        ExtensionLoadError(path=resolved, error=f"Extension path does not exist: {resolved}")
                    )
        self._extensions_result = extensions_result

        if self._no_skills:
            skill_paths = self._merge_paths(cli_enabled_skills, self._additional_skill_paths)
        else:
            skill_paths = self._merge_paths([*cli_enabled_skills, *enabled_skills], self._additional_skill_paths)

        self._last_skill_paths = skill_paths
        self._update_skills_from_paths(skill_paths, metadata_by_path)
        for path in self._additional_skill_paths:
            if is_local_path(path):
                resolved = self._resolve_resource_path(path)
                if not os.path.exists(resolved) and not any(d.path == resolved for d in self._skill_diagnostics):
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
        self._update_prompts_from_paths(prompt_paths, metadata_by_path)
        for path in self._additional_prompt_template_paths:
            if is_local_path(path):
                resolved = self._resolve_resource_path(path)
                if not os.path.exists(resolved) and not any(d.path == resolved for d in self._prompt_diagnostics):
                    self._prompt_diagnostics.append(
                        ResourceDiagnostic(type="error", message="Prompt template path does not exist", path=resolved)
                    )

        enabled_themes = get_enabled_paths(resolved_paths.themes)
        cli_enabled_themes = get_enabled_paths(cli_extension_paths.themes)
        if self._no_themes:
            theme_paths = self._merge_paths(cli_enabled_themes, self._additional_theme_paths)
        else:
            theme_paths = self._merge_paths([*cli_enabled_themes, *enabled_themes], self._additional_theme_paths)
        self._update_themes_from_paths(theme_paths, metadata_by_path)

        agents_files = (
            [] if self._no_context_files else load_project_context_files(cwd=self._cwd, agent_dir=self._agent_dir)
        )
        self._agents_files = (
            self._agents_files_override(agents_files) if self._agents_files_override is not None else agents_files
        )

        base_system_prompt = _resolve_prompt_input(
            self._system_prompt_source
            if self._system_prompt_source is not None
            else self._discover_system_prompt_file(),
            "system prompt",
        )
        self._system_prompt = (
            self._system_prompt_override(base_system_prompt)
            if self._system_prompt_override is not None
            else base_system_prompt
        )

        if self._append_system_prompt_source is not None:
            append_sources = self._append_system_prompt_source
        else:
            discovered = self._discover_append_system_prompt_file()
            append_sources = [discovered] if discovered is not None else []
        base_append = [
            resolved
            for source in append_sources
            if (resolved := _resolve_prompt_input(source, "append system prompt")) is not None
        ]
        self._append_system_prompt = (
            self._append_system_prompt_override(base_append)
            if self._append_system_prompt_override is not None
            else base_append
        )

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

    def _update_skills_from_paths(
        self, skill_paths: list[str], metadata_by_path: dict[str, PathMetadata] | None = None
    ) -> None:
        if self._no_skills and not skill_paths:
            skills_result = LoadSkillsResult(skills=[], diagnostics=[])
        else:
            skills_result = load_skills(
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

    def _update_prompts_from_paths(
        self, prompt_paths: list[str], metadata_by_path: dict[str, PathMetadata] | None = None
    ) -> None:
        if self._no_prompt_templates and not prompt_paths:
            prompts_result = LoadPromptsResult(prompts=[], diagnostics=[])
        else:
            all_prompts = load_prompt_templates(
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
        from ..modes.interactive.theme import load_theme_from_path

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
