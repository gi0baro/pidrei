"""Mirror of pi coding-agent src/core/skills.ts (npm `ignore` → pathspec)."""

import os
import stat as stat_module
from dataclasses import dataclass

import pathspec

from ..config import CONFIG_DIR_NAME, get_agent_dir
from ..utils.frontmatter import parse_frontmatter
from ..utils.paths import canonicalize_path, resolve_path
from .diagnostics import ResourceCollision, ResourceDiagnostic
from .source_info import SourceInfo, create_synthetic_source_info


# Max name length per spec
MAX_NAME_LENGTH = 64

# Max description length per spec
MAX_DESCRIPTION_LENGTH = 1024

IGNORE_FILE_NAMES = (".gitignore", ".ignore", ".fdignore")

_NAME_RE_ALLOWED = set("abcdefghijklmnopqrstuvwxyz0123456789-")


class IgnoreMatcher:
    """Accumulating gitignore matcher (npm `ignore` equivalent on pathspec)."""

    def __init__(self) -> None:
        self._patterns: list[str] = []
        self._spec: pathspec.PathSpec | None = None

    def add(self, patterns: list[str]) -> None:
        self._patterns.extend(patterns)
        self._spec = None

    def ignores(self, path: str) -> bool:
        if not self._patterns:
            return False
        if self._spec is None:
            self._spec = pathspec.PathSpec.from_lines("gitwildmatch", self._patterns)
        return self._spec.match_file(path)


def prefix_ignore_pattern(line: str, prefix: str) -> str | None:
    trimmed = line.strip()
    if not trimmed:
        return None
    if trimmed.startswith("#") and not trimmed.startswith("\\#"):
        return None

    pattern = line
    negated = False

    if pattern.startswith("!"):
        negated = True
        pattern = pattern[1:]
    elif pattern.startswith("\\!"):
        pattern = pattern[1:]

    pattern = pattern.removeprefix("/")

    prefixed = f"{prefix}{pattern}" if prefix else pattern
    return f"!{prefixed}" if negated else prefixed


def add_ignore_rules(ig: IgnoreMatcher, dir: str, root_dir: str) -> None:
    relative_dir = os.path.relpath(dir, root_dir)
    prefix = f"{relative_dir}/" if relative_dir != "." else ""

    for filename in IGNORE_FILE_NAMES:
        ignore_path = os.path.join(dir, filename)
        if not os.path.exists(ignore_path):
            continue
        try:
            with open(ignore_path, encoding="utf-8") as f:
                content = f.read()
            patterns = [
                prefixed
                for line in content.splitlines()
                if (prefixed := prefix_ignore_pattern(line, prefix)) is not None
            ]
            if patterns:
                ig.add(patterns)
        except Exception:
            pass


@dataclass(slots=True)
class Skill:
    name: str
    description: str
    file_path: str
    base_dir: str
    source_info: SourceInfo
    disable_model_invocation: bool


@dataclass(slots=True)
class LoadSkillsResult:
    skills: list[Skill]
    diagnostics: list[ResourceDiagnostic]


def _validate_name(name: str) -> list[str]:
    """Validate skill name per Agent Skills spec."""
    errors: list[str] = []

    if len(name) > MAX_NAME_LENGTH:
        errors.append(f"name exceeds {MAX_NAME_LENGTH} characters ({len(name)})")

    if not name or any(char not in _NAME_RE_ALLOWED for char in name):
        errors.append("name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)")

    if name.startswith("-") or name.endswith("-"):
        errors.append("name must not start or end with a hyphen")

    if "--" in name:
        errors.append("name must not contain consecutive hyphens")

    return errors


def _validate_description(description: str | None) -> list[str]:
    errors: list[str] = []

    if not description or description.strip() == "":
        errors.append("description is required")
    elif len(description) > MAX_DESCRIPTION_LENGTH:
        errors.append(f"description exceeds {MAX_DESCRIPTION_LENGTH} characters ({len(description)})")

    return errors


def _create_skill_source_info(file_path: str, base_dir: str, source: str) -> SourceInfo:
    if source == "user":
        return create_synthetic_source_info(file_path, source="local", scope="user", base_dir=base_dir)
    if source == "project":
        return create_synthetic_source_info(file_path, source="local", scope="project", base_dir=base_dir)
    if source == "path":
        return create_synthetic_source_info(file_path, source="local", base_dir=base_dir)
    return create_synthetic_source_info(file_path, source=source, base_dir=base_dir)


def load_skills_from_dir(*, dir: str, source: str) -> LoadSkillsResult:
    """Load skills from a directory.

    Discovery rules:
    - if a directory contains SKILL.md, treat it as a skill root and do not recurse further
    - otherwise, load direct .md children in the root
    - recurse into subdirectories to find SKILL.md
    """
    return _load_skills_from_dir_internal(dir, source, True)


def _stat_kind(full_path: str) -> tuple[bool, bool] | None:
    """(is_directory, is_file) following symlinks; None for broken links."""
    try:
        stats = os.stat(full_path)
    except OSError:
        return None

    return stat_module.S_ISDIR(stats.st_mode), stat_module.S_ISREG(stats.st_mode)


def _load_skills_from_dir_internal(
    dir: str,
    source: str,
    include_root_files: bool,
    ignore_matcher: IgnoreMatcher | None = None,
    root_dir: str | None = None,
) -> LoadSkillsResult:
    skills: list[Skill] = []
    diagnostics: list[ResourceDiagnostic] = []

    if not os.path.exists(dir):
        return LoadSkillsResult(skills, diagnostics)

    root = root_dir if root_dir is not None else dir
    ig = ignore_matcher if ignore_matcher is not None else IgnoreMatcher()
    add_ignore_rules(ig, dir, root)

    try:
        entries = sorted(os.scandir(dir), key=lambda entry: entry.name)

        for entry in entries:
            if entry.name != "SKILL.md":
                continue

            full_path = os.path.join(dir, entry.name)
            kind = _stat_kind(full_path)
            if kind is None:
                continue
            _is_directory, is_file = kind

            rel_path = os.path.relpath(full_path, root)
            if not is_file or ig.ignores(rel_path):
                continue

            result_skill, file_diagnostics = _load_skill_from_file(full_path, source)
            if result_skill is not None:
                skills.append(result_skill)
            diagnostics.extend(file_diagnostics)
            return LoadSkillsResult(skills, diagnostics)

        for entry in entries:
            if entry.name.startswith("."):
                continue

            # Skip node_modules to avoid scanning dependencies
            if entry.name == "node_modules":
                continue

            full_path = os.path.join(dir, entry.name)
            kind = _stat_kind(full_path)
            if kind is None:
                continue  # Broken symlink, skip it
            is_directory, is_file = kind

            rel_path = os.path.relpath(full_path, root)
            ignore_path = f"{rel_path}/" if is_directory else rel_path
            if ig.ignores(ignore_path):
                continue

            if is_directory:
                sub_result = _load_skills_from_dir_internal(full_path, source, False, ig, root)
                skills.extend(sub_result.skills)
                diagnostics.extend(sub_result.diagnostics)
                continue

            if not is_file or not include_root_files or not entry.name.endswith(".md"):
                continue

            result_skill, file_diagnostics = _load_skill_from_file(full_path, source)
            if result_skill is not None:
                skills.append(result_skill)
            diagnostics.extend(file_diagnostics)
    except OSError:
        pass

    return LoadSkillsResult(skills, diagnostics)


def _load_skill_from_file(file_path: str, source: str) -> tuple[Skill | None, list[ResourceDiagnostic]]:
    diagnostics: list[ResourceDiagnostic] = []

    try:
        with open(file_path, encoding="utf-8") as f:
            raw_content = f.read()
        frontmatter, _body = parse_frontmatter(raw_content)
        if not isinstance(frontmatter, dict):
            frontmatter = {}
        skill_dir = os.path.dirname(file_path)
        parent_dir_name = os.path.basename(skill_dir)

        description = frontmatter.get("description")
        for error in _validate_description(description):
            diagnostics.append(ResourceDiagnostic(type="warning", message=error, path=file_path))

        # Use name from frontmatter, or fall back to parent directory name
        name = frontmatter.get("name") or parent_dir_name

        for error in _validate_name(name):
            diagnostics.append(ResourceDiagnostic(type="warning", message=error, path=file_path))

        # Still load the skill even with warnings (unless description is completely missing)
        if not description or description.strip() == "":
            return None, diagnostics

        return (
            Skill(
                name=name,
                description=description,
                file_path=file_path,
                base_dir=skill_dir,
                source_info=_create_skill_source_info(file_path, skill_dir, source),
                disable_model_invocation=frontmatter.get("disable-model-invocation") is True,
            ),
            diagnostics,
        )
    except Exception as error:
        message = str(error) or "failed to parse skill file"
        diagnostics.append(ResourceDiagnostic(type="warning", message=message, path=file_path))
        return None, diagnostics


def _escape_xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def format_skills_for_prompt(skills: list[Skill]) -> str:
    """Format skills for inclusion in a system prompt (Agent Skills XML format).

    Skills with disable_model_invocation=True are excluded from the prompt
    (they can only be invoked explicitly via /skill:name commands).
    """
    visible_skills = [skill for skill in skills if not skill.disable_model_invocation]

    if not visible_skills:
        return ""

    lines = [
        "\n\nThe following skills provide specialized instructions for specific tasks.",
        "Use the read tool to load a skill's file when the task matches its description.",
        (
            "When a skill file references a relative path, resolve it against the skill directory "
            "(parent of SKILL.md / dirname of the path) and use that absolute path in tool commands."
        ),
        "",
        "<available_skills>",
    ]

    for skill in visible_skills:
        lines.append("  <skill>")
        lines.append(f"    <name>{_escape_xml(skill.name)}</name>")
        lines.append(f"    <description>{_escape_xml(skill.description)}</description>")
        lines.append(f"    <location>{_escape_xml(skill.file_path)}</location>")
        lines.append("  </skill>")

    lines.append("</available_skills>")

    return "\n".join(lines)


def _is_under_path(target: str, root: str) -> bool:
    normalized_root = os.path.abspath(root)
    if target == normalized_root:
        return True
    prefix = normalized_root if normalized_root.endswith(os.sep) else f"{normalized_root}{os.sep}"
    return target.startswith(prefix)


def load_skills(
    *,
    cwd: str,
    agent_dir: str | None,
    skill_paths: list[str],
    include_defaults: bool,
) -> LoadSkillsResult:
    """Load skills from all configured locations, deduplicating by name."""
    resolved_cwd = resolve_path(cwd)
    resolved_agent_dir = resolve_path(agent_dir if agent_dir is not None else get_agent_dir())

    skill_map: dict[str, Skill] = {}
    real_path_set: set[str] = set()
    all_diagnostics: list[ResourceDiagnostic] = []
    collision_diagnostics: list[ResourceDiagnostic] = []

    def add_skills(result: LoadSkillsResult) -> None:
        all_diagnostics.extend(result.diagnostics)
        for skill in result.skills:
            # Resolve symlinks to detect duplicate files
            real_path = canonicalize_path(skill.file_path)

            # Skip silently if we've already loaded this exact file (via symlink)
            if real_path in real_path_set:
                continue

            existing = skill_map.get(skill.name)
            if existing is not None:
                collision_diagnostics.append(
                    ResourceDiagnostic(
                        type="collision",
                        message=f'name "{skill.name}" collision',
                        path=skill.file_path,
                        collision=ResourceCollision(
                            resource_type="skill",
                            name=skill.name,
                            winner_path=existing.file_path,
                            loser_path=skill.file_path,
                        ),
                    )
                )
            else:
                skill_map[skill.name] = skill
                real_path_set.add(real_path)

    user_skills_dir = os.path.join(resolved_agent_dir, "skills")
    project_skills_dir = os.path.join(resolved_cwd, CONFIG_DIR_NAME, "skills")

    if include_defaults:
        add_skills(_load_skills_from_dir_internal(user_skills_dir, "user", True))
        add_skills(_load_skills_from_dir_internal(project_skills_dir, "project", True))

    def get_source(resolved_path: str) -> str:
        if not include_defaults:
            if _is_under_path(resolved_path, user_skills_dir):
                return "user"
            if _is_under_path(resolved_path, project_skills_dir):
                return "project"
        return "path"

    for raw_path in skill_paths:
        resolved_path = resolve_path(raw_path, resolved_cwd, trim=True)
        if not os.path.exists(resolved_path):
            all_diagnostics.append(
                ResourceDiagnostic(type="warning", message="skill path does not exist", path=resolved_path)
            )
            continue

        try:
            source = get_source(resolved_path)
            if os.path.isdir(resolved_path):
                add_skills(_load_skills_from_dir_internal(resolved_path, source, True))
            elif os.path.isfile(resolved_path) and resolved_path.endswith(".md"):
                skill, file_diagnostics = _load_skill_from_file(resolved_path, source)
                if skill is not None:
                    add_skills(LoadSkillsResult(skills=[skill], diagnostics=file_diagnostics))
                else:
                    all_diagnostics.extend(file_diagnostics)
            else:
                all_diagnostics.append(
                    ResourceDiagnostic(type="warning", message="skill path is not a markdown file", path=resolved_path)
                )
        except Exception as error:
            message = str(error) or "failed to read skill path"
            all_diagnostics.append(ResourceDiagnostic(type="warning", message=message, path=resolved_path))

    return LoadSkillsResult(
        skills=list(skill_map.values()),
        diagnostics=[*all_diagnostics, *collision_diagnostics],
    )
