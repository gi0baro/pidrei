"""Skill loading and invocation (port of pi `harness/skills.ts`).

Ignore-file matching uses `pathspec` (gitwildmatch) instead of the npm
`ignore` package — both implement gitignore semantics.
"""

import re
from dataclasses import dataclass
from typing import Any, Literal

import pathspec

from .prompt_templates import parse_frontmatter
from .types import FileInfo, Skill, to_error


MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
IGNORE_FILE_NAMES = [".gitignore", ".ignore", ".fdignore"]

type SkillDiagnosticCode = Literal["file_info_failed", "list_failed", "read_failed", "parse_failed", "invalid_metadata"]


@dataclass(slots=True, kw_only=True)
class SkillDiagnostic:
    """Warning produced while loading skills."""

    code: SkillDiagnosticCode
    message: str
    path: str
    type: Literal["warning"] = "warning"


@dataclass(slots=True)
class LoadedSkills:
    skills: list[Skill]
    diagnostics: list[SkillDiagnostic]


@dataclass(slots=True)
class SourcedSkill:
    skill: Skill
    source: Any


@dataclass(slots=True)
class SourcedSkillDiagnostic(SkillDiagnostic):
    source: Any = None


@dataclass(slots=True)
class LoadedSourcedSkills:
    skills: list[SourcedSkill]
    diagnostics: list[SourcedSkillDiagnostic]


class _IgnoreMatcher:
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


def dirname_env_path(path: str) -> str:
    normalized = path.rstrip("/")
    slash_index = normalized.rfind("/")
    return "/" if slash_index <= 0 else normalized[:slash_index]


def basename_env_path(path: str) -> str:
    normalized = path.rstrip("/")
    slash_index = normalized.rfind("/")
    return normalized if slash_index == -1 else normalized[slash_index + 1 :]


def _join_env_path(base: str, child: str) -> str:
    return f"{base.rstrip('/')}/{child.lstrip('/')}"


def _relative_env_path(root: str, path: str) -> str:
    normalized_root = root.rstrip("/")
    normalized_path = path.rstrip("/")
    if normalized_path == normalized_root:
        return ""
    if normalized_path.startswith(f"{normalized_root}/"):
        return normalized_path[len(normalized_root) + 1 :]
    return normalized_path.lstrip("/")


def format_skill_invocation(skill: Skill, additional_instructions: str | None = None) -> str:
    """Format a skill invocation prompt, optionally appending additional user instructions."""
    skill_block = (
        f'<skill name="{skill.name}" location="{skill.file_path}">\n'
        f"References are relative to {dirname_env_path(skill.file_path)}.\n\n"
        f"{skill.content}\n</skill>"
    )
    return f"{skill_block}\n\n{additional_instructions}" if additional_instructions else skill_block


async def load_skills(env, dirs: str | list[str]) -> LoadedSkills:
    """Load skills from one or more directories.

    Traverses directories recursively, loads `SKILL.md` files, loads direct
    root `.md` files as skills, honors ignore files, and returns diagnostics
    for invalid skill files. Missing input directories are skipped.
    """
    skills: list[Skill] = []
    diagnostics: list[SkillDiagnostic] = []
    for directory in dirs if isinstance(dirs, list) else [dirs]:
        root_info_result = await env.file_info(directory)
        if not root_info_result.ok:
            if root_info_result.error.code != "not_found":
                diagnostics.append(
                    SkillDiagnostic(code="file_info_failed", message=root_info_result.error.message, path=directory)
                )
            continue
        root_info = root_info_result.value
        if await _resolve_kind(env, root_info, diagnostics) != "directory":
            continue
        result = await _load_skills_from_dir(env, root_info.path, True, _IgnoreMatcher(), root_info.path)
        skills.extend(result.skills)
        diagnostics.extend(result.diagnostics)
    return LoadedSkills(skills=skills, diagnostics=diagnostics)


async def load_sourced_skills(env, inputs: list[dict[str, Any]], map_skill=None) -> LoadedSourcedSkills:
    """Load skills from source-tagged directories.

    Source values are preserved exactly and attached to every loaded skill and
    diagnostic.
    """
    skills: list[SourcedSkill] = []
    diagnostics: list[SourcedSkillDiagnostic] = []
    for entry in inputs:
        result = await load_skills(env, entry["path"])
        for skill in result.skills:
            skills.append(
                SourcedSkill(skill=map_skill(skill, entry["source"]) if map_skill else skill, source=entry["source"])
            )
        for diagnostic in result.diagnostics:
            diagnostics.append(
                SourcedSkillDiagnostic(
                    code=diagnostic.code, message=diagnostic.message, path=diagnostic.path, source=entry["source"]
                )
            )
    return LoadedSourcedSkills(skills=skills, diagnostics=diagnostics)


async def _load_skills_from_dir(
    env,
    directory: str,
    include_root_files: bool,
    ignore_matcher: _IgnoreMatcher,
    root_dir: str,
) -> LoadedSkills:
    skills: list[Skill] = []
    diagnostics: list[SkillDiagnostic] = []

    dir_info_result = await env.file_info(directory)
    if not dir_info_result.ok:
        if dir_info_result.error.code != "not_found":
            diagnostics.append(
                SkillDiagnostic(code="file_info_failed", message=dir_info_result.error.message, path=directory)
            )
        return LoadedSkills(skills=skills, diagnostics=diagnostics)
    if await _resolve_kind(env, dir_info_result.value, diagnostics) != "directory":
        return LoadedSkills(skills=skills, diagnostics=diagnostics)

    await _add_ignore_rules(env, ignore_matcher, directory, root_dir, diagnostics)

    entries_result = await env.list_dir(directory)
    if not entries_result.ok:
        diagnostics.append(SkillDiagnostic(code="list_failed", message=entries_result.error.message, path=directory))
        return LoadedSkills(skills=skills, diagnostics=diagnostics)
    entries = entries_result.value

    for entry in entries:
        if entry.name != "SKILL.md":
            continue
        kind = await _resolve_kind(env, entry, diagnostics)
        if kind != "file":
            continue
        rel_path = _relative_env_path(root_dir, entry.path)
        if ignore_matcher.ignores(rel_path):
            continue

        skill, file_diagnostics = await _load_skill_from_file(env, entry.path)
        if skill is not None:
            skills.append(skill)
        diagnostics.extend(file_diagnostics)
        return LoadedSkills(skills=skills, diagnostics=diagnostics)

    for entry in sorted(entries, key=lambda info: info.name):
        if entry.name.startswith(".") or entry.name == "node_modules":
            continue
        kind = await _resolve_kind(env, entry, diagnostics)
        if kind is None:
            continue

        rel_path = _relative_env_path(root_dir, entry.path)
        ignore_path = f"{rel_path}/" if kind == "directory" else rel_path
        if ignore_matcher.ignores(ignore_path):
            continue

        if kind == "directory":
            result = await _load_skills_from_dir(env, entry.path, False, ignore_matcher, root_dir)
            skills.extend(result.skills)
            diagnostics.extend(result.diagnostics)
            continue

        if kind != "file" or not include_root_files or not entry.name.endswith(".md"):
            continue
        skill, file_diagnostics = await _load_skill_from_file(env, entry.path)
        if skill is not None:
            skills.append(skill)
        diagnostics.extend(file_diagnostics)

    return LoadedSkills(skills=skills, diagnostics=diagnostics)


async def _add_ignore_rules(env, matcher: _IgnoreMatcher, directory: str, root_dir: str, diagnostics: list) -> None:
    relative_dir = _relative_env_path(root_dir, directory)
    prefix = f"{relative_dir}/" if relative_dir else ""

    for filename in IGNORE_FILE_NAMES:
        ignore_path = _join_env_path(directory, filename)
        info = await env.file_info(ignore_path)
        if not info.ok:
            if info.error.code != "not_found":
                diagnostics.append(
                    SkillDiagnostic(code="file_info_failed", message=info.error.message, path=ignore_path)
                )
            continue
        if info.value.kind != "file":
            continue
        content = await env.read_text_file(ignore_path)
        if not content.ok:
            diagnostics.append(SkillDiagnostic(code="read_failed", message=content.error.message, path=ignore_path))
            continue
        patterns = [
            prefixed
            for line in content.value.replace("\r\n", "\n").split("\n")
            if (prefixed := _prefix_ignore_pattern(line, prefix)) is not None
        ]
        if patterns:
            matcher.add(patterns)


def _prefix_ignore_pattern(line: str, prefix: str) -> str | None:
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


async def _load_skill_from_file(env, file_path: str) -> tuple[Skill | None, list[SkillDiagnostic]]:
    diagnostics: list[SkillDiagnostic] = []
    raw_content = await env.read_text_file(file_path)
    if not raw_content.ok:
        diagnostics.append(SkillDiagnostic(code="read_failed", message=raw_content.error.message, path=file_path))
        return None, diagnostics

    try:
        frontmatter, body = parse_frontmatter(raw_content.value)
    except Exception as error:
        diagnostics.append(SkillDiagnostic(code="parse_failed", message=str(to_error(error)), path=file_path))
        return None, diagnostics

    skill_dir = dirname_env_path(file_path)
    parent_dir_name = basename_env_path(skill_dir)
    description = frontmatter.get("description") if isinstance(frontmatter.get("description"), str) else None

    for error in _validate_description(description):
        diagnostics.append(SkillDiagnostic(code="invalid_metadata", message=error, path=file_path))

    frontmatter_name = frontmatter.get("name") if isinstance(frontmatter.get("name"), str) else None
    name = frontmatter_name or parent_dir_name
    for error in _validate_name(name, parent_dir_name):
        diagnostics.append(SkillDiagnostic(code="invalid_metadata", message=error, path=file_path))

    if not description or description.strip() == "":
        return None, diagnostics

    return (
        Skill(
            name=name,
            description=description,
            content=body,
            file_path=file_path,
            disable_model_invocation=frontmatter.get("disable-model-invocation") is True,
        ),
        diagnostics,
    )


def _validate_name(name: str, parent_dir_name: str) -> list[str]:
    errors: list[str] = []
    if name != parent_dir_name:
        errors.append(f'name "{name}" does not match parent directory "{parent_dir_name}"')
    if len(name) > MAX_NAME_LENGTH:
        errors.append(f"name exceeds {MAX_NAME_LENGTH} characters ({len(name)})")
    if not re.fullmatch(r"[a-z0-9-]+", name):
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


async def _resolve_kind(env, info: FileInfo, diagnostics: list) -> str | None:
    if info.kind in ("file", "directory"):
        return info.kind
    canonical_path = await env.canonical_path(info.path)
    if not canonical_path.ok:
        if canonical_path.error.code != "not_found":
            diagnostics.append(
                SkillDiagnostic(code="file_info_failed", message=canonical_path.error.message, path=info.path)
            )
        return None
    target = await env.file_info(canonical_path.value)
    if not target.ok:
        if target.error.code != "not_found":
            diagnostics.append(SkillDiagnostic(code="file_info_failed", message=target.error.message, path=info.path))
        return None
    return target.value.kind if target.value.kind in ("file", "directory") else None
