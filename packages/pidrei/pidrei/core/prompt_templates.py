"""Mirror of pi coding-agent src/core/prompt-templates.ts."""

import os
import re
from collections.abc import Awaitable
from dataclasses import dataclass

import tonio.colored as tonio

from ..config import CONFIG_DIR_NAME
from ..utils.frontmatter import parse_frontmatter
from ..utils.paths import resolve_path
from .source_info import SourceInfo, create_synthetic_source_info


@dataclass(slots=True)
class PromptTemplate:
    """A prompt template loaded from a markdown file."""

    name: str
    description: str
    content: str
    source_info: SourceInfo
    file_path: str  # Absolute path to the template file
    argument_hint: str | None = None


def parse_command_args(args_string: str) -> list[str]:
    """Parse command arguments respecting quoted strings (bash-style)."""
    args: list[str] = []
    current = ""
    in_quote: str | None = None

    for char in args_string:
        if in_quote:
            if char == in_quote:
                in_quote = None
            else:
                current += char
        elif char in ('"', "'"):
            in_quote = char
        elif char.isspace():
            if current:
                args.append(current)
                current = ""
        else:
            current += char

    if current:
        args.append(current)

    return args


_SUBSTITUTION_RE = re.compile(r"\$\{(\d+|ARGUMENTS|@):-([^}]*)\}|\$\{@:(\d+)(?::(\d+))?\}|\$(ARGUMENTS|@|\d+)")


def substitute_args(content: str, args: list[str]) -> str:
    """Substitute argument placeholders in template content.

    Supports:
    - $1, $2, ... for positional args
    - $@ and $ARGUMENTS for all args
    - ${N:-default} for positional arg N with default when missing/empty
    - ${@:-default} and ${ARGUMENTS:-default} for all args with a default when empty
    - ${@:N} for args from Nth onwards (bash-style slicing)
    - ${@:N:L} for L args starting from Nth

    Replacement happens on the template string only. Argument and default
    values containing patterns like $1, $@, or $ARGUMENTS are NOT recursively
    substituted.
    """
    all_args = " ".join(args)

    def replacement(match: re.Match) -> str:
        default_target, default_value, slice_start, slice_length, simple = match.groups()

        if default_target:
            if default_target in ("@", "ARGUMENTS"):
                value = all_args
            else:
                index = int(default_target) - 1
                value = args[index] if 0 <= index < len(args) else None
            return value if value else default_value

        if slice_start:
            start = int(slice_start) - 1  # Convert to 0-indexed (user provides 1-indexed)
            # Treat 0 as 1 (bash convention: args start at 1)
            start = max(start, 0)

            if slice_length:
                length = int(slice_length)
                return " ".join(args[start : start + length])
            return " ".join(args[start:])

        if simple in ("ARGUMENTS", "@"):
            return all_args

        index = int(simple) - 1
        return args[index] if 0 <= index < len(args) else ""

    return _SUBSTITUTION_RE.sub(replacement, content)


def _load_template_from_file(file_path: str, source_info: SourceInfo) -> PromptTemplate | None:
    try:
        with open(file_path, encoding="utf-8") as f:
            raw_content = f.read()
        frontmatter, body = parse_frontmatter(raw_content)
        if not isinstance(frontmatter, dict):
            frontmatter = {}

        name = re.sub(r"\.md$", "", os.path.basename(file_path))

        # Get description from frontmatter or first non-empty line
        description = frontmatter.get("description") or ""
        if not description:
            first_line = next((line for line in body.split("\n") if line.strip()), None)
            if first_line:
                # Truncate if too long
                description = first_line[:60]
                if len(first_line) > 60:
                    description += "..."

        argument_hint = frontmatter.get("argument-hint")
        return PromptTemplate(
            name=name,
            description=description,
            argument_hint=argument_hint if argument_hint else None,
            content=body,
            source_info=source_info,
            file_path=file_path,
        )
    except Exception:
        return None


def _load_templates_from_dir(dir: str, get_source_info) -> list[PromptTemplate]:
    """Scan a directory for .md files (non-recursive) and load them as prompt templates."""
    templates: list[PromptTemplate] = []

    if not os.path.exists(dir):
        return templates

    try:
        entries = sorted(os.scandir(dir), key=lambda entry: entry.name)
    except OSError:
        return templates

    for entry in entries:
        full_path = os.path.join(dir, entry.name)

        # For symlinks, check if they point to a file
        try:
            is_file = os.path.isfile(full_path)
        except OSError:
            continue  # Broken symlink, skip it

        if is_file and entry.name.endswith(".md"):
            template = _load_template_from_file(full_path, get_source_info(full_path))
            if template is not None:
                templates.append(template)

    return templates


def _is_under_path(target: str, root: str) -> bool:
    normalized_root = os.path.abspath(root)
    if target == normalized_root:
        return True
    prefix = normalized_root if normalized_root.endswith(os.sep) else f"{normalized_root}{os.sep}"
    return target.startswith(prefix)


def load_prompt_templates(
    *,
    cwd: str,
    agent_dir: str,
    prompt_paths: list[str],
    include_defaults: bool,
) -> Awaitable[list[PromptTemplate]]:
    """Load all prompt templates from the global/project prompt dirs and explicit paths.

    A directory scan plus one read per template is a single blocking unit, so
    it goes to the pool whole rather than as a dozen separate `fs` hops. The
    helpers below stay sync because they only ever run there.

    Sync, returning the awaitable rather than `async def ...: return await ...`:
    the caller awaits it either way, and the extra coroutine frame buys nothing.
    """
    return tonio.spawn_blocking(
        _load_prompt_templates_sync,
        cwd=cwd,
        agent_dir=agent_dir,
        prompt_paths=prompt_paths,
        include_defaults=include_defaults,
    )


def _load_prompt_templates_sync(
    *,
    cwd: str,
    agent_dir: str,
    prompt_paths: list[str],
    include_defaults: bool,
) -> list[PromptTemplate]:
    resolved_cwd = resolve_path(cwd)
    resolved_agent_dir = resolve_path(agent_dir)

    templates: list[PromptTemplate] = []

    global_prompts_dir = os.path.join(resolved_agent_dir, "prompts")
    project_prompts_dir = os.path.join(resolved_cwd, CONFIG_DIR_NAME, "prompts")

    def get_source_info(resolved_path: str) -> SourceInfo:
        if _is_under_path(resolved_path, global_prompts_dir):
            return create_synthetic_source_info(
                resolved_path, source="local", scope="user", base_dir=global_prompts_dir
            )
        if _is_under_path(resolved_path, project_prompts_dir):
            return create_synthetic_source_info(
                resolved_path, source="local", scope="project", base_dir=project_prompts_dir
            )
        return create_synthetic_source_info(
            resolved_path,
            source="local",
            base_dir=resolved_path if os.path.isdir(resolved_path) else os.path.dirname(resolved_path),
        )

    if include_defaults:
        templates.extend(_load_templates_from_dir(global_prompts_dir, get_source_info))
        templates.extend(_load_templates_from_dir(project_prompts_dir, get_source_info))

    # Load explicit prompt paths
    for raw_path in prompt_paths:
        resolved_path = resolve_path(raw_path, resolved_cwd, trim=True)
        if not os.path.exists(resolved_path):
            continue

        try:
            if os.path.isdir(resolved_path):
                templates.extend(_load_templates_from_dir(resolved_path, get_source_info))
            elif os.path.isfile(resolved_path) and resolved_path.endswith(".md"):
                template = _load_template_from_file(resolved_path, get_source_info(resolved_path))
                if template is not None:
                    templates.append(template)
        except OSError:
            pass  # Ignore read failures

    return templates


_TEMPLATE_INVOCATION_RE = re.compile(r"^/(\S+)(?:\s+([\s\S]*))?$")


def expand_prompt_template(text: str, templates: list[PromptTemplate]) -> str:
    """Expand a prompt template if it matches a template name.

    Returns the expanded content or the original text if not a template.
    """
    if not text.startswith("/"):
        return text

    match = _TEMPLATE_INVOCATION_RE.match(text)
    if match is None:
        return text

    template_name = match.group(1)
    args_string = match.group(2) or ""

    template = next((entry for entry in templates if entry.name == template_name), None)
    if template is not None:
        args = parse_command_args(args_string)
        return substitute_args(template.content, args)

    return text
