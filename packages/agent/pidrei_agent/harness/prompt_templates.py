"""Prompt template loading and invocation (port of pi `harness/prompt-templates.ts`).

YAML frontmatter parses with PyYAML (`safe_load`) instead of the npm `yaml`
package; error message text differs, structure does not.
"""

import re
from dataclasses import dataclass
from typing import Any, Literal

import yaml

from pidrei_ai.utils.cancel import CancelToken

from .types import FileInfo, to_error


type PromptTemplateDiagnosticCode = Literal["file_info_failed", "list_failed", "read_failed", "parse_failed"]


@dataclass(slots=True)
class PromptTemplate:
    """Prompt template that can be formatted into a prompt for explicit invocation."""

    # Stable template name used for lookup or application command routing.
    name: str
    # Template content. Argument placeholders are formatted by `format_prompt_template_invocation`.
    content: str
    # Optional description for command lists or autocomplete.
    description: str | None = None


@dataclass(slots=True, kw_only=True)
class PromptTemplateDiagnostic:
    """Warning produced while loading prompt templates."""

    code: PromptTemplateDiagnosticCode
    message: str
    path: str
    type: Literal["warning"] = "warning"


@dataclass(slots=True)
class LoadedPromptTemplates:
    prompt_templates: list[PromptTemplate]
    diagnostics: list[PromptTemplateDiagnostic]


@dataclass(slots=True)
class SourcedPromptTemplate:
    prompt_template: PromptTemplate
    source: Any


@dataclass(slots=True)
class SourcedPromptTemplateDiagnostic(PromptTemplateDiagnostic):
    source: Any = None


@dataclass(slots=True)
class LoadedSourcedPromptTemplates:
    prompt_templates: list[SourcedPromptTemplate]
    diagnostics: list[SourcedPromptTemplateDiagnostic]


async def load_prompt_templates(
    env, paths: str | list[str], cancel: CancelToken | None = None
) -> LoadedPromptTemplates:
    """Load prompt templates from one or more paths.

    Directory inputs load direct `.md` children non-recursively. File inputs
    load explicit `.md` files. Missing paths and non-markdown files are
    skipped. Read and parse failures are returned as diagnostics.
    """
    prompt_templates: list[PromptTemplate] = []
    diagnostics: list[PromptTemplateDiagnostic] = []
    for path in paths if isinstance(paths, list) else [paths]:
        info_result = await env.file_info(path, cancel)
        if not info_result.ok:
            if info_result.error.code != "not_found":
                diagnostics.append(
                    PromptTemplateDiagnostic(code="file_info_failed", message=info_result.error.message, path=path)
                )
            continue
        info = info_result.value
        kind = await _resolve_kind(env, info, diagnostics, cancel)
        if kind == "directory":
            result = await _load_templates_from_dir(env, info.path, cancel)
            prompt_templates.extend(result.prompt_templates)
            diagnostics.extend(result.diagnostics)
        elif kind == "file" and info.name.endswith(".md"):
            template, file_diagnostics = await _load_template_from_file(env, info.path, info.name, cancel)
            if template is not None:
                prompt_templates.append(template)
            diagnostics.extend(file_diagnostics)
    return LoadedPromptTemplates(prompt_templates=prompt_templates, diagnostics=diagnostics)


async def load_sourced_prompt_templates(
    env,
    inputs: list[dict[str, Any]],
    map_prompt_template=None,
    cancel: CancelToken | None = None,
) -> LoadedSourcedPromptTemplates:
    """Load prompt templates from source-tagged paths.

    Source values are preserved exactly and attached to every loaded prompt
    template and diagnostic. `map_prompt_template` receives
    `(prompt_template, source, cancel)`.
    """
    prompt_templates: list[SourcedPromptTemplate] = []
    diagnostics: list[SourcedPromptTemplateDiagnostic] = []
    for entry in inputs:
        result = await load_prompt_templates(env, entry["path"], cancel)
        for prompt_template in result.prompt_templates:
            mapped = (
                map_prompt_template(prompt_template, entry["source"], cancel)
                if map_prompt_template
                else prompt_template
            )
            prompt_templates.append(SourcedPromptTemplate(prompt_template=mapped, source=entry["source"]))
        for diagnostic in result.diagnostics:
            diagnostics.append(
                SourcedPromptTemplateDiagnostic(
                    code=diagnostic.code, message=diagnostic.message, path=diagnostic.path, source=entry["source"]
                )
            )
    return LoadedSourcedPromptTemplates(prompt_templates=prompt_templates, diagnostics=diagnostics)


async def _load_templates_from_dir(env, directory: str, cancel: CancelToken | None) -> LoadedPromptTemplates:
    prompt_templates: list[PromptTemplate] = []
    diagnostics: list[PromptTemplateDiagnostic] = []
    entries_result = await env.list_dir(directory, cancel)
    if not entries_result.ok:
        diagnostics.append(
            PromptTemplateDiagnostic(code="list_failed", message=entries_result.error.message, path=directory)
        )
        return LoadedPromptTemplates(prompt_templates=prompt_templates, diagnostics=diagnostics)

    for entry in sorted(entries_result.value, key=lambda info: info.name):
        kind = await _resolve_kind(env, entry, diagnostics, cancel)
        if kind != "file" or not entry.name.endswith(".md"):
            continue
        template, file_diagnostics = await _load_template_from_file(env, entry.path, entry.name, cancel)
        if template is not None:
            prompt_templates.append(template)
        diagnostics.extend(file_diagnostics)
    return LoadedPromptTemplates(prompt_templates=prompt_templates, diagnostics=diagnostics)


async def _load_template_from_file(
    env, file_path: str, file_name: str, cancel: CancelToken | None
) -> tuple[PromptTemplate | None, list[PromptTemplateDiagnostic]]:
    diagnostics: list[PromptTemplateDiagnostic] = []
    raw_content = await env.read_text_file(file_path, cancel)
    if not raw_content.ok:
        diagnostics.append(
            PromptTemplateDiagnostic(code="read_failed", message=raw_content.error.message, path=file_path)
        )
        return None, diagnostics

    try:
        frontmatter, body = parse_frontmatter(raw_content.value)
    except Exception as error:
        diagnostics.append(PromptTemplateDiagnostic(code="parse_failed", message=str(to_error(error)), path=file_path))
        return None, diagnostics

    first_line = next((line for line in body.split("\n") if line.strip()), None)
    description = frontmatter.get("description") if isinstance(frontmatter.get("description"), str) else ""
    if not description and first_line:
        description = first_line[:60]
        if len(first_line) > 60:
            description += "..."
    return (
        PromptTemplate(
            name=re.sub(r"\.md$", "", file_name, flags=re.IGNORECASE),
            description=description,
            content=body,
        ),
        diagnostics,
    )


async def _resolve_kind(env, info: FileInfo, diagnostics: list, cancel: CancelToken | None) -> str | None:
    if info.kind in ("file", "directory"):
        return info.kind
    canonical_path = await env.canonical_path(info.path, cancel)
    if not canonical_path.ok:
        if canonical_path.error.code != "not_found":
            diagnostics.append(
                PromptTemplateDiagnostic(code="file_info_failed", message=canonical_path.error.message, path=info.path)
            )
        return None
    target = await env.file_info(canonical_path.value, cancel)
    if not target.ok:
        if target.error.code != "not_found":
            diagnostics.append(
                PromptTemplateDiagnostic(code="file_info_failed", message=target.error.message, path=info.path)
            )
        return None
    return target.value.kind if target.value.kind in ("file", "directory") else None


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Split `---` YAML frontmatter from the body; raises on invalid YAML."""
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.startswith("---"):
        return {}, normalized
    end_index = normalized.find("\n---", 3)
    if end_index == -1:
        return {}, normalized
    yaml_string = normalized[4:end_index]
    body = normalized[end_index + 4 :].strip()
    parsed = yaml.safe_load(yaml_string)
    return (parsed if parsed is not None else {}), body


def parse_command_args(args_string: str) -> list[str]:
    """Parse an argument string using simple shell-style single and double quotes."""
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
        elif char in (" ", "\t"):
            if current:
                args.append(current)
                current = ""
        else:
            current += char
    if current:
        args.append(current)
    return args


def substitute_args(content: str, args: list[str]) -> str:
    """Substitute placeholders (`$1`, `$@`, `$ARGUMENTS`, `${@:N}`, `${@:N:L}`) with command arguments."""
    result = re.sub(
        r"\$(\d+)", lambda m: args[int(m.group(1)) - 1] if 0 < int(m.group(1)) <= len(args) else "", content
    )

    def slice_replacement(match: re.Match) -> str:
        start = max(0, int(match.group(1)) - 1)
        if match.group(2):
            return " ".join(args[start : start + int(match.group(2))])
        return " ".join(args[start:])

    result = re.sub(r"\$\{@:(\d+)(?::(\d+))?\}", slice_replacement, result)
    all_args = " ".join(args)
    result = result.replace("$ARGUMENTS", all_args)
    result = result.replace("$@", all_args)
    return result


def format_prompt_template_invocation(template: PromptTemplate, args: list[str] | None = None) -> str:
    """Format a prompt template invocation with positional arguments."""
    return substitute_args(template.content, args if args is not None else [])
