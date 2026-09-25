"""Mirror of pi coding-agent src/core/system-prompt.ts.

System prompt construction and project context loading. The prompt text is
byte-identical to pi's except the harness name ("pi" → "pidrei"), which pi
hardcodes in the template, and the PowerShell file-operation rule (the
powershell tool is not ported; PORT_0.84.3.md decision 2).

The prompt is a set of ordered, independently replaceable sections: the
transcript's system messages carry them, and `diff_system_prompt_sections`
turns a prompt change into a section patch.
"""

import dataclasses
import re
from dataclasses import dataclass, field

from pidrei_ai.types import SystemMessage
from pidrei_ai.utils.text import get_system_message_text

from ..config import get_docs_path, get_examples_path, get_readme_path
from .skills import Skill, format_skills_for_prompt


@dataclass(slots=True)
class ContextFile:
    path: str
    content: str


@dataclass(slots=True, kw_only=True)
class BuildSystemPromptOptions:
    # Working directory.
    cwd: str
    # Custom system prompt (replaces the default prefix).
    custom_prompt: str | None = None
    # Exact full prompt replacement set by a before_agent_start handler.
    force_system_prompt: str | None = None
    # Tools to include in prompt. Default: [read, bash, edit, write].
    selected_tools: list[str] | None = None
    # Optional one-line tool snippets keyed by tool name.
    tool_snippets: dict[str, str] | None = None
    # Guideline bullets contributed by each tool, keyed by tool name.
    tool_guidelines: dict[str, list[str]] | None = None
    # Additional guideline bullets appended to the default system prompt rules.
    prompt_guidelines: list[str] | None = None
    # Text appended from user configuration before project context, skills, and cwd.
    append_system_prompt: str | None = None
    # Additional XML-wrapped prompt sections keyed by tag name.
    sections: dict[str, str] | None = None
    # Pre-loaded context files.
    context_files: list[ContextFile] = field(default_factory=list)
    # Pre-loaded skills.
    skills: list[Skill] = field(default_factory=list)


#: `BuildSystemPromptOptions` with every collection present (pi's intersection type);
#: `normalize_build_system_prompt_options` produces it and extensions mutate it in place.
type NormalizedBuildSystemPromptOptions = BuildSystemPromptOptions

#: Ordered system prompt sections, keyed by name. `preamble` is untagged text; every other
#: section is wrapped in a tag of the same name so the model can match later updates to it.
#: These become `SystemMessage.sections` in the transcript.
type SystemPromptSections = dict[str, str]

_SYSTEM_PROMPT_SECTION_NAME = re.compile(r"^[a-z][a-z0-9_-]*$")


def normalize_build_system_prompt_options(input: BuildSystemPromptOptions) -> NormalizedBuildSystemPromptOptions:
    """Normalize prompt input into the mutable, collection-complete shape exposed to extensions."""
    return BuildSystemPromptOptions(
        custom_prompt=input.custom_prompt,
        force_system_prompt=input.force_system_prompt,
        selected_tools=list(
            input.selected_tools if input.selected_tools is not None else ["read", "bash", "edit", "write"]
        ),
        tool_snippets=dict(input.tool_snippets or {}),
        tool_guidelines={name: list(guidelines) for name, guidelines in (input.tool_guidelines or {}).items()},
        prompt_guidelines=list(input.prompt_guidelines or []),
        append_system_prompt=input.append_system_prompt or "",
        sections=dict(input.sections or {}),
        cwd=input.cwd,
        context_files=[dataclasses.replace(context_file) for context_file in input.context_files or []],
        skills=[dataclasses.replace(skill) for skill in input.skills or []],
    )


def _render_project_context(context_files: list[ContextFile]) -> str:
    return "\n\n".join(
        [
            "Project-specific instructions and guidelines:",
            *(
                f'<project_instructions path="{context_file.path}">\n{context_file.content}\n</project_instructions>'
                for context_file in context_files
            ),
        ]
    )


def _build_rules(selected_tools: list[str], tool_guidelines: dict[str, list[str]], prompt_guidelines: list[str]) -> str:
    rules: list[str] = []
    seen: set[str] = set()

    def add_rule(rule: str) -> None:
        normalized = rule.strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        rules.append(normalized)

    has_bash = "bash" in selected_tools
    has_grep = "grep" in selected_tools
    has_find = "find" in selected_tools
    has_ls = "ls" in selected_tools

    if has_bash and not has_grep and not has_find and not has_ls:
        add_rule("Use bash for file operations like ls, rg, find")

    for name in selected_tools:
        for rule in tool_guidelines.get(name, []):
            add_rule(rule)
    for rule in prompt_guidelines:
        add_rule(rule)
    add_rule("Be concise in your responses")
    add_rule("Show file paths clearly when working with files")
    return "\n".join(f"- {rule}" for rule in rules)


def build_system_prompt_sections(input: BuildSystemPromptOptions) -> SystemPromptSections:
    """Build the ordered, independently replaceable sections of the structured system prompt."""
    options = normalize_build_system_prompt_options(input)
    selected_tools = options.selected_tools or []
    tool_snippets = options.tool_snippets or {}
    custom_sections = options.sections or {}

    for name in custom_sections:
        if not _SYSTEM_PROMPT_SECTION_NAME.match(name) or name == "preamble":
            raise Exception(f"Invalid system prompt section name: {name}")

    prompt_sections: dict[str, str] = {}
    if options.custom_prompt:
        prompt_sections["preamble"] = options.custom_prompt
    else:
        prompt_sections["preamble"] = (
            "You are an expert coding assistant operating inside pidrei, a coding agent harness. "
            "You help users by reading files, executing commands, editing code, and writing new files."
        )
        visible_tools = [name for name in selected_tools if tool_snippets.get(name)]
        tools = "\n".join(f"- {name}: {tool_snippets[name]}" for name in visible_tools) if visible_tools else "(none)"
        prompt_sections["tools"] = (
            f"{tools}\n\nIn addition to the tools above, you may have access to other custom tools depending on the project."
        )
        prompt_sections["rules"] = _build_rules(
            selected_tools, options.tool_guidelines or {}, options.prompt_guidelines or []
        )
        prompt_sections[
            "docs"
        ] = f"""pidrei documentation (read only when the user asks about pidrei itself, its SDK, extensions, themes, skills, or TUI):
- Main documentation: {get_readme_path()}
- Additional docs: {get_docs_path()}
- Examples: {get_examples_path()} (extensions, custom tools, SDK)
- When reading pidrei docs or examples, resolve docs/... under Additional docs and examples/... under Examples, not the current working directory
- When asked about: extensions (docs/extensions.md, examples/extensions/), themes (docs/themes.md), skills (docs/skills.md), prompt templates (docs/prompt-templates.md), TUI components (docs/tui.md), keybindings (docs/keybindings.md), SDK integrations (docs/sdk.md), custom providers (docs/custom-provider.md), adding models (docs/models.md), pidrei packages (docs/packages.md), environment variables (docs/environment-variables.md)
- When working on pidrei topics, read the docs and examples, and follow .md cross-references before implementing
- Always read pidrei .md files completely and follow links to related docs (e.g., tui.md for TUI API details)"""

    if options.append_system_prompt:
        prompt_sections["addendum"] = options.append_system_prompt
    if options.context_files:
        prompt_sections["project_context"] = _render_project_context(options.context_files)
    skill_file_read_tool = next((tool for tool in ("read", "bash") if tool in selected_tools), None)
    if skill_file_read_tool and options.skills:
        skills_prompt = format_skills_for_prompt(options.skills, skill_file_read_tool).strip()
        if skills_prompt:
            prompt_sections["skills"] = skills_prompt
    prompt_sections["cwd"] = options.cwd
    for name, content in custom_sections.items():
        if content:
            prompt_sections[name] = content

    sections: SystemPromptSections = {"preamble": prompt_sections["preamble"]}
    for name, content in prompt_sections.items():
        if name != "preamble":
            sections[name] = f"<{name}>\n{content}\n</{name}>"
    return sections


@dataclass(slots=True)
class SystemPromptState:
    content: str
    sections: SystemPromptSections | None = None


def build_system_prompt_state(input: BuildSystemPromptOptions) -> SystemPromptState:
    """The complete prompt state for `input`. A forced prompt is opaque and lives in `content`
    with no sections; otherwise `content` is empty and the structured sections carry the prompt."""
    if input.force_system_prompt is not None:
        return SystemPromptState(content=input.force_system_prompt)
    return SystemPromptState(content="", sections=build_system_prompt_sections(input))


def build_system_prompt(input: BuildSystemPromptOptions) -> str:
    """Build the system prompt text, rendered exactly as the transcript's system message replays it."""
    state = build_system_prompt_state(input)
    return get_system_message_text(SystemMessage(content=state.content, sections=state.sections, timestamp=0))


def diff_system_prompt_sections(
    previous: dict[str, str | None], current: SystemPromptSections
) -> dict[str, str | None] | None:
    """Diff the sections the model currently has (replayed from the transcript, so never null)
    against the desired ones. Returns a `SystemMessage.sections` patch, or None when nothing
    changed."""
    patch: dict[str, str | None] = {}
    for name, text in current.items():
        if previous.get(name) != text:
            patch[name] = text
    for name in previous:
        if name not in current:
            patch[name] = None
    return patch if patch else None
