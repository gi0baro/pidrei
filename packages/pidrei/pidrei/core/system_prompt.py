"""Mirror of pi coding-agent src/core/system-prompt.ts.

System prompt construction and project context loading. The prompt text is
byte-identical to pi's except the harness name ("pi" → "pidrei"), which pi
hardcodes in the template.
"""

from dataclasses import dataclass, field

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
    # Custom system prompt (replaces default).
    custom_prompt: str | None = None
    # Tools to include in prompt. Default: [read, bash, edit, write]
    selected_tools: list[str] | None = None
    # Optional one-line tool snippets keyed by tool name.
    tool_snippets: dict[str, str] | None = None
    # Additional guideline bullets appended to the default system prompt guidelines.
    prompt_guidelines: list[str] | None = None
    # Text to append to system prompt.
    append_system_prompt: str | None = None
    # Pre-loaded context files.
    context_files: list[ContextFile] = field(default_factory=list)
    # Pre-loaded skills.
    skills: list[Skill] = field(default_factory=list)


def _project_context_section(context_files: list[ContextFile]) -> str:
    section = "\n\n<project_context>\n\n"
    section += "Project-specific instructions and guidelines:\n\n"
    for context_file in context_files:
        section += (
            f'<project_instructions path="{context_file.path}">\n{context_file.content}\n</project_instructions>\n\n'
        )
    section += "</project_context>\n"
    return section


def build_system_prompt(options: BuildSystemPromptOptions) -> str:
    """Build the system prompt with tools, guidelines, and context."""
    prompt_cwd = options.cwd
    selected_tools = options.selected_tools
    tool_snippets = options.tool_snippets

    append_section = f"\n\n{options.append_system_prompt}" if options.append_system_prompt else ""

    context_files = options.context_files
    skills = options.skills

    if options.custom_prompt:
        prompt = options.custom_prompt

        if append_section:
            prompt += append_section

        # Append project context files
        if context_files:
            prompt += _project_context_section(context_files)

        # Append skills section (only if read tool is available)
        custom_prompt_has_read = selected_tools is None or "read" in selected_tools
        if custom_prompt_has_read and skills:
            prompt += format_skills_for_prompt(skills)

        prompt += f"\nCurrent working directory: {prompt_cwd}"

        return prompt

    # Get absolute paths to documentation and examples
    readme_path = get_readme_path()
    docs_path = get_docs_path()
    examples_path = get_examples_path()

    # Build tools list based on selected tools.
    # A tool appears in Available tools only when the caller provides a one-line snippet.
    tools = selected_tools if selected_tools is not None else ["read", "bash", "edit", "write"]
    visible_tools = [name for name in tools if tool_snippets and tool_snippets.get(name)]
    tools_list = "\n".join(f"- {name}: {tool_snippets[name]}" for name in visible_tools) if visible_tools else "(none)"

    # Build guidelines based on which tools are actually available
    guidelines_list: list[str] = []
    guidelines_set: set[str] = set()

    def add_guideline(guideline: str) -> None:
        if guideline in guidelines_set:
            return
        guidelines_set.add(guideline)
        guidelines_list.append(guideline)

    has_bash = "bash" in tools
    has_grep = "grep" in tools
    has_find = "find" in tools
    has_ls = "ls" in tools
    has_read = "read" in tools

    # File exploration guidelines
    if has_bash and not has_grep and not has_find and not has_ls:
        add_guideline("Use bash for file operations like ls, rg, find")

    for guideline in options.prompt_guidelines or []:
        normalized = guideline.strip()
        if normalized:
            add_guideline(normalized)

    # Always include these
    add_guideline("Be concise in your responses")
    add_guideline("Show file paths clearly when working with files")

    guidelines = "\n".join(f"- {guideline}" for guideline in guidelines_list)

    prompt = f"""You are an expert coding assistant operating inside pidrei, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.

Available tools:
{tools_list}

In addition to the tools above, you may have access to other custom tools depending on the project.

Guidelines:
{guidelines}

pidrei documentation (read only when the user asks about pidrei itself, its SDK, extensions, themes, skills, or TUI):
- Main documentation: {readme_path}
- Additional docs: {docs_path}
- Examples: {examples_path} (extensions, custom tools, SDK)
- When reading pidrei docs or examples, resolve docs/... under Additional docs and examples/... under Examples, not the current working directory
- When asked about: extensions (docs/extensions.md, examples/extensions/), themes (docs/themes.md), skills (docs/skills.md), prompt templates (docs/prompt-templates.md), TUI components (docs/tui.md), keybindings (docs/keybindings.md), SDK integrations (docs/sdk.md), custom providers (docs/custom-provider.md), adding models (docs/models.md), pidrei packages (docs/packages.md), environment variables (docs/environment-variables.md)
- When working on pidrei topics, read the docs and examples, and follow .md cross-references before implementing
- Always read pidrei .md files completely and follow links to related docs (e.g., tui.md for TUI API details)"""

    if append_section:
        prompt += append_section

    # Append project context files
    if context_files:
        prompt += _project_context_section(context_files)

    # Append skills section (only if read tool is available)
    if has_read and skills:
        prompt += format_skills_for_prompt(skills)

    prompt += f"\nCurrent working directory: {prompt_cwd}"

    return prompt
