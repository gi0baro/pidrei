"""Prompt Customizer

Demonstrates using `event["systemPromptOptions"]` to make informed,
context-aware modifications to the system prompt without re-discovering
resources. The options are the `BuildSystemPromptOptions` the prompt was
built from: selected tools, loaded skills, the user's append prompt, and so
on.

This extension adds tool-specific guidance based on what tools and skills are
currently active, respecting whatever the user has configured.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/prompt_customizer.py
"""


def add_tool_guidance(options, base_prompt: str) -> str:
    """Add tool-specific guidance that adapts to the active tool set.

    Instead of appending one-size-fits-all instructions, this reads what is
    actually loaded and tailors the guidance accordingly.
    """
    selected_tools = options.selected_tools or []

    parts: list[str] = []

    if "read" in selected_tools:
        parts.append("• Use the `read` tool for file contents (supports text and images).")
        parts.append("  - For large files, use `offset` and `limit` to read in chunks.")

    if "bash" in selected_tools:
        parts.append("• Execute commands with the `bash` tool. Use it for file operations like `ls`, `find`, `grep`.")

    if "edit" in selected_tools:
        parts.append(
            "• Use the `edit` tool for precise text replacements in files. Match exact content including whitespace."
        )

    if "write" in selected_tools:
        parts.append("• Use the `write` tool to create new files or overwrite existing ones completely.")

    if options.skills:
        skill_names = ", ".join(skill.name for skill in options.skills)
        parts.append(f"\nAvailable skills: {skill_names}")
        parts.append("Use skill documentation for best practices on specific tools.")

    if not parts:
        return base_prompt

    guidance = "\n".join(parts)
    return f"{base_prompt}\n\n## Tool Guidance\n\n{guidance}\n"


def merge_with_user_append(options) -> str:
    """Merge extension instructions with user-provided append prompts.

    This respects whatever the user configured via --append-system-prompt
    flags or files, rather than duplicating that work.
    """
    extension_specific = """
## Extension-Added Context

This prompt includes tool guidance and skill information loaded dynamically.
If you have additional requirements, configure them via --append-system-prompt or project context files.
"""

    if options.append_system_prompt:
        return f"{options.append_system_prompt}\n\n{extension_specific}"

    return extension_specific


def extension(pi):
    async def on_before_agent_start(event, _ctx):
        options = event["systemPromptOptions"]

        custom_prompt = add_tool_guidance(options, event["systemPrompt"])
        append_section = merge_with_user_append(options)

        return {"systemPrompt": f"{custom_prompt}{append_section}"}

    pi.on("before_agent_start", on_before_agent_start)
