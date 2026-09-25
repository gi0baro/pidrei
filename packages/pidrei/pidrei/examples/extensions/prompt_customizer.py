"""Prompt Customizer

Demonstrates using `event["systemPromptOptions"]` to add context-aware prompt
sections without replacing or reparsing the complete rendered prompt.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/prompt_customizer.py
"""


def build_tool_guidance(options) -> str:
    selected_tools = options.selected_tools or []
    rules: list[str] = []

    if "read" in selected_tools:
        rules.append("- Use `read` for file contents; it supports text and images.")
        rules.append("- For large files, use `offset` and `limit` to read in chunks.")
    if "bash" in selected_tools:
        rules.append("- Use `bash` for file operations such as `ls`, `find`, and `grep`.")
    if "edit" in selected_tools:
        rules.append("- Use `edit` for precise text replacements that match existing content exactly.")
    if "write" in selected_tools:
        rules.append("- Use `write` to create new files or replace existing files completely.")
    if options.skills:
        rules.append(f"- Available skills: {', '.join(skill.name for skill in options.skills)}.")

    return "\n".join(rules)


def extension(pi):
    async def on_before_agent_start(event, _ctx):
        options = event["systemPromptOptions"]
        guidance = build_tool_guidance(options)
        if guidance:
            options.sections["tool_guidance"] = guidance
        else:
            options.sections.pop("tool_guidance", None)

    pi.on("before_agent_start", on_before_agent_start)
