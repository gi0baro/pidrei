"""Claude Rules

Scans the project's .claude/rules/ folder for rule files and lists them in
the system prompt. The agent can then use the read tool to load specific
rules when needed.

Best practices for .claude/rules/:
- Keep rules focused: each file should cover one topic (e.g., testing.md,
  api-design.md)
- Use descriptive filenames: the filename should indicate what the rules
  cover
- Organize with subdirectories: group related rules (e.g., frontend/,
  backend/)

Start pidrei with this extension:
    pidrei -e ./examples/extensions/claude_rules.py

Then create a .claude/rules/ folder in your project root and add .md files
with your rules.
"""

import os

from tonio.colored import fs


async def find_markdown_files(rules_dir: str) -> list[str]:
    """Recursively find all .md files under `rules_dir`, as relative paths."""
    root = fs.Path(rules_dir)
    if not await root.exists():
        return []
    files = []
    for path in await root.rglob("*.md"):
        if await path.is_file():
            files.append(path.relative_to(root).as_posix())
    return sorted(files)


def extension(pi):
    state: dict[str, list[str]] = {"rule_files": []}

    # Scan for rules on session start
    async def on_session_start(_event, ctx):
        rules_dir = os.path.join(ctx.cwd, ".claude", "rules")
        state["rule_files"] = await find_markdown_files(rules_dir)

        if state["rule_files"]:
            ctx.ui.notify(f"Found {len(state['rule_files'])} rule(s) in .claude/rules/", "info")

    # Append available rules to system prompt
    async def on_before_agent_start(event, _ctx):
        if not state["rule_files"]:
            return None

        rules_list = "\n".join(f"- .claude/rules/{f}" for f in state["rule_files"])

        return {
            "systemPrompt": event["systemPrompt"]
            + f"""

## Project Rules

The following project rules are available in .claude/rules/:

{rules_list}

When working on tasks related to these rules, use the read tool to load the relevant rule files for guidance.
"""
        }

    pi.on("session_start", on_session_start)
    pi.on("before_agent_start", on_before_agent_start)
