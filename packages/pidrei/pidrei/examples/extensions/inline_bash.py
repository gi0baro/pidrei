"""Inline Bash

Expands inline bash commands in user prompts.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/inline_bash.py

Then type prompts with inline bash:
    What's in !{pwd}?
    The current branch is !{git branch --show-current} and status: !{git status --short}
    My python version is !{python3 --version}

The !{command} patterns are executed and replaced with their output before
the prompt is sent to the agent.

Note: Regular !command syntax (whole-line bash) is preserved and works as
before.
"""

import re
from dataclasses import dataclass


PATTERN = re.compile(r"!\{([^}]+)\}")
TIMEOUT_MS = 30_000


@dataclass(slots=True)
class Expansion:
    command: str
    output: str
    error: str | None = None


def extension(pi):
    async def on_input(event, ctx):
        text = event["text"]

        # Don't process if it's a whole-line bash command (starts with !)
        # This preserves the existing !command behavior
        stripped = text.lstrip()
        if stripped.startswith("!") and not stripped.startswith("!{"):
            return {"action": "continue"}

        matches = PATTERN.findall(text)
        if not matches:
            return {"action": "continue"}

        # Execute each command and collect results. pi.exec never raises:
        # spawn errors resolve with code 1, timeouts with killed=True.
        result = text
        expansions: list[Expansion] = []
        for command in matches:
            bash_result = await pi.exec("bash", ["-c", command], timeout=TIMEOUT_MS)

            output = bash_result.stdout or bash_result.stderr or ""
            trimmed = output.strip()

            if bash_result.killed:
                expansions.append(Expansion(command=command, output="", error="timed out"))
                result = result.replace(f"!{{{command}}}", "[error: timed out]")
                continue

            if bash_result.code != 0 and bash_result.stderr:
                expansions.append(Expansion(command=command, output=trimmed, error=f"exit code {bash_result.code}"))
            else:
                expansions.append(Expansion(command=command, output=trimmed))

            result = result.replace(f"!{{{command}}}", trimmed)

        # Show what was expanded (if UI available)
        if ctx.has_ui and expansions:
            lines = []
            for expansion in expansions:
                status = f" ({expansion.error})" if expansion.error else ""
                preview = f"{expansion.output[:50]}..." if len(expansion.output) > 50 else expansion.output
                lines.append(f'!{{{expansion.command}}}{status} -> "{preview}"')
            summary = "\n".join(lines)
            ctx.ui.notify(f"Expanded {len(expansions)} inline command(s):\n{summary}", "info")

        return {"action": "transform", "text": result, "images": event["images"]}

    pi.on("input", on_input)
