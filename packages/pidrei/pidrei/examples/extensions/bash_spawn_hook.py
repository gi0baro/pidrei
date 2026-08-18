"""Bash spawn hook.

Adjusts command, cwd, and env before the bash tool executes: sources the
user's profile ahead of every command and marks the environment so scripts
can tell they run under the hook.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/bash_spawn_hook.py
"""

import dataclasses
import os

from pidrei.core.tools.bash import create_bash_tool_definition


def extension(pi):
    cwd = os.getcwd()

    def spawn_hook(context):
        # context is a BashSpawnContext(command, cwd, env); return a modified
        # copy. Leaving a field untouched keeps the tool's default.
        return dataclasses.replace(
            context,
            command=f"source ~/.profile\n{context.command}",
            env={**context.env, "PIDREI_SPAWN_HOOK": "1"},
        )

    # The definition's execute already receives the extension runner context,
    # so it registers as-is — no wrapper needed (pi re-wraps execute to adapt
    # the ctx argument; pidrei's signatures already match).
    pi.register_tool(create_bash_tool_definition(cwd, spawn_hook=spawn_hook))
