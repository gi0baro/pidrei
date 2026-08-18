"""Interactive Shell Commands

Enables running interactive commands (vim, git rebase -i, htop, etc.) from
`!` input with full terminal access. The TUI suspends while they run.

    !vim file.txt        # Auto-detected as interactive
    !i any-command       # Force interactive mode with !i prefix
    !git rebase -i HEAD~3
    !htop

Configuration via environment variables:
    INTERACTIVE_COMMANDS - Additional commands (comma-separated)
    INTERACTIVE_EXCLUDE  - Commands to exclude (comma-separated)

Note: this only intercepts user `!` commands, not agent bash tool calls.
If the agent runs an interactive command, it will fail (which is fine).

Start pidrei with this extension:
    pidrei -e ./examples/extensions/interactive_shell.py
"""

import os
import sys

import tonio.colored as tonio


# Default interactive commands - editors, pagers, git ops, TUIs
DEFAULT_INTERACTIVE_COMMANDS = [
    # Editors
    "vim",
    "nvim",
    "vi",
    "nano",
    "emacs",
    "pico",
    "micro",
    "helix",
    "hx",
    "kak",
    # Pagers
    "less",
    "more",
    "most",
    # Git interactive
    "git commit",
    "git rebase",
    "git merge",
    "git cherry-pick",
    "git revert",
    "git add -p",
    "git add --patch",
    "git add -i",
    "git add --interactive",
    "git stash -p",
    "git stash --patch",
    "git reset -p",
    "git reset --patch",
    "git checkout -p",
    "git checkout --patch",
    "git difftool",
    "git mergetool",
    # System monitors
    "htop",
    "top",
    "btop",
    "glances",
    # File managers
    "ranger",
    "nnn",
    "lf",
    "mc",
    "vifm",
    # Git TUIs
    "tig",
    "lazygit",
    "gitui",
    # Fuzzy finders
    "fzf",
    "sk",
    # Remote sessions
    "ssh",
    "telnet",
    "mosh",
    # Database clients
    "psql",
    "mysql",
    "sqlite3",
    "mongosh",
    "redis-cli",
    # Kubernetes/Docker
    "kubectl edit",
    "kubectl exec -it",
    "docker exec -it",
    "docker run -it",
    # Other
    "tmux",
    "screen",
    "ncdu",
]


def get_interactive_commands() -> list[str]:
    additional = [part.strip() for part in os.environ.get("INTERACTIVE_COMMANDS", "").split(",") if part.strip()]
    excluded = {part.strip().lower() for part in os.environ.get("INTERACTIVE_EXCLUDE", "").split(",") if part.strip()}
    return [cmd for cmd in [*DEFAULT_INTERACTIVE_COMMANDS, *additional] if cmd.lower() not in excluded]


def is_interactive_command(command: str) -> bool:
    trimmed = command.strip().lower()

    for cmd in get_interactive_commands():
        cmd_lower = cmd.lower()
        # Match at start
        if trimmed == cmd_lower or trimmed.startswith((f"{cmd_lower} ", f"{cmd_lower}\t")):
            return True
        # Match after pipe: "cat file | less"
        pipe_idx = trimmed.rfind("|")
        if pipe_idx != -1:
            after_pipe = trimmed[pipe_idx + 1 :].strip()
            if after_pipe == cmd_lower or after_pipe.startswith(f"{cmd_lower} "):
                return True
    return False


def extension(pi):
    async def on_user_bash(event, ctx):
        command = event["command"]
        force_interactive = False

        # Check for !i prefix (command comes without the leading !)
        # The prefix parsing happens before this event, so we check if command
        # starts with "i "
        if command.startswith(("i ", "i\t")):
            force_interactive = True
            command = command[2:].strip()

        if not (force_interactive or is_interactive_command(command)):
            return None  # Let normal handling proceed

        # No UI available (print mode, RPC, etc.)
        if ctx.mode != "tui":
            return {
                "result": {
                    "output": "(interactive commands require TUI)",
                    "exitCode": 1,
                    "cancelled": False,
                    "truncated": False,
                }
            }

        # Use ctx.ui.custom() to get TUI access, then run the command
        async def run_in_terminal(tui, _theme, _keybindings, done):
            # Stop TUI to release the terminal
            await tui.stop()

            # Clear screen
            sys.stdout.write("\x1b[2J\x1b[H")
            sys.stdout.flush()

            # Run command with full terminal access. The process inherits
            # stdio, and the runtime waits for it without blocking the loop —
            # this is the same pattern pidrei's own external editor uses.
            shell = os.environ.get("SHELL") or "/bin/sh"
            try:
                process = await tonio.open_process([shell, "-c", command])
                exit_code = await process.wait()
            except Exception:
                exit_code = None

            # Restart TUI
            await tui.start()
            tui.request_render(True)

            # Signal completion; no component to show since we are done
            done(exit_code)

        exit_code = await ctx.ui.custom(run_in_terminal)

        # Return result to prevent default bash handling
        output = (
            "(interactive command completed successfully)"
            if exit_code == 0
            else f"(interactive command exited with code {exit_code})"
        )
        return {
            "result": {
                "output": output,
                "exitCode": exit_code if exit_code is not None else 1,
                "cancelled": False,
                "truncated": False,
            }
        }

    pi.on("user_bash", on_user_bash)
