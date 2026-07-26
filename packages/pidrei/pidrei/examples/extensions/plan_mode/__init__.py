"""Plan Mode Extension

Read-only exploration mode for safe code analysis. When enabled, built-in
write tools are disabled.

Features:
- /plan command or Ctrl+Alt+P to toggle
- Bash restricted to allowlisted read-only commands
- Extracts numbered plan steps from "Plan:" sections
- [DONE:n] markers to complete steps during execution
- Progress tracking widget during execution

Start pidrei with this extension:
    pidrei -e ./examples/extensions/plan_mode
"""

from typing import Any

from pidrei_tui.keys import Key

from .utils import TodoItem, extract_todo_items, is_safe_command, mark_completed_steps


PLAN_MODE_TOOLS = ["read", "bash", "grep", "find", "ls", "questionnaire"]
NORMAL_MODE_TOOLS = ["read", "bash", "edit", "write"]
PLAN_MODE_DISABLED_TOOLS = {"edit", "write"}
PLAN_MANAGED_TOOLS = {*PLAN_MODE_TOOLS, *NORMAL_MODE_TOOLS}


def _is_assistant_message(message: Any) -> bool:
    role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    return role == "assistant" and isinstance(content, list)


def _get_text_content(message: Any) -> str:
    content = message.get("content") if isinstance(message, dict) else message.content
    texts = []
    for block in content:
        block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if block_type == "text":
            texts.append(block.get("text") if isinstance(block, dict) else block.text)
    return "\n".join(texts)


def _unique(names: list[str]) -> list[str]:
    return list(dict.fromkeys(names))


class PlanMode:
    """pi keeps all of this in one factory closure, which is the JS idiom;
    methods on a small object read better in Python and keep each piece
    independently testable."""

    def __init__(self, pi):
        self.pi = pi
        self.plan_mode_enabled = False
        self.execution_mode = False
        self.todo_items: list[TodoItem] = []
        self.tools_before_plan_mode: list[str] | None = None

    def wire(self) -> None:
        self.pi.register_flag(
            "plan", type="boolean", description="Start in plan mode (read-only exploration)", default=False
        )
        self.pi.register_command(
            "plan", description="Toggle plan mode (read-only exploration)", handler=self.plan_command
        )
        self.pi.register_command("todos", description="Show current plan todo list", handler=self.todos_command)
        self.pi.register_shortcut(Key.ctrl_alt("p"), description="Toggle plan mode", handler=self.toggle_plan_mode)
        self.pi.on("tool_call", self.on_tool_call)
        self.pi.on("context", self.on_context)
        self.pi.on("before_agent_start", self.on_before_agent_start)
        self.pi.on("turn_end", self.on_turn_end)
        self.pi.on("agent_end", self.on_agent_end)
        self.pi.on("session_start", self.on_session_start)

    def update_status(self, ctx) -> None:
        todo_items: list[TodoItem] = self.todo_items

        # Footer status
        if self.execution_mode and todo_items:
            completed = sum(1 for item in todo_items if item.completed)
            ctx.ui.set_status("plan-mode", ctx.ui.theme.fg("accent", f"📋 {completed}/{len(todo_items)}"))
        elif self.plan_mode_enabled:
            ctx.ui.set_status("plan-mode", ctx.ui.theme.fg("warning", "⏸ plan"))
        else:
            ctx.ui.set_status("plan-mode", None)

        # Widget showing the todo list
        if self.execution_mode and todo_items:
            lines = []
            for item in todo_items:
                if item.completed:
                    lines.append(
                        ctx.ui.theme.fg("success", "☑ ")
                        + ctx.ui.theme.fg("muted", ctx.ui.theme.strikethrough(item.text))
                    )
                else:
                    lines.append(f"{ctx.ui.theme.fg('muted', '☐ ')}{item.text}")
            ctx.ui.set_widget("plan-todos", lines)
        else:
            ctx.ui.set_widget("plan-todos", None)

    def get_plan_mode_tools(self, active_tool_names: list[str]) -> list[str]:
        return _unique(
            [*(name for name in active_tool_names if name not in PLAN_MODE_DISABLED_TOOLS), *PLAN_MODE_TOOLS]
        )

    def get_normal_mode_tools(self, active_tool_names: list[str]) -> list[str]:
        return _unique([*NORMAL_MODE_TOOLS, *(name for name in active_tool_names if name not in PLAN_MANAGED_TOOLS)])

    def enable_plan_mode_tools(self) -> None:
        if self.tools_before_plan_mode is None:
            self.tools_before_plan_mode = self.pi.get_active_tools()
        self.pi.set_active_tools(self.get_plan_mode_tools(self.tools_before_plan_mode))

    def restore_normal_mode_tools(self) -> None:
        self.pi.set_active_tools(self.tools_before_plan_mode or self.get_normal_mode_tools(self.pi.get_active_tools()))
        self.tools_before_plan_mode = None

    def persist_state(self) -> None:
        self.pi.append_entry(
            "plan-mode",
            {
                "enabled": self.plan_mode_enabled,
                "todos": [
                    {"step": item.step, "text": item.text, "completed": item.completed} for item in self.todo_items
                ],
                "executing": self.execution_mode,
                "toolsBeforePlanMode": self.tools_before_plan_mode,
            },
        )

    def toggle_plan_mode(self, ctx) -> None:
        self.plan_mode_enabled = not self.plan_mode_enabled
        self.execution_mode = False
        self.todo_items = []

        if self.plan_mode_enabled:
            self.enable_plan_mode_tools()
            ctx.ui.notify("Plan mode enabled. Built-in write tools disabled.")
        else:
            self.restore_normal_mode_tools()
            ctx.ui.notify("Plan mode disabled. Full access restored.")
        self.update_status(ctx)
        self.persist_state()

    async def plan_command(self, _args: str, ctx) -> None:
        self.toggle_plan_mode(ctx)

    async def todos_command(self, _args: str, ctx) -> None:
        todo_items: list[TodoItem] = self.todo_items
        if not todo_items:
            ctx.ui.notify("No todos. Create a plan first with /plan", "info")
            return
        listing = "\n".join(
            f"{index + 1}. {'✓' if item.completed else '○'} {item.text}" for index, item in enumerate(todo_items)
        )
        ctx.ui.notify(f"Plan Progress:\n{listing}", "info")

    async def on_tool_call(self, event, _ctx):
        """Block destructive bash commands in plan mode."""
        if not self.plan_mode_enabled or event.get("toolName") != "bash":
            return None

        command = event.get("input", {}).get("command")
        if not is_safe_command(command):
            return {
                "block": True,
                "reason": (
                    "Plan mode: command blocked (not allowlisted). Use /plan to disable plan mode first.\n"
                    f"Command: {command}"
                ),
            }
        return None

    async def on_context(self, event, _ctx):
        """Filter out stale plan mode context when not in plan mode."""
        if self.plan_mode_enabled:
            return None

        def keep(message: Any) -> bool:
            custom_type = (
                message.get("customType") if isinstance(message, dict) else getattr(message, "custom_type", None)
            )
            if custom_type == "plan-mode-context":
                return False
            role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
            if role != "user":
                return True

            content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
            if isinstance(content, str):
                return "[PLAN MODE ACTIVE]" not in content
            if isinstance(content, list):
                for block in content:
                    block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
                    text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
                    if block_type == "text" and text and "[PLAN MODE ACTIVE]" in text:
                        return False
            return True

        return {"messages": [message for message in event["messages"] if keep(message)]}

    async def on_before_agent_start(self, _event, _ctx):
        """Inject plan/execution context before the agent starts."""
        if self.plan_mode_enabled:
            return {
                "message": {
                    "customType": "plan-mode-context",
                    "content": """[PLAN MODE ACTIVE]
    You are in plan mode - a read-only exploration mode for safe code analysis.

    Restrictions:
    - Built-in edit and write tools are disabled
    - Other currently active tools remain available
    - Bash is restricted to an allowlist of read-only commands

    Ask clarifying questions using the questionnaire tool.
    Use brave-search skill via bash for web research.

    Create a detailed numbered plan under a "Plan:" header:

    Plan:
    1. First step description
    2. Second step description
    ...

    Do NOT attempt to make changes - just describe what you would do.""",
                    "display": False,
                }
            }

        todo_items: list[TodoItem] = self.todo_items
        if self.execution_mode and todo_items:
            remaining = [item for item in todo_items if not item.completed]
            todo_list = "\n".join(f"{item.step}. {item.text}" for item in remaining)
            return {
                "message": {
                    "customType": "plan-execution-context",
                    "content": f"""[EXECUTING PLAN - Full tool access enabled]

    Remaining steps:
    {todo_list}

    Execute each step in order.
    After completing a step, include a [DONE:n] tag in your response.""",
                    "display": False,
                }
            }
        return None

    async def on_turn_end(self, event, ctx) -> None:
        """Track progress after each turn."""
        todo_items: list[TodoItem] = self.todo_items
        if not self.execution_mode or not todo_items:
            return
        message = event.get("message")
        if not _is_assistant_message(message):
            return

        if mark_completed_steps(_get_text_content(message), todo_items) > 0:
            self.update_status(ctx)
        self.persist_state()

    async def on_agent_end(self, event, ctx) -> None:
        """Handle plan completion and the plan mode prompt."""
        todo_items: list[TodoItem] = self.todo_items

        if self.execution_mode and todo_items:
            if all(item.completed for item in todo_items):
                completed_list = "\n".join(f"~~{item.text}~~" for item in todo_items)
                self.pi.send_message(
                    {
                        "customType": "plan-complete",
                        "content": f"**Plan Complete!** ✓\n\n{completed_list}",
                        "display": True,
                    },
                    {"triggerTurn": False},
                )
                self.execution_mode = False
                self.todo_items = []
                self.update_status(ctx)
                # Save the cleared state so a resume does not restore it.
                self.persist_state()
            return

        if not self.plan_mode_enabled or not ctx.has_ui:
            return

        last_assistant = next(
            (message for message in reversed(event["messages"]) if _is_assistant_message(message)), None
        )
        if last_assistant is not None:
            extracted = extract_todo_items(_get_text_content(last_assistant))
            if extracted:
                self.todo_items = extracted

        todo_items = self.todo_items
        if not todo_items:
            return
        self.persist_state()

        todo_list_text = "\n".join(f"{index + 1}. ☐ {item.text}" for index, item in enumerate(todo_items))
        plan_todo_list_message = {
            "customType": "plan-todo-list",
            "content": f"**Plan Steps ({len(todo_items)}):**\n\n{todo_list_text}",
            "display": True,
        }

        choice = await ctx.ui.select(
            "Plan mode - what next?",
            ["Execute the plan (track progress)", "Stay in plan mode", "Refine the plan"],
        )

        if choice and choice.startswith("Execute"):
            first_item = todo_items[0]
            self.plan_mode_enabled = False
            self.execution_mode = True
            self.restore_normal_mode_tools()
            self.update_status(ctx)
            self.persist_state()

            remaining_list = "\n".join(f"{item.step}. {item.text}" for item in todo_items)
            exec_message = f"""Execute the plan.

    Remaining steps:
    {remaining_list}

    Start with: {first_item.text}
    After completing a step, include a [DONE:n] tag in your response."""
            self.pi.send_message(plan_todo_list_message, {"deliverAs": "followUp"})
            self.pi.send_message(
                {"customType": "plan-mode-execute", "content": exec_message, "display": True},
                {"triggerTurn": True, "deliverAs": "followUp"},
            )
        elif choice == "Refine the plan":
            refinement = await ctx.ui.editor("Refine the plan:", "")
            if refinement and refinement.strip():
                self.pi.send_message(plan_todo_list_message, {"deliverAs": "followUp"})
                self.pi.send_user_message(refinement.strip(), {"deliverAs": "followUp"})

    async def on_session_start(self, _event, ctx) -> None:
        """Restore state on session start/resume."""
        if self.pi.get_flag("plan") is True:
            self.plan_mode_enabled = True

        entries = ctx.session_manager.get_entries()

        plan_mode_entries = [
            entry for entry in entries if entry.get("type") == "custom" and entry.get("customType") == "plan-mode"
        ]
        plan_mode_entry = plan_mode_entries[-1] if plan_mode_entries else None

        if plan_mode_entry is not None and plan_mode_entry.get("data"):
            data = plan_mode_entry["data"]
            self.plan_mode_enabled = data.get("enabled", self.plan_mode_enabled)
            if data.get("todos") is not None:
                self.todo_items = [
                    TodoItem(step=todo["step"], text=todo["text"], completed=todo.get("completed", False))
                    for todo in data["todos"]
                ]
            self.execution_mode = data.get("executing", self.execution_mode)
            self.tools_before_plan_mode = data.get("toolsBeforePlanMode", self.tools_before_plan_mode)

        # On resume, re-scan messages to rebuild completion state — but only
        # those after the last plan-mode-execute, so [DONE:n] markers from a
        # previous plan are not picked up.
        if plan_mode_entry is not None and self.execution_mode and self.todo_items:
            execute_index = -1
            for index in range(len(entries) - 1, -1, -1):
                if entries[index].get("customType") == "plan-mode-execute":
                    execute_index = index
                    break

            messages = [
                entry["message"]
                for entry in entries[execute_index + 1 :]
                if entry.get("type") == "message" and "message" in entry and _is_assistant_message(entry["message"])
            ]
            mark_completed_steps("\n".join(_get_text_content(m) for m in messages), self.todo_items)

        if self.plan_mode_enabled:
            self.enable_plan_mode_tools()
        self.update_status(ctx)


def extension(pi):
    PlanMode(pi).wire()
