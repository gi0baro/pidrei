"""Todo list — state management via session entries.

This extension:
- Registers a `todo` tool for the LLM to manage todos
- Registers a `/todos` command for users to view the list

State is stored in tool result details (not external files), which allows
proper branching - when you branch, the todo state is automatically correct
for that point in history.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/todo.py
"""

from pidrei.core.extensions.types import ToolDefinition
from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent
from pidrei_tui import Text, matches_key, truncate_to_width


TODO_PARAMS = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["list", "add", "toggle", "clear"]},
        "text": {"type": "string", "description": "Todo text (for add)"},
        "id": {"type": "number", "description": "Todo ID (for toggle)"},
    },
    "required": ["action"],
}


class TodoListComponent:
    """UI component for the /todos command."""

    def __init__(self, todos: list[dict], theme, on_close) -> None:
        self._todos = todos
        self._theme = theme
        self._on_close = on_close
        self._cached_width: int | None = None
        self._cached_lines: list[str] | None = None

    async def handle_input(self, data: str) -> None:
        if matches_key(data, "escape") or matches_key(data, "ctrl+c"):
            self._on_close()

    def render(self, width: int) -> list[str]:
        if self._cached_lines is not None and self._cached_width == width:
            return self._cached_lines

        th = self._theme
        lines: list[str] = [""]

        title = th.fg("accent", " Todos ")
        header_line = th.fg("borderMuted", "─" * 3) + title + th.fg("borderMuted", "─" * max(0, width - 10))
        lines.append(truncate_to_width(header_line, width))
        lines.append("")

        if not self._todos:
            lines.append(truncate_to_width(f"  {th.fg('dim', 'No todos yet. Ask the agent to add some!')}", width))
        else:
            done = sum(1 for t in self._todos if t["done"])
            lines.append(truncate_to_width(f"  {th.fg('muted', f'{done}/{len(self._todos)} completed')}", width))
            lines.append("")

            for todo in self._todos:
                check = th.fg("success", "✓") if todo["done"] else th.fg("dim", "○")
                todo_id = th.fg("accent", f"#{todo['id']}")
                text = th.fg("dim", todo["text"]) if todo["done"] else th.fg("text", todo["text"])
                lines.append(truncate_to_width(f"  {check} {todo_id} {text}", width))

        lines.append("")
        lines.append(truncate_to_width(f"  {th.fg('dim', 'Press Escape to close')}", width))
        lines.append("")

        self._cached_width = width
        self._cached_lines = lines
        return lines

    def invalidate(self) -> None:
        self._cached_width = None
        self._cached_lines = None


def extension(pi):
    # In-memory state (reconstructed from session on load). The `nextId` key
    # matches the persisted details, so state round-trips unchanged.
    state: dict = {"todos": [], "nextId": 1}

    def details(action: str, error: str | None = None) -> dict:
        d = {"action": action, "todos": [dict(t) for t in state["todos"]], "nextId": state["nextId"]}
        if error is not None:
            d["error"] = error
        return d

    def reconstruct_state(ctx) -> None:
        """Reconstruct state from session entries.

        Scans tool results for this tool and applies them in order."""
        state["todos"] = []
        state["nextId"] = 1

        for entry in ctx.session_manager.get_branch():
            if entry.get("type") != "message":
                continue
            msg = entry.get("message")
            if getattr(msg, "role", None) != "toolResult" or msg.tool_name != "todo":
                continue

            if msg.details:
                state["todos"] = [dict(t) for t in msg.details["todos"]]
                state["nextId"] = msg.details["nextId"]

    # Reconstruct state on session events
    async def on_session_event(_event, ctx) -> None:
        reconstruct_state(ctx)

    pi.on("session_start", on_session_event)
    pi.on("session_tree", on_session_event)

    def result(text: str, action: str, error: str | None = None) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text=text)], details=details(action, error))

    # The tool the LLM calls
    async def execute(_tool_call_id, params, _cancel=None, _on_update=None, _ctx=None):
        action = params["action"]
        todos: list[dict] = state["todos"]

        if action == "list":
            listing = (
                "\n".join(f"[{'x' if t['done'] else ' '}] #{t['id']}: {t['text']}" for t in todos)
                if todos
                else "No todos"
            )
            return result(listing, "list")

        if action == "add":
            if not params.get("text"):
                return result("Error: text required for add", "add", error="text required")
            new_todo = {"id": state["nextId"], "text": params["text"], "done": False}
            state["nextId"] += 1
            todos.append(new_todo)
            return result(f"Added todo #{new_todo['id']}: {new_todo['text']}", "add")

        if action == "toggle":
            todo_id = params.get("id")
            if todo_id is None:
                return result("Error: id required for toggle", "toggle", error="id required")
            todo = next((t for t in todos if t["id"] == todo_id), None)
            if todo is None:
                return result(f"Todo #{todo_id} not found", "toggle", error=f"#{todo_id} not found")
            todo["done"] = not todo["done"]
            return result(f"Todo #{todo['id']} {'completed' if todo['done'] else 'uncompleted'}", "toggle")

        if action == "clear":
            count = len(todos)
            state["todos"] = []
            state["nextId"] = 1
            return result(f"Cleared {count} todos", "clear")

        return result(f"Unknown action: {action}", "list", error=f"unknown action: {action}")

    def render_call(args, theme, _context):
        args = args or {}
        text = theme.fg("toolTitle", theme.bold("todo ")) + theme.fg("muted", args.get("action", ""))
        if args.get("text"):
            text += " " + theme.fg("dim", f'"{args["text"]}"')
        if args.get("id") is not None:
            text += " " + theme.fg("accent", f"#{args['id']}")
        return Text(text, 0, 0)

    def render_result(result, options, theme, _context):
        result_details = result["details"] if isinstance(result, dict) else result.details
        if not result_details:
            content = result["content"] if isinstance(result, dict) else result.content
            first = content[0] if content else None
            return Text(getattr(first, "text", "") or "", 0, 0)

        if result_details.get("error"):
            return Text(theme.fg("error", f"Error: {result_details['error']}"), 0, 0)

        todo_list = result_details["todos"]
        action = result_details["action"]

        if action == "list":
            if not todo_list:
                return Text(theme.fg("dim", "No todos"), 0, 0)
            list_text = theme.fg("muted", f"{len(todo_list)} todo(s):")
            display = todo_list if options.get("expanded") else todo_list[:5]
            for t in display:
                check = theme.fg("success", "✓") if t["done"] else theme.fg("dim", "○")
                item_text = theme.fg("dim", t["text"]) if t["done"] else theme.fg("muted", t["text"])
                list_text += "\n" + check + " " + theme.fg("accent", f"#{t['id']}") + " " + item_text
            if not options.get("expanded") and len(todo_list) > 5:
                list_text += f"\n{theme.fg('dim', f'... {len(todo_list) - 5} more')}"
            return Text(list_text, 0, 0)

        if action == "add":
            added = todo_list[-1]
            return Text(
                theme.fg("success", "✓ Added ")
                + theme.fg("accent", f"#{added['id']}")
                + " "
                + theme.fg("muted", added["text"]),
                0,
                0,
            )

        if action == "toggle":
            content = result["content"] if isinstance(result, dict) else result.content
            first = content[0] if content else None
            msg = getattr(first, "text", "") or ""
            return Text(theme.fg("success", "✓ ") + theme.fg("muted", msg), 0, 0)

        # clear
        return Text(theme.fg("success", "✓ ") + theme.fg("muted", "Cleared all todos"), 0, 0)

    pi.register_tool(
        ToolDefinition(
            name="todo",
            label="Todo",
            description="Manage a todo list. Actions: list, add (text), toggle (id), clear",
            parameters=TODO_PARAMS,
            execute=execute,
            render_call=render_call,
            render_result=render_result,
        )
    )

    # The /todos command for users
    async def todos_command(_args: str, ctx) -> None:
        if ctx.mode != "tui":
            ctx.ui.notify("/todos requires interactive mode", "error")
            return

        await ctx.ui.custom(lambda _tui, theme, _kb, done: TodoListComponent(state["todos"], theme, lambda: done(None)))

    pi.register_command("todos", handler=todos_command, description="Show all todos on the current branch")
