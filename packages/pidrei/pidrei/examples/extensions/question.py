"""Question tool - single question with options.

Full custom UI via ctx.ui.custom: an option list plus an inline editor behind
"Type something.". Escape in the editor returns to the options; Escape in the
options cancels.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/question.py
"""

from pidrei.core.extensions.types import ToolDefinition
from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent
from pidrei_tui import Editor, Key, Text, matches_key, visible_width, wrap_text_with_ansi


QUESTION_PARAMS = {
    "type": "object",
    "properties": {
        "question": {"type": "string", "description": "The question to ask the user"},
        "options": {
            "type": "array",
            "description": "Options for the user to choose from",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "Display label for the option"},
                    "description": {"type": "string", "description": "Optional description shown below label"},
                },
                "required": ["label"],
            },
        },
    },
    "required": ["question", "options"],
}


class QuestionComponent:
    """The component ctx.ui.custom mounts: render + invalidate + handle_input.

    `done` resolves the awaiting ctx.ui.custom call with whatever it is given;
    passing None means the user cancelled.
    """

    def __init__(self, tui, theme, question: str, options: list[dict], done) -> None:
        self._tui = tui
        self._theme = theme
        self._question = question
        self._options = options
        self._done = done
        self._option_index = 0
        self._edit_mode = False
        self._cached_lines: list[str] | None = None
        # Focusable marker: the TUI flips this and routes input to us.
        self.focused = False

        editor_theme = {
            "borderColor": lambda s: theme.fg("accent", s),
            "selectList": {
                "selectedPrefix": lambda t: theme.fg("accent", t),
                "selectedText": lambda t: theme.fg("accent", t),
                "description": lambda t: theme.fg("muted", t),
                "scrollInfo": lambda t: theme.fg("dim", t),
                "noMatch": lambda t: theme.fg("warning", t),
            },
        }
        self._editor = Editor(tui, editor_theme)
        self._editor.on_submit = self._on_editor_submit

    # -- state ---------------------------------------------------------------

    def _refresh(self) -> None:
        self._cached_lines = None
        self._tui.request_render()

    def _leave_edit_mode(self) -> None:
        self._edit_mode = False
        self._editor.focused = False
        self._editor.set_text("")
        self._refresh()

    def _on_editor_submit(self, value: str) -> None:
        trimmed = value.strip()
        if trimmed:
            self._done({"answer": trimmed, "wasCustom": True})
        else:
            self._leave_edit_mode()

    # -- input ---------------------------------------------------------------

    async def handle_input(self, data: str) -> None:
        if self._edit_mode:
            if matches_key(data, Key.escape):
                self._leave_edit_mode()
                return
            await self._editor.handle_input(data)
            self._refresh()
            return

        if matches_key(data, Key.up):
            self._option_index = max(0, self._option_index - 1)
            self._refresh()
            return
        if matches_key(data, Key.down):
            self._option_index = min(len(self._options) - 1, self._option_index + 1)
            self._refresh()
            return

        if matches_key(data, Key.enter):
            selected = self._options[self._option_index]
            if selected.get("isOther"):
                self._edit_mode = True
                self._editor.focused = True
                self._refresh()
            else:
                self._done({"answer": selected["label"], "wasCustom": False, "index": self._option_index + 1})
            return

        if matches_key(data, Key.escape):
            self._done(None)

    # -- rendering -----------------------------------------------------------

    def invalidate(self) -> None:
        self._cached_lines = None

    def render(self, width: int) -> list[str]:
        if self._cached_lines is not None:
            return self._cached_lines

        theme = self._theme
        lines: list[str] = []
        render_width = max(1, width)

        def add_wrapped(text: str) -> None:
            lines.extend(wrap_text_with_ansi(text, render_width))

        def add_wrapped_with_prefix(prefix: str, text: str) -> None:
            prefix_width = visible_width(prefix)
            if prefix_width >= render_width:
                add_wrapped(prefix + text)
                return
            wrapped = wrap_text_with_ansi(text, render_width - prefix_width)
            continuation_prefix = " " * prefix_width
            for index, line in enumerate(wrapped):
                lines.append(f"{prefix if index == 0 else continuation_prefix}{line}")

        lines.append(theme.fg("accent", "─" * render_width))
        add_wrapped_with_prefix(" ", theme.fg("text", self._question))
        lines.append("")

        for index, opt in enumerate(self._options):
            selected = index == self._option_index
            is_other = bool(opt.get("isOther"))
            prefix = theme.fg("accent", "> ") if selected else "  "
            label = f"{index + 1}. {opt['label']}{' ✎' if is_other and self._edit_mode else ''}"
            color = "accent" if selected or (is_other and self._edit_mode) else "text"

            add_wrapped_with_prefix(prefix, theme.fg(color, label))

            # Show description if present
            if opt.get("description"):
                add_wrapped_with_prefix("     ", theme.fg("muted", opt["description"]))

        if self._edit_mode:
            lines.append("")
            add_wrapped_with_prefix(" ", theme.fg("muted", "Your answer:"))
            for line in self._editor.render(max(1, render_width - 2)):
                lines.append(f" {line}")

        lines.append("")
        if self._edit_mode:
            add_wrapped_with_prefix(" ", theme.fg("dim", "Enter to submit • Esc to go back"))
        else:
            add_wrapped_with_prefix(" ", theme.fg("dim", "↑↓ navigate • Enter to select • Esc to cancel"))
        lines.append(theme.fg("accent", "─" * render_width))

        self._cached_lines = lines
        return lines


def extension(pi):
    async def execute(_tool_call_id, params, _cancel=None, _on_update=None, ctx=None):
        question = params["question"]
        simple_options = [opt["label"] for opt in params["options"]]

        if ctx is None or ctx.mode != "tui":
            return AgentToolResult(
                content=[TextContent(text="Error: UI not available (running in non-interactive mode)")],
                details={"question": question, "options": simple_options, "answer": None},
            )

        if not params["options"]:
            return AgentToolResult(
                content=[TextContent(text="Error: No options provided")],
                details={"question": question, "options": [], "answer": None},
            )

        all_options = [*params["options"], {"label": "Type something.", "isOther": True}]

        result = await ctx.ui.custom(
            lambda tui, theme, _kb, done: QuestionComponent(tui, theme, question, all_options, done)
        )

        if result is None:
            return AgentToolResult(
                content=[TextContent(text="User cancelled the selection")],
                details={"question": question, "options": simple_options, "answer": None},
            )

        if result["wasCustom"]:
            return AgentToolResult(
                content=[TextContent(text=f"User wrote: {result['answer']}")],
                details={
                    "question": question,
                    "options": simple_options,
                    "answer": result["answer"],
                    "wasCustom": True,
                },
            )
        return AgentToolResult(
            content=[TextContent(text=f"User selected: {result['index']}. {result['answer']}")],
            details={
                "question": question,
                "options": simple_options,
                "answer": result["answer"],
                "wasCustom": False,
            },
        )

    def render_call(args, theme, _context):
        text = theme.fg("toolTitle", theme.bold("question ")) + theme.fg("muted", args.get("question") or "")
        opts = args.get("options") if isinstance(args.get("options"), list) else []
        if opts:
            labels = [opt.get("label", "") for opt in opts]
            numbered = [f"{index + 1}. {label}" for index, label in enumerate([*labels, "Type something."])]
            text += "\n" + theme.fg("dim", f"  Options: {', '.join(numbered)}")
        return Text(text, 0, 0)

    def render_result(result, _options, theme, _context):
        details = result.get("details") if isinstance(result, dict) else getattr(result, "details", None)
        if not details:
            content = result.get("content") if isinstance(result, dict) else getattr(result, "content", None)
            first = content[0] if content else None
            return Text(getattr(first, "text", "") if first is not None else "", 0, 0)

        if details["answer"] is None:
            return Text(theme.fg("warning", "Cancelled"), 0, 0)

        if details.get("wasCustom"):
            return Text(
                theme.fg("success", "✓ ") + theme.fg("muted", "(wrote) ") + theme.fg("accent", details["answer"]),
                0,
                0,
            )
        index = details["options"].index(details["answer"]) + 1 if details["answer"] in details["options"] else 0
        display = f"{index}. {details['answer']}" if index > 0 else details["answer"]
        return Text(theme.fg("success", "✓ ") + theme.fg("accent", display), 0, 0)

    pi.register_tool(
        ToolDefinition(
            name="question",
            label="Question",
            description=(
                "Ask the user a question and let them pick from options. Use when you need user input to proceed."
            ),
            parameters=QUESTION_PARAMS,
            execution_mode="sequential",
            execute=execute,
            render_call=render_call,
            render_result=render_result,
        )
    )
