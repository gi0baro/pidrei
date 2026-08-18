"""Questionnaire tool - single or multiple questions in one custom UI.

Single question: simple options list.
Multiple questions: tab bar navigation between questions plus a Submit tab.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/questionnaire.py
"""

from pidrei.core.extensions.types import ToolDefinition
from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent
from pidrei_tui import Editor, Key, Text, matches_key, visible_width, wrap_text_with_ansi


QUESTIONNAIRE_PARAMS = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "description": "Questions to ask the user",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Unique identifier for this question"},
                    "label": {
                        "type": "string",
                        "description": (
                            "Short contextual label for tab bar, e.g. 'Scope', 'Priority' (defaults to Q1, Q2)"
                        ),
                    },
                    "prompt": {"type": "string", "description": "The full question text to display"},
                    "options": {
                        "type": "array",
                        "description": "Available options to choose from",
                        "items": {
                            "type": "object",
                            "properties": {
                                "value": {"type": "string", "description": "The value returned when selected"},
                                "label": {"type": "string", "description": "Display label for the option"},
                                "description": {
                                    "type": "string",
                                    "description": "Optional description shown below label",
                                },
                            },
                            "required": ["value", "label"],
                        },
                    },
                    "allowOther": {
                        "type": "boolean",
                        "description": "Allow 'Type something' option (default: true)",
                    },
                },
                "required": ["id", "prompt", "options"],
            },
        },
    },
    "required": ["questions"],
}


class QuestionnaireComponent:
    """The component ctx.ui.custom mounts.

    Holds tab state, per-question answers, and an inline editor for the
    "Type something." option. `done` resolves the awaiting ctx.ui.custom call
    with a {"questions", "answers", "cancelled"} dict.
    """

    def __init__(self, tui, theme, questions: list[dict], done) -> None:
        self._tui = tui
        self._theme = theme
        self._questions = questions
        self._done = done
        self._is_multi = len(questions) > 1
        self._total_tabs = len(questions) + 1  # questions + Submit
        self._current_tab = 0
        self._option_index = 0
        self._input_mode = False
        self._input_question_id: str | None = None
        self._cached_lines: list[str] | None = None
        self._answers: dict[str, dict] = {}
        # Focusable marker: the TUI flips this and routes input to us.
        self.focused = False

        # Editor for the "Type something" option
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

    # -- helpers -------------------------------------------------------------

    def _refresh(self) -> None:
        self._cached_lines = None
        self._tui.request_render()

    def _submit(self, cancelled: bool) -> None:
        self._done({"questions": self._questions, "answers": list(self._answers.values()), "cancelled": cancelled})

    def _current_question(self) -> dict | None:
        if self._current_tab < len(self._questions):
            return self._questions[self._current_tab]
        return None

    def _current_options(self) -> list[dict]:
        question = self._current_question()
        if question is None:
            return []
        opts = [*question["options"]]
        if question["allowOther"]:
            opts.append({"value": "__other__", "label": "Type something.", "isOther": True})
        return opts

    def _all_answered(self) -> bool:
        return all(question["id"] in self._answers for question in self._questions)

    def _advance_after_answer(self) -> None:
        if not self._is_multi:
            self._submit(False)
            return
        if self._current_tab < len(self._questions) - 1:
            self._current_tab += 1
        else:
            self._current_tab = len(self._questions)  # Submit tab
        self._option_index = 0
        self._refresh()

    def _save_answer(self, question_id: str, value: str, label: str, was_custom: bool, index: int | None = None):
        self._answers[question_id] = {
            "id": question_id,
            "value": value,
            "label": label,
            "wasCustom": was_custom,
            "index": index,
        }

    def _leave_input_mode(self) -> None:
        self._input_mode = False
        self._input_question_id = None
        self._editor.focused = False
        self._editor.set_text("")

    def _on_editor_submit(self, value: str) -> None:
        if self._input_question_id is None:
            return
        trimmed = value.strip() or "(no response)"
        self._save_answer(self._input_question_id, trimmed, trimmed, True)
        self._leave_input_mode()
        self._advance_after_answer()

    # -- input ---------------------------------------------------------------

    async def handle_input(self, data: str) -> None:
        # Input mode: route to editor
        if self._input_mode:
            if matches_key(data, Key.escape):
                self._leave_input_mode()
                self._refresh()
                return
            await self._editor.handle_input(data)
            self._refresh()
            return

        question = self._current_question()
        opts = self._current_options()

        # Tab navigation (multi-question only)
        if self._is_multi:
            if matches_key(data, Key.tab) or matches_key(data, Key.right):
                self._current_tab = (self._current_tab + 1) % self._total_tabs
                self._option_index = 0
                self._refresh()
                return
            if matches_key(data, Key.shift("tab")) or matches_key(data, Key.left):
                self._current_tab = (self._current_tab - 1 + self._total_tabs) % self._total_tabs
                self._option_index = 0
                self._refresh()
                return

        # Submit tab
        if self._current_tab == len(self._questions):
            if matches_key(data, Key.enter) and self._all_answered():
                self._submit(False)
            elif matches_key(data, Key.escape):
                self._submit(True)
            return

        # Option navigation
        if matches_key(data, Key.up):
            self._option_index = max(0, self._option_index - 1)
            self._refresh()
            return
        if matches_key(data, Key.down):
            self._option_index = min(len(opts) - 1, self._option_index + 1)
            self._refresh()
            return

        # Select option
        if matches_key(data, Key.enter) and question is not None:
            opt = opts[self._option_index]
            if opt.get("isOther"):
                self._input_mode = True
                self._input_question_id = question["id"]
                self._editor.focused = True
                self._editor.set_text("")
                self._refresh()
                return
            self._save_answer(question["id"], opt["value"], opt["label"], False, self._option_index + 1)
            self._advance_after_answer()
            return

        # Cancel
        if matches_key(data, Key.escape):
            self._submit(True)

    # -- rendering -----------------------------------------------------------

    def invalidate(self) -> None:
        self._cached_lines = None

    def render(self, width: int) -> list[str]:
        if self._cached_lines is not None:
            return self._cached_lines

        theme = self._theme
        lines: list[str] = []
        render_width = max(1, width)
        question = self._current_question()
        opts = self._current_options()

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

        # Tab bar (multi-question only)
        if self._is_multi:
            tabs = ["← "]
            for index, tab_question in enumerate(self._questions):
                is_active = index == self._current_tab
                is_answered = tab_question["id"] in self._answers
                box = "■" if is_answered else "□"
                color = "success" if is_answered else "muted"
                text = f" {box} {tab_question['label']} "
                styled = theme.bg("selectedBg", theme.fg("text", text)) if is_active else theme.fg(color, text)
                tabs.append(f"{styled} ")
            can_submit = self._all_answered()
            is_submit_tab = self._current_tab == len(self._questions)
            submit_text = " ✓ Submit "
            submit_styled = (
                theme.bg("selectedBg", theme.fg("text", submit_text))
                if is_submit_tab
                else theme.fg("success" if can_submit else "dim", submit_text)
            )
            tabs.append(f"{submit_styled} →")
            add_wrapped_with_prefix(" ", "".join(tabs))
            lines.append("")

        # Helper to render options list
        def render_options() -> None:
            for index, opt in enumerate(opts):
                selected = index == self._option_index
                is_other = bool(opt.get("isOther"))
                prefix = theme.fg("accent", "> ") if selected else "  "
                label = f"{index + 1}. {opt['label']}{' ✎' if is_other and self._input_mode else ''}"
                color = "accent" if selected or (is_other and self._input_mode) else "text"

                add_wrapped_with_prefix(prefix, theme.fg(color, label))
                if opt.get("description"):
                    add_wrapped_with_prefix("     ", theme.fg("muted", opt["description"]))

        # Content
        if self._input_mode and question is not None:
            add_wrapped_with_prefix(" ", theme.fg("text", question["prompt"]))
            lines.append("")
            # Show options for reference
            render_options()
            lines.append("")
            add_wrapped_with_prefix(" ", theme.fg("muted", "Your answer:"))
            for line in self._editor.render(max(1, render_width - 2)):
                lines.append(f" {line}")
            lines.append("")
            add_wrapped_with_prefix(" ", theme.fg("dim", "Enter to submit • Esc to cancel"))
        elif self._current_tab == len(self._questions):
            add_wrapped_with_prefix(" ", theme.fg("accent", theme.bold("Ready to submit")))
            lines.append("")
            for summary_question in self._questions:
                answer = self._answers.get(summary_question["id"])
                if answer is not None:
                    prefix = "(wrote) " if answer["wasCustom"] else ""
                    summary = theme.fg("muted", f"{summary_question['label']}: ") + theme.fg(
                        "text", prefix + answer["label"]
                    )
                    add_wrapped_with_prefix(" ", summary)
            lines.append("")
            if self._all_answered():
                add_wrapped_with_prefix(" ", theme.fg("success", "Press Enter to submit"))
            else:
                missing = ", ".join(q["label"] for q in self._questions if q["id"] not in self._answers)
                add_wrapped_with_prefix(" ", theme.fg("warning", f"Unanswered: {missing}"))
        elif question is not None:
            add_wrapped_with_prefix(" ", theme.fg("text", question["prompt"]))
            lines.append("")
            render_options()

        lines.append("")
        if not self._input_mode:
            help_text = (
                "Tab/←→ navigate • ↑↓ select • Enter confirm • Esc cancel"
                if self._is_multi
                else "↑↓ navigate • Enter select • Esc cancel"
            )
            add_wrapped_with_prefix(" ", theme.fg("dim", help_text))
        lines.append(theme.fg("accent", "─" * render_width))

        self._cached_lines = lines
        return lines


def _error_result(message: str, questions: list[dict] | None = None) -> AgentToolResult:
    return AgentToolResult(
        content=[TextContent(text=message)],
        details={"questions": questions or [], "answers": [], "cancelled": True},
    )


def extension(pi):
    async def execute(_tool_call_id, params, _cancel=None, _on_update=None, ctx=None):
        if ctx is None or ctx.mode != "tui":
            return _error_result("Error: UI not available (running in non-interactive mode)")
        if not params["questions"]:
            return _error_result("Error: No questions provided")

        # Normalize questions with defaults
        questions = [
            {
                **q,
                "label": q.get("label") or f"Q{index + 1}",
                "allowOther": q.get("allowOther") is not False,
            }
            for index, q in enumerate(params["questions"])
        ]

        result = await ctx.ui.custom(lambda tui, theme, _kb, done: QuestionnaireComponent(tui, theme, questions, done))

        if result is None or result["cancelled"]:
            return AgentToolResult(
                content=[TextContent(text="User cancelled the questionnaire")],
                details=result or {"questions": questions, "answers": [], "cancelled": True},
            )

        answer_lines = []
        for answer in result["answers"]:
            q_label = next((q["label"] for q in questions if q["id"] == answer["id"]), answer["id"])
            if answer["wasCustom"]:
                answer_lines.append(f"{q_label}: user wrote: {answer['label']}")
            else:
                answer_lines.append(f"{q_label}: user selected: {answer['index']}. {answer['label']}")

        return AgentToolResult(content=[TextContent(text="\n".join(answer_lines))], details=result)

    def render_call(args, theme, _context):
        questions = args.get("questions") if isinstance(args.get("questions"), list) else []
        count = len(questions)
        labels = ", ".join(q.get("label") or q.get("id", "") for q in questions)
        text = theme.fg("toolTitle", theme.bold("questionnaire "))
        text += theme.fg("muted", f"{count} question{'s' if count != 1 else ''}")
        if labels:
            text += theme.fg("dim", f" ({labels})")
        return Text(text, 0, 0)

    def render_result(result, _options, theme, _context):
        details = result.get("details") if isinstance(result, dict) else getattr(result, "details", None)
        if not details:
            content = result.get("content") if isinstance(result, dict) else getattr(result, "content", None)
            first = content[0] if content else None
            return Text(getattr(first, "text", "") if first is not None else "", 0, 0)
        if details["cancelled"]:
            return Text(theme.fg("warning", "Cancelled"), 0, 0)
        lines = []
        for answer in details["answers"]:
            if answer["wasCustom"]:
                lines.append(
                    theme.fg("success", "✓ ")
                    + theme.fg("accent", answer["id"])
                    + ": "
                    + theme.fg("muted", "(wrote) ")
                    + answer["label"]
                )
            else:
                display = f"{answer['index']}. {answer['label']}" if answer.get("index") else answer["label"]
                lines.append(theme.fg("success", "✓ ") + theme.fg("accent", answer["id"]) + f": {display}")
        return Text("\n".join(lines), 0, 0)

    pi.register_tool(
        ToolDefinition(
            name="questionnaire",
            label="Questionnaire",
            description=(
                "Ask the user one or more questions. Use for clarifying requirements, getting preferences, "
                "or confirming decisions. For single questions, shows a simple option list. For multiple "
                "questions, shows a tab-based interface."
            ),
            parameters=QUESTIONNAIRE_PARAMS,
            execute=execute,
            render_call=render_call,
            render_result=render_result,
        )
    )
