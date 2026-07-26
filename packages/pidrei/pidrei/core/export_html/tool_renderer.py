"""Mirror of pi coding-agent src/core/export-html/tool-renderer.ts.

Renders custom tool calls and results to HTML by invoking their TUI
renderers and converting the ANSI output to HTML.
"""

import re

from .ansi_to_html import ansi_lines_to_html


_ANSI_ESCAPE_REGEX = re.compile(r"\x1b\[[\d;]*m")


def _is_blank_rendered_line(line: str) -> bool:
    return len(_ANSI_ESCAPE_REGEX.sub("", line).strip()) == 0


def _trim_rendered_result_lines(lines: list) -> list:
    start = 0
    end = len(lines)
    while start < end and _is_blank_rendered_line(lines[start]):
        start += 1
    while end > start and _is_blank_rendered_line(lines[end - 1]):
        end -= 1
    return lines[start:end]


class ToolHtmlRenderer:
    """Renders custom tools via their render_call/render_result hooks.

    ``deps`` is ``{"getToolDefinition", "theme", "cwd", "width"?}``.
    """

    def __init__(self, deps: dict) -> None:
        self._get_tool_definition = deps["getToolDefinition"]
        self._theme = deps["theme"]
        self._cwd = deps["cwd"]
        self._width = deps.get("width", 100)
        self._rendered_call_components: dict = {}
        self._rendered_result_components: dict = {}
        self._rendered_states: dict = {}
        self._rendered_args: dict = {}

    def _get_state(self, tool_call_id: str) -> dict:
        state = self._rendered_states.get(tool_call_id)
        if state is None:
            state = {}
            self._rendered_states[tool_call_id] = state
        return state

    def _create_render_context(
        self, tool_call_id: str, last_component, expanded: bool, is_partial: bool, is_error: bool
    ) -> dict:
        return {
            "args": self._rendered_args.get(tool_call_id),
            "toolCallId": tool_call_id,
            "invalidate": lambda: None,
            "lastComponent": last_component,
            "state": self._get_state(tool_call_id),
            "cwd": self._cwd,
            "executionStarted": True,
            "argsComplete": True,
            "isPartial": is_partial,
            "expanded": expanded,
            "showImages": False,
            "isError": is_error,
        }

    def render_call(self, tool_call_id: str, tool_name: str, args) -> str | None:
        """Render a tool call to HTML; None if the tool has no renderer."""
        try:
            self._rendered_args[tool_call_id] = args
            tool_def = self._get_tool_definition(tool_name)
            if tool_def is None or getattr(tool_def, "render_call", None) is None:
                return None

            component = tool_def.render_call(
                args,
                self._theme,
                self._create_render_context(
                    tool_call_id, self._rendered_call_components.get(tool_call_id), False, True, False
                ),
            )
            self._rendered_call_components[tool_call_id] = component
            lines = component.render(self._width)
            return ansi_lines_to_html(lines)
        except Exception:
            # On error, return None so HTML export can fall back to
            # structured result rendering
            return None

    def render_result(self, tool_call_id: str, tool_name: str, result: list, details, is_error: bool) -> dict | None:
        """Render a tool result to ``{"collapsed"?, "expanded"}`` HTML."""
        try:
            tool_def = self._get_tool_definition(tool_name)
            if tool_def is None or getattr(tool_def, "render_result", None) is None:
                return None

            # Build the tool result record from the content array (session
            # storage uses generic records)
            agent_tool_result = {"content": result, "details": details, "isError": is_error}

            # Render collapsed
            collapsed_component = tool_def.render_result(
                agent_tool_result,
                {"expanded": False, "isPartial": False},
                self._theme,
                self._create_render_context(
                    tool_call_id, self._rendered_result_components.get(tool_call_id), False, False, is_error
                ),
            )
            self._rendered_result_components[tool_call_id] = collapsed_component
            collapsed = ansi_lines_to_html(_trim_rendered_result_lines(collapsed_component.render(self._width)))

            # Render expanded
            expanded_component = tool_def.render_result(
                agent_tool_result,
                {"expanded": True, "isPartial": False},
                self._theme,
                self._create_render_context(
                    tool_call_id, self._rendered_result_components.get(tool_call_id), True, False, is_error
                ),
            )
            self._rendered_result_components[tool_call_id] = expanded_component
            expanded = ansi_lines_to_html(_trim_rendered_result_lines(expanded_component.render(self._width)))

            rendered: dict = {}
            if collapsed and collapsed != expanded:
                rendered["collapsed"] = collapsed
            rendered["expanded"] = expanded
            return rendered
        except Exception:
            # On error, return None so HTML export can fall back to
            # structured result rendering
            return None


def create_tool_html_renderer(deps: dict) -> ToolHtmlRenderer:
    return ToolHtmlRenderer(deps)
