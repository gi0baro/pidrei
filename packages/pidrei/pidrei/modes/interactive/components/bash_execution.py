"""Mirror of pi coding-agent src/modes/interactive/components/bash-execution.ts.

Component for displaying bash command execution with streaming output.
"""

from pidrei_tui import Container, Loader, Spacer, Text

from ....core.tools.truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, truncate_tail
from ....utils.ansi import strip_ansi
from ..theme import theme
from .dynamic_border import DynamicBorder
from .keybinding_hints import key_hint, key_text
from .visual_truncate import truncate_to_visual_lines


# Preview line limit when not expanded (matches tool execution behavior)
PREVIEW_LINES = 20


class _CachedVisualTruncation:
    """Width-aware render cache for the collapsed preview (pi's inline object)."""

    def __init__(self, styled_input: str) -> None:
        self._styled_input = styled_input
        self._cached_width: int | None = None
        self._cached_lines: list | None = None

    def render(self, width: int) -> list:
        if self._cached_lines is None or self._cached_width != width:
            result = truncate_to_visual_lines(self._styled_input, PREVIEW_LINES, width, 1)
            self._cached_lines = result["visualLines"]
            self._cached_width = width
        return self._cached_lines if self._cached_lines is not None else []

    def invalidate(self) -> None:
        self._cached_width = None
        self._cached_lines = None


class BashExecutionComponent(Container):
    def __init__(self, command: str, ui, exclude_from_context: bool = False) -> None:
        super().__init__()
        self._command = command
        self._output_lines: list = []
        self._status = "running"  # "running" | "complete" | "cancelled" | "error"
        self._exit_code: int | None = None
        self._truncation_result = None
        self._full_output_path: str | None = None
        self._expanded = False

        # Use dim border for excluded-from-context commands (!! prefix)
        color_key = "dim" if exclude_from_context else "bashMode"

        def border_color(text: str) -> str:
            return theme.fg(color_key, text)

        # Add spacer
        self.add_child(Spacer(1))

        # Top border
        self.add_child(DynamicBorder(border_color))

        # Content container (holds dynamic content between borders)
        self._content_container = Container()
        self.add_child(self._content_container)

        # Command header
        header = Text(theme.fg(color_key, theme.bold(f"$ {command}")), 1, 0)
        self._content_container.add_child(header)

        # Loader
        self._loader = Loader(
            ui,
            lambda spinner: theme.fg(color_key, spinner),
            lambda text: theme.fg("muted", text),
            f"Running... ({key_text('tui.select.cancel')} to cancel)",  # Plain text for loader
        )
        self._content_container.add_child(self._loader)

        # Bottom border
        self.add_child(DynamicBorder(border_color))

    def set_expanded(self, expanded: bool) -> None:
        """Expanded shows full output; collapsed shows the preview only."""
        self._expanded = expanded
        self._update_display()

    def invalidate(self) -> None:
        super().invalidate()
        self._update_display()

    def append_output(self, chunk: str) -> None:
        # Strip ANSI codes and normalize line endings. Binary data is already
        # sanitized before reaching this component.
        clean = strip_ansi(chunk).replace("\r\n", "\n").replace("\r", "\n")

        # Append to output lines
        new_lines = clean.split("\n")
        if self._output_lines and new_lines:
            # Append first chunk to last line (incomplete line continuation)
            self._output_lines[-1] += new_lines[0]
            self._output_lines.extend(new_lines[1:])
        else:
            self._output_lines.extend(new_lines)

        self._update_display()

    def set_complete(
        self,
        exit_code: int | None,
        cancelled: bool,
        truncation_result=None,
        full_output_path: str | None = None,
    ) -> None:
        self._exit_code = exit_code
        if cancelled:
            self._status = "cancelled"
        elif exit_code is not None and exit_code != 0:
            self._status = "error"
        else:
            self._status = "complete"
        self._truncation_result = truncation_result
        self._full_output_path = full_output_path

        # Stop loader
        self._loader.stop()

        self._update_display()

    def _update_display(self) -> None:
        # Apply truncation for LLM context limits (same limits as bash tool)
        full_output = "\n".join(self._output_lines)
        context_truncation = truncate_tail(full_output, DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES)

        # Get the lines to potentially display (after context truncation)
        available_lines = context_truncation.content.split("\n") if context_truncation.content else []

        # Apply preview truncation based on expanded state
        preview_logical_lines = available_lines[-PREVIEW_LINES:]
        hidden_line_count = len(available_lines) - len(preview_logical_lines)

        # Rebuild content container
        self._content_container.clear()

        # Command header
        header = Text(theme.fg("bashMode", theme.bold(f"$ {self._command}")), 1, 0)
        self._content_container.add_child(header)

        # Output
        if available_lines:
            if self._expanded:
                # Show all lines
                display_text = "\n".join(theme.fg("muted", line) for line in available_lines)
                self._content_container.add_child(Text(f"\n{display_text}", 1, 0))
            else:
                # Use shared visual truncation utility with width-aware caching
                styled_output = "\n".join(theme.fg("muted", line) for line in preview_logical_lines)
                self._content_container.add_child(_CachedVisualTruncation(f"\n{styled_output}"))

        # Loader or status
        if self._status == "running":
            self._content_container.add_child(self._loader)
        else:
            status_parts: list = []

            # Show how many lines are hidden (collapsed preview)
            if hidden_line_count > 0:
                if self._expanded:
                    status_parts.append(
                        theme.fg("muted", "(") + key_hint("app.tools.expand", "to collapse") + theme.fg("muted", ")")
                    )
                else:
                    status_parts.append(
                        theme.fg("muted", f"... {hidden_line_count} more lines (")
                        + key_hint("app.tools.expand", "to expand")
                        + theme.fg("muted", ")")
                    )

            if self._status == "cancelled":
                status_parts.append(theme.fg("warning", "(cancelled)"))
            elif self._status == "error":
                status_parts.append(theme.fg("error", f"(exit {self._exit_code})"))

            # Add truncation warning (context truncation, not preview truncation)
            was_truncated = (
                self._truncation_result is not None and self._truncation_result.truncated
            ) or context_truncation.truncated
            if was_truncated and self._full_output_path:
                status_parts.append(theme.fg("warning", f"Output truncated. Full output: {self._full_output_path}"))

            if status_parts:
                self._content_container.add_child(Text("\n" + "\n".join(status_parts), 1, 0))

    def get_output(self) -> str:
        """Get the raw output for creating BashExecutionMessage."""
        return "\n".join(self._output_lines)

    def get_command(self) -> str:
        """Get the command that was executed."""
        return self._command
