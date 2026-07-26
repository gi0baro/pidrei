"""Mirror of pi coding-agent src/modes/interactive/components/compaction-summary-message.ts."""

from pidrei_tui import Box, Markdown, Spacer, Text

from ..theme import get_markdown_theme, theme
from .keybinding_hints import key_text


class CompactionSummaryMessageComponent(Box):
    """Renders a compaction message with collapsed/expanded state.

    Uses the same background color as custom messages for visual consistency.
    """

    def __init__(self, message, markdown_theme: dict | None = None) -> None:
        super().__init__(1, 1, lambda t: theme.bg("customMessageBg", t))
        self._expanded = False
        self._message = message
        self._markdown_theme = markdown_theme if markdown_theme is not None else get_markdown_theme()
        self._update_display()

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded
        self._update_display()

    def invalidate(self) -> None:
        super().invalidate()
        self._update_display()

    def _update_display(self) -> None:
        self.clear()

        # JS toLocaleString() thousands separators
        token_str = f"{self._message.tokens_before:,}"
        label = theme.fg("customMessageLabel", "\x1b[1m[compaction]\x1b[22m")
        self.add_child(Text(label, 0, 0))
        self.add_child(Spacer(1))

        if self._expanded:
            header = f"**Compacted from {token_str} tokens**\n\n"
            self.add_child(
                Markdown(
                    header + self._message.summary,
                    0,
                    0,
                    self._markdown_theme,
                    {"color": lambda text: theme.fg("customMessageText", text)},
                )
            )
        else:
            self.add_child(
                Text(
                    theme.fg("customMessageText", f"Compacted from {token_str} tokens (")
                    + theme.fg("dim", key_text("app.tools.expand"))
                    + theme.fg("customMessageText", " to expand)"),
                    0,
                    0,
                )
            )
