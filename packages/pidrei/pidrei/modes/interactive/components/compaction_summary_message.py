"""Mirror of pi coding-agent src/modes/interactive/components/compaction-summary-message.ts."""

from pidrei_tui import Box, Container, Markdown, MouseRegion, Spacer, Text, TuiMouseEvent, TuiMouseEventResult

from ..theme import get_markdown_theme, theme
from .keybinding_hints import key_text
from .markdown_transform import create_markdown_transform


class CompactionSummaryMessageComponent(Box):
    """Renders a compaction message with collapsed/expanded state.

    Uses the same background color as custom messages for visual consistency.
    """

    def __init__(
        self,
        message,
        markdown_theme: dict | None = None,
        output_pad: int = 1,
        markdown_transformers=(),
        message_id: str | None = None,
        timestamp: str | None = None,
    ) -> None:
        super().__init__(output_pad, 1, lambda t: theme.bg("customMessageBg", t))
        self._expanded = False
        self._message = message
        self._markdown_theme = markdown_theme if markdown_theme is not None else get_markdown_theme()
        self._markdown_transformers = markdown_transformers
        self._message_id = message_id
        self._timestamp = timestamp
        self._update_display()

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded
        self._update_display()

    def set_output_pad(self, output_pad: int) -> None:
        self.set_padding_x(output_pad)

    def invalidate(self) -> None:
        super().invalidate()
        self._update_display()

    def _update_display(self) -> None:
        self.clear()
        content = Container()

        # JS toLocaleString() thousands separators
        token_str = f"{self._message.tokens_before:,}"
        label = theme.fg("customMessageLabel", "\x1b[1m[compaction]\x1b[22m")
        content.add_child(Text(label, 0, 0))
        content.add_child(Spacer(1))

        if self._expanded:
            header = f"**Compacted from {token_str} tokens**\n\n"
            transform = create_markdown_transform(
                "summary", lambda: False, self._markdown_transformers, self._message_id, self._timestamp
            )
            content.add_child(
                Markdown(
                    self._message.summary,
                    0,
                    0,
                    self._markdown_theme,
                    {"color": lambda text: theme.fg("customMessageText", text)},
                    # Transformers see only the summary content, not PiDrei's header.
                    {"transform": lambda markdown, width: header + transform(markdown, width)},
                )
            )
        else:
            content.add_child(
                Text(
                    theme.fg("customMessageText", f"Compacted from {token_str} tokens (")
                    + theme.fg("dim", key_text("app.tools.expand"))
                    + theme.fg("customMessageText", " to expand)"),
                    0,
                    0,
                )
            )

        def on_mouse(event: TuiMouseEvent) -> TuiMouseEventResult | None:
            if event.type != "click" or event.button != "left":
                return None
            self.set_expanded(not self._expanded)
            return TuiMouseEventResult(handled=True)

        self.add_child(MouseRegion(content, on_mouse))
