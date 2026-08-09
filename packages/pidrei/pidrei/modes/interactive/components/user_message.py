"""Mirror of pi coding-agent src/modes/interactive/components/user-message.ts."""

from pidrei_tui import Box, Container, Markdown

from ..theme import get_markdown_theme, theme
from .markdown_transform import create_markdown_transform


OSC133_ZONE_START = "\x1b]133;A\x07"
OSC133_ZONE_END = "\x1b]133;B\x07"
OSC133_ZONE_FINAL = "\x1b]133;C\x07"


class UserMessageComponent(Container):
    """Component that renders a user message."""

    def __init__(
        self,
        text: str,
        markdown_theme: dict | None = None,
        output_pad: int = 1,
        markdown_transformers=(),
    ) -> None:
        super().__init__()
        self._text = text
        self._markdown_theme = markdown_theme if markdown_theme is not None else get_markdown_theme()
        self._output_pad = output_pad
        self._markdown_transformers = markdown_transformers
        self._rebuild()

    def set_output_pad(self, padding: int) -> None:
        self._output_pad = padding
        self._rebuild()

    def _rebuild(self) -> None:
        self.clear()
        content_box = Box(self._output_pad, 1, lambda content: theme.bg("userMessageBg", content))
        content_box.add_child(
            Markdown(
                self._text,
                0,
                0,
                self._markdown_theme,
                {"color": lambda content: theme.fg("userMessageText", content)},
                {
                    "preserveOrderedListMarkers": True,
                    "preserveBackslashEscapes": True,
                    "transform": create_markdown_transform("user", False, self._markdown_transformers),
                },
            )
        )
        self.add_child(content_box)

    def render(self, width: int) -> list:
        lines = super().render(width)
        if not lines:
            return lines

        lines[0] = OSC133_ZONE_START + lines[0]
        lines[-1] = OSC133_ZONE_END + OSC133_ZONE_FINAL + lines[-1]
        return lines
