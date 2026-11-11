"""Mirror of pi coding-agent src/modes/interactive/components/skill-invocation-message.ts."""

from pidrei_tui import Box, Container, Markdown, MouseRegion, Text, TuiMouseEvent, TuiMouseEventResult

from ..theme import get_markdown_theme, theme
from .keybinding_hints import key_text
from .markdown_transform import create_markdown_transform


class SkillInvocationMessageComponent(Box):
    """Renders a skill invocation message with collapsed/expanded state.

    Uses the same background color as custom messages for visual consistency.
    Only renders the skill block itself - the user message is rendered
    separately.
    """

    def __init__(
        self,
        skill_block,
        markdown_theme: dict | None = None,
        output_pad: int = 1,
        markdown_transformers=(),
        message_id: str | None = None,
        timestamp: str | None = None,
    ) -> None:
        super().__init__(output_pad, 1, lambda t: theme.bg("customMessageBg", t))
        self._expanded = False
        self._skill_block = skill_block
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

        if self._expanded:
            # Expanded: label + skill name header + full content
            label = theme.fg("customMessageLabel", "\x1b[1m[skill]\x1b[22m")
            content.add_child(Text(label, 0, 0))
            header = f"**{self._skill_block.name}**\n\n"
            transform = create_markdown_transform(
                "custom", lambda: False, self._markdown_transformers, self._message_id, self._timestamp
            )
            content.add_child(
                Markdown(
                    self._skill_block.content,
                    0,
                    0,
                    self._markdown_theme,
                    {"color": lambda text: theme.fg("customMessageText", text)},
                    # Transformers see only the skill content, not PiDrei's header.
                    {"transform": lambda markdown, width: header + transform(markdown, width)},
                )
            )
        else:
            # Collapsed: single line - [skill] name (hint to expand)
            line = (
                theme.fg("customMessageLabel", "\x1b[1m[skill]\x1b[22m ")
                + theme.fg("customMessageText", self._skill_block.name)
                + theme.fg("dim", f" ({key_text('app.tools.expand')} to expand)")
            )
            content.add_child(Text(line, 0, 0))

        def on_mouse(event: TuiMouseEvent) -> TuiMouseEventResult | None:
            if event.type != "click" or event.button != "left":
                return None
            self.set_expanded(not self._expanded)
            return TuiMouseEventResult(handled=True)

        self.add_child(MouseRegion(content, on_mouse))
