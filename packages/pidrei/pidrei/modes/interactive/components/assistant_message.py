"""Mirror of pi coding-agent src/modes/interactive/components/assistant-message.ts."""

from pidrei_tui import Container, Markdown, Spacer, Text

from ..theme import get_markdown_theme, theme
from .markdown_transform import create_markdown_transform


OSC133_ZONE_START = "\x1b]133;A\x07"
OSC133_ZONE_END = "\x1b]133;B\x07"
OSC133_ZONE_FINAL = "\x1b]133;C\x07"


def _has_visible_content(content) -> bool:
    return (content.type == "text" and content.text.strip()) or (
        content.type == "thinking" and content.thinking.strip()
    )


class AssistantMessageComponent(Container):
    """Component that renders a complete assistant message."""

    def __init__(
        self,
        message=None,
        hide_thinking_block: bool = False,
        markdown_theme: dict | None = None,
        hidden_thinking_label: str = "Thinking...",
        output_pad: int = 1,
        markdown_transformers=(),
    ) -> None:
        super().__init__()

        self._hide_thinking_block = hide_thinking_block
        self._markdown_theme = markdown_theme if markdown_theme is not None else get_markdown_theme()
        self._hidden_thinking_label = hidden_thinking_label
        self._output_pad = output_pad
        self._markdown_transformers = markdown_transformers
        self._last_message = None
        self._has_tool_calls = False
        self._is_streaming = False

        # Container for text/thinking content
        self._content_container = Container()
        self.add_child(self._content_container)

        if message is not None:
            self.update_content(message)

    def invalidate(self) -> None:
        super().invalidate()
        if self._last_message is not None:
            self.update_content(self._last_message)

    def set_hide_thinking_block(self, hide: bool) -> None:
        self._hide_thinking_block = hide
        if self._last_message is not None:
            self.update_content(self._last_message)

    def set_hidden_thinking_label(self, label: str) -> None:
        self._hidden_thinking_label = label
        if self._last_message is not None:
            self.update_content(self._last_message)

    def set_output_pad(self, padding: int) -> None:
        self._output_pad = padding
        if self._last_message is not None:
            self.update_content(self._last_message)

    def render(self, width: int) -> list:
        lines = super().render(width)
        if self._has_tool_calls or not lines:
            return lines

        lines[0] = OSC133_ZONE_START + lines[0]
        lines[-1] = OSC133_ZONE_END + OSC133_ZONE_FINAL + lines[-1]
        return lines

    def update_content(self, message, is_streaming: bool | None = None) -> None:
        # pi defaults the parameter to `this.isStreaming`; None means "keep".
        self._last_message = message
        if is_streaming is not None:
            self._is_streaming = is_streaming

        # Rebuilt per `message_update` while the render loop reads the
        # container on another thread: build locally, publish once at the end.
        children: list = []

        if any(_has_visible_content(c) for c in message.content):
            children.append(Spacer(1))

        # Render content in order
        i = 0
        while i < len(message.content):
            content = message.content[i]
            if content.type == "text" and content.text.strip():
                # Assistant text messages with no background - trim the text.
                # paddingY=0 avoids extra spacing before tool executions.
                children.append(
                    Markdown(
                        content.text.strip(),
                        self._output_pad,
                        0,
                        self._markdown_theme,
                        None,
                        {
                            "transform": create_markdown_transform(
                                "assistant", self._is_streaming, self._markdown_transformers
                            )
                        },
                    )
                )
            elif content.type == "thinking":
                thinking_blocks: list = []
                while i < len(message.content):
                    thinking_content = message.content[i]
                    if thinking_content.type != "thinking":
                        break
                    thinking = thinking_content.thinking.strip()
                    if thinking:
                        thinking_blocks.append(thinking)
                    i += 1
                i -= 1

                if not thinking_blocks:
                    i += 1
                    continue

                # Add spacing only when another visible assistant content
                # block follows. This avoids a superfluous blank line before
                # separately-rendered tool execution blocks.
                has_visible_content_after = any(_has_visible_content(c) for c in message.content[i + 1 :])

                if self._hide_thinking_block:
                    # Show one static label for each run of thinking blocks when hidden.
                    children.append(
                        Text(
                            theme.italic(theme.fg("thinkingText", self._hidden_thinking_label)),
                            self._output_pad,
                            0,
                        )
                    )
                else:
                    # Render each run of thinking blocks as one Markdown section.
                    children.append(
                        Markdown(
                            "\n\n".join(thinking_blocks),
                            self._output_pad,
                            0,
                            self._markdown_theme,
                            {"color": lambda text: theme.fg("thinkingText", text), "italic": True},
                            {
                                "transform": create_markdown_transform(
                                    "assistant-thinking", self._is_streaming, self._markdown_transformers
                                )
                            },
                        )
                    )
                if has_visible_content_after:
                    children.append(Spacer(1))
            i += 1

        # Check if incomplete/failed - show after partial content.
        # For aborted/error tool calls, tool execution components show the
        # error. Length stops can happen before a tool call is complete, so
        # surface them here too.
        has_tool_calls = any(c.type == "toolCall" for c in message.content)
        self._has_tool_calls = has_tool_calls
        if message.stop_reason == "length":
            children.append(Spacer(1))
            children.append(Text(theme.fg("error", "Response was truncated before completion."), self._output_pad, 0))
        elif not has_tool_calls:
            if message.stop_reason == "aborted":
                if message.error_message and message.error_message != "Request was aborted":
                    abort_message = message.error_message
                else:
                    abort_message = "Operation aborted"
                children.append(Spacer(1))
                children.append(Text(theme.fg("error", abort_message), self._output_pad, 0))
            elif message.stop_reason == "error":
                error_msg = message.error_message or "Unknown error"
                children.append(Spacer(1))
                children.append(Text(theme.fg("error", f"Error: {error_msg}"), self._output_pad, 0))

        self._content_container.set_children(children)
