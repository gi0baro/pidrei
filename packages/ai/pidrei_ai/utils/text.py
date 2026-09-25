"""Port of pi's text content helper (packages/ai/src/utils/text.ts)."""

from pidrei_ai.types import SystemMessage


def content_text(content: str | list, separator: str = "\n") -> str:
    """Extract and join text from message content."""
    if isinstance(content, str):
        return content
    return separator.join(block.text for block in content if block.type == "text")


def get_system_message_text(message: SystemMessage) -> str:
    """Render a system message as a complete prompt: its content followed by its sections."""
    parts = [content_text(message.content)]
    for text in (message.sections or {}).values():
        if text is not None:
            parts.append(text)
    return "\n\n".join(part for part in parts if len(part) > 0)


def render_system_message_update(message: SystemMessage) -> str:
    """Render a later system message for APIs that accept system messages mid-conversation.

    Section changes are framed by name so the model can relate them to the leading
    prompt. This framing is request-time only and may change between versions.
    """
    parts: list[str] = []
    text = content_text(message.content)
    if len(text) > 0:
        parts.append(text)
    for name, value in (message.sections or {}).items():
        parts.append(
            f'Removed system prompt section "{name}".'
            if value is None
            else f'Updated system prompt section "{name}":\n\n{value}'
        )
    return "\n\n".join(parts)
