"""Port of pi's text content helper (packages/ai/src/utils/text.ts)."""


def content_text(content: str | list, separator: str = "\n") -> str:
    """Extract and join text from message content."""
    if isinstance(content, str):
        return content
    return separator.join(block.text for block in content if block.type == "text")
