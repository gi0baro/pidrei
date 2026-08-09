"""Width-aware Markdown transformer chain (port of ``markdown-transform.ts``).

Transformers are registered by extensions through
``pi.register_markdown_transformer`` and run in extension load order, each
receiving the Markdown the previous one returned. A transformer that raises is
skipped and the chain continues with the Markdown produced so far.

Transformers are SYNC — the one deliberate exception to the async-only callback
rule. They run inside ``Markdown.render()``, once per render (every streaming
update and every width change), where there is nothing to await into.
"""

from typing import Any


__all__ = ["create_markdown_transform"]


def create_markdown_transform(message_type: str, is_streaming: bool, transformers) -> Any:
    """Build the ``(markdown, available_width) -> str`` callable a Markdown takes."""

    def transform(markdown: str, available_width: int) -> str:
        context = {
            "messageType": message_type,
            "isStreaming": is_streaming,
            "availableWidth": available_width,
        }
        return _apply_markdown_transformers(markdown, context, transformers)

    return transform


def _apply_markdown_transformers(markdown: str, context: dict, transformers) -> str:
    transformed_markdown = markdown
    for transformer in transformers:
        try:
            transformed = transformer(transformed_markdown, context)
        except Exception:  # noqa: S112 - keep the current Markdown, try the next transformer
            continue
        if isinstance(transformed, str):
            transformed_markdown = transformed
    return transformed_markdown
