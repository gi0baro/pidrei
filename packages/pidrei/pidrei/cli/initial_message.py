"""Mirror of pi coding-agent src/cli/initial-message.ts."""

from dataclasses import dataclass
from typing import Any

from .args import Args


@dataclass(slots=True)
class InitialMessageResult:
    initial_message: str | None = None
    initial_images: list[Any] | None = None


def build_initial_message(
    *,
    parsed: Args,
    file_text: str | None = None,
    file_images: list[Any] | None = None,
    stdin_content: str | None = None,
) -> InitialMessageResult:
    """Combine stdin content, @file text, and the first CLI message into a
    single initial prompt for non-interactive mode."""
    parts: list[str] = []
    if stdin_content is not None:
        parts.append(stdin_content)
    if file_text:
        parts.append(file_text)

    if len(parsed.messages) > 0:
        parts.append(parsed.messages[0])
        parsed.messages.pop(0)

    return InitialMessageResult(
        initial_message="".join(parts) if parts else None,
        initial_images=file_images if file_images else None,
    )
