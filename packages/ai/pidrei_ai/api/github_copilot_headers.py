"""Port of pi's Copilot request headers (packages/ai/src/api/github-copilot-headers.ts)."""

from typing import Literal

from pidrei_ai.types import Message


def infer_copilot_initiator(messages: list[Message]) -> Literal["user", "agent"]:
    """Copilot expects X-Initiator to indicate whether the request is user-initiated
    or agent-initiated (e.g. follow-up after assistant/tool messages)."""
    last = messages[-1] if messages else None
    return "agent" if last is not None and last.role != "user" else "user"


def has_copilot_vision_input(messages: list[Message]) -> bool:
    """Copilot requires Copilot-Vision-Request header when sending images."""
    for message in messages:
        if (
            message.role in ("user", "toolResult")
            and isinstance(message.content, list)
            and any(part.type == "image" for part in message.content)
        ):
            return True
    return False


def build_copilot_dynamic_headers(messages: list[Message], has_images: bool) -> dict[str, str]:
    headers = {
        "X-Initiator": infer_copilot_initiator(messages),
        "Openai-Intent": "conversation-edits",
    }

    if has_images:
        headers["Copilot-Vision-Request"] = "true"

    return headers
