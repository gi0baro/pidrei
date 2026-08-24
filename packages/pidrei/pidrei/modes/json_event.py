"""Mirror of pi coding-agent src/modes/json-event.ts.

pi expresses `JsonAgentSessionEvent` as a conditional type over
`AgentSessionEvent`; Python has no counterpart, so the shape lives only in the
runtime conversion below and in `docs/json.md`.
"""

import dataclasses
from typing import Any

from ..core.json_wire import _camel


__all__ = ["to_json_event"]


def _to_json_assistant_message_event(event: Any) -> Any:
    if not dataclasses.is_dataclass(event):
        return event

    fields = dataclasses.fields(event)
    if not any(field.name == "partial" for field in fields):
        return event

    # Built as a dict rather than a replaced dataclass so `partial` is gone from
    # the wire entirely; None values are dropped exactly as `to_wire` drops them
    # for dataclasses.
    delta = {}
    for field in fields:
        if field.name == "partial":
            continue
        value = getattr(event, field.name)
        if value is None:
            continue
        delta[_camel(field.name)] = value

    if event.type == "toolcall_start":
        tool_call = event.partial.content[event.content_index]
        if getattr(tool_call, "type", None) != "toolCall":
            raise RuntimeError(f"toolcall_start content at index {event.content_index} is not a tool call")
        delta["id"] = tool_call.id
        delta["toolName"] = tool_call.name

    return delta


def to_json_event(event: Any) -> Any:
    """Remove cumulative assistant snapshots from streaming wire events.

    `message_start` provides the initial message, deltas build it, and
    `message_end` provides the final authoritative message. Cumulative usage,
    tool-call ids, and tool names remain available because their size is
    constant — so a stream's total size stays linear in its output.
    """
    if getattr(event, "type", None) != "message_update":
        return event
    if getattr(event.message, "role", None) != "assistant":
        raise RuntimeError("message_update message is not an assistant message")

    return {
        "type": "message_update",
        "usage": event.message.usage,
        "assistantMessageEvent": _to_json_assistant_message_event(event.assistant_message_event),
    }
