"""Mirror of pi coding-agent src/modes/json-event.ts.

pi expresses `JsonAgentSessionEvent` as a conditional type over
`AgentSessionEvent`; Python has no counterpart, so the shape lives only in the
runtime conversion below and in `docs/json.md`.
"""

import dataclasses
from typing import Any

from ..core.json_wire import _camel


__all__ = ["to_json_event"]


def to_json_event(event: Any) -> Any:
    """Remove cumulative assistant snapshots from streaming wire events.

    `message_start` provides the initial message, deltas build it, and
    `message_end` provides the final authoritative message. Cumulative usage
    remains available because its size is constant. The inner event loses its
    `partial` snapshot — so a stream's total size stays linear in its output.
    """
    if getattr(event, "type", None) != "message_update":
        return event
    if getattr(event.message, "role", None) != "assistant":
        raise RuntimeError("message_update message is not an assistant message")

    usage = event.message.usage
    assistant_message_event = event.assistant_message_event
    if not dataclasses.is_dataclass(assistant_message_event):
        return {"type": "message_update", "usage": usage, "assistantMessageEvent": assistant_message_event}

    fields = dataclasses.fields(assistant_message_event)
    if not any(field.name == "partial" for field in fields):
        return {"type": "message_update", "usage": usage, "assistantMessageEvent": assistant_message_event}

    # Built as a dict rather than a replaced dataclass so `partial` is gone from
    # the wire entirely; None values are dropped exactly as `to_wire` drops them
    # for dataclasses.
    delta = {}
    for field in fields:
        if field.name == "partial":
            continue
        value = getattr(assistant_message_event, field.name)
        if value is None:
            continue
        delta[_camel(field.name)] = value
    return {"type": "message_update", "usage": usage, "assistantMessageEvent": delta}
