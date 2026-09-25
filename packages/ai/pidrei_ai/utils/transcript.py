"""Port of pi's transcript helpers (packages/ai/src/utils/transcript.ts).

System prompt text and tool declarations live in the transcript as system
messages: the leading one is the prompt, later ones patch it (see
`SystemMessage`). These helpers normalize a caller `Context` into that shape and
replay system messages into the current prompt and tool set.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pidrei_ai.types import Context, Message, SystemMessage, Tool, ToolReference, TranscriptContext
from pidrei_ai.utils.text import content_text, get_system_message_text


__all__ = [
    "ToolStateChanges",
    "TranscriptContext",
    "TranscriptTools",
    "collapse_system_messages",
    "create_initial_system_message",
    "declarations_equal",
    "get_current_system_message",
    "get_current_system_prompt",
    "get_current_tools",
    "get_declared_tools",
    "get_initial_system_message",
    "get_tool_state_changes",
    "has_non_additive_tool_changes",
    "has_tool_redefinitions",
    "normalize_context",
    "resolve_transcript",
    "resolve_transcript_tools",
    "to_tool_declaration",
    "without_initial_system_message",
]


def create_initial_system_message(system_prompt: str | None, tools: list[Tool] | None) -> SystemMessage | None:
    """Build the leading system message for a prompt and tool set. Returns None when
    both are empty, so an empty transcript stays empty."""
    has_system_prompt = system_prompt is not None and len(system_prompt) > 0
    has_tools = tools is not None and len(tools) > 0
    if not has_system_prompt and not has_tools:
        return None
    # A copy: the message is frozen and published, and the caller keeps its list.
    return SystemMessage(content=system_prompt or "", tools_added=list(tools) if has_tools else None, timestamp=0)


def normalize_context(context: Context | TranscriptContext) -> TranscriptContext:
    """Fold `Context.system_prompt` and `Context.tools` into a leading system message.

    This is the only entry point that produces a `TranscriptContext`; every
    provider-facing function expects the result. An already-normalized context
    passes through unchanged: pi's structural typing lets a `TranscriptContext`
    flow into `Models.streamSimple(context: Context)` (the agent loop does this),
    where normalizing it again is a no-op.
    """
    if isinstance(context, TranscriptContext):
        return context
    initial_message = create_initial_system_message(context.system_prompt, context.tools)
    # Always a new list: adapters read it later on their setup task, while the
    # caller keeps its `Context`.
    messages = [initial_message, *context.messages] if initial_message is not None else list(context.messages)
    return TranscriptContext(messages=messages)


# Any message sequence. The replay helpers only read entries whose role is "system",
# so agent transcripts that carry custom message roles can be passed without filtering.
type TranscriptMessages = Sequence[Any]


def _is_system_message(message: Any) -> bool:
    return getattr(message, "role", None) == "system"


def get_initial_system_message(messages: TranscriptMessages) -> SystemMessage | None:
    """Return the leading system message, if the transcript starts with one."""
    if messages and _is_system_message(messages[0]):
        return messages[0]
    return None


def without_initial_system_message(messages: list[Message]) -> list[Message]:
    """Drop the leading system message for APIs that carry the prompt outside the message list."""
    return messages[1:] if get_initial_system_message(messages) is not None else messages


def get_current_tools(messages: TranscriptMessages) -> list[Tool]:
    """Resolve the tools available after applying every transcript delta in order."""
    tools: dict[str, Tool] = {}
    for message in messages:
        if not _is_system_message(message):
            continue
        for tool in message.tools_removed or []:
            tools.pop(tool.name, None)
        for tool in message.tools_added or []:
            # Like JS Map.set: an existing name keeps its position.
            tools[tool.name] = tool
    return list(tools.values())


def get_current_system_message(messages: TranscriptMessages) -> SystemMessage | None:
    """Replay every system message into one leading system message holding the current
    prompt and tools. Later `content` is appended to the base prompt, `sections` are
    patched by name, and tools are resolved with `get_current_tools`."""
    content: list[str] = []
    sections: dict[str, str] = {}
    timestamp: int | None = None
    for message in messages:
        if not _is_system_message(message):
            continue
        if timestamp is None:
            timestamp = message.timestamp
        text = content_text(message.content)
        if len(text) > 0:
            content.append(text)
        for name, value in (message.sections or {}).items():
            if value is None:
                sections.pop(name, None)
            else:
                sections[name] = value
    tools = get_current_tools(messages)
    if timestamp is None and len(tools) == 0:
        return None
    return SystemMessage(
        content="\n\n".join(content),
        sections=dict(sections) if sections else None,
        tools_added=tools if tools else None,
        timestamp=timestamp if timestamp is not None else 0,
    )


def get_current_system_prompt(messages: TranscriptMessages) -> str:
    """Render the current system prompt text after replaying every system message."""
    message = get_current_system_message(messages)
    return get_system_message_text(message) if message is not None else ""


def collapse_system_messages(context: TranscriptContext) -> TranscriptContext:
    """Rebuild the transcript for APIs without mid-conversation system messages: the
    replayed system message leads, and every later system message is dropped."""
    head = get_current_system_message(context.messages)
    messages = [message for message in context.messages if message.role != "system"]
    return TranscriptContext(messages=[head, *messages] if head is not None else messages)


def resolve_transcript(
    context: TranscriptContext, supports_mid_convo_system_messages: bool | None
) -> TranscriptContext:
    """Keep later system messages in place when the model accepts them; otherwise collapse them."""
    return context if supports_mid_convo_system_messages else collapse_system_messages(context)


def to_tool_declaration(tool: Tool) -> Tool:
    """Strip executable and display-only fields from a tool before transcript comparison or persistence."""
    return Tool(
        name=tool.name,
        description=tool.description,
        # JSON round-trip, like pi's: a detached, plain-data copy of the schema.
        parameters=json.loads(json.dumps(tool.parameters)),
        constrained_sampling=tool.constrained_sampling,
    )


def _declaration_json(tool: Tool) -> str:
    declaration = to_tool_declaration(tool)
    shape: dict[str, Any] = {
        "name": declaration.name,
        "description": declaration.description,
        "parameters": declaration.parameters,
    }
    sampling = declaration.constrained_sampling
    if sampling is False:
        shape["constrainedSampling"] = False
    elif sampling is not None:
        shape["constrainedSampling"] = (
            {"type": "json_schema", "strict": sampling.strict}
            if sampling.type == "json_schema"
            else {"type": "grammar", "variants": dict(sampling.variants)}
        )
    return json.dumps(shape, sort_keys=True)


def declarations_equal(left: Tool, right: Tool) -> bool:
    """Whether two tools declare the same interface to the model.

    pi compares `JSON.stringify` of both declarations; this compares a sorted-key
    JSON rendering of the same declaration shape.
    """
    return _declaration_json(left) == _declaration_json(right)


@dataclass(slots=True)
class ToolStateChanges:
    tools_added: list[Tool]
    tools_removed: list[ToolReference]


def get_tool_state_changes(previous: Sequence[Tool], current: Sequence[Tool]) -> ToolStateChanges:
    """Compare two complete tool states. A changed definition is a removal followed by an addition."""
    previous_tools = {tool.name: tool for tool in previous}
    current_tools = {tool.name: tool for tool in current}
    return ToolStateChanges(
        tools_added=[
            to_tool_declaration(tool)
            for tool in current
            if (previous_tool := previous_tools.get(tool.name)) is None or not declarations_equal(previous_tool, tool)
        ],
        tools_removed=[
            ToolReference(name=tool.name)
            for tool in previous
            if (current_tool := current_tools.get(tool.name)) is None or not declarations_equal(tool, current_tool)
        ],
    )


def get_declared_tools(messages: TranscriptMessages) -> list[Tool]:
    """Every definition referenced by transcript tool state, in first-declaration order."""
    definitions: dict[str, Tool] = {}
    for message in messages:
        if not _is_system_message(message):
            continue
        for tool in message.tools_added or []:
            definitions[tool.name] = tool
    return list(definitions.values())


def has_tool_redefinitions(messages: TranscriptMessages) -> bool:
    """Whether a tool name was declared twice with different definitions. Transports that
    reference tools by name (Anthropic `tool_addition`/`tool_removal`) cannot express that."""
    declared: dict[str, Tool] = {}
    for message in messages:
        if not _is_system_message(message):
            continue
        for tool in message.tools_added or []:
            previous = declared.get(tool.name)
            if previous is not None and not declarations_equal(previous, tool):
                return True
            declared[tool.name] = tool
    return False


def has_non_additive_tool_changes(messages: TranscriptMessages) -> bool:
    """Whether tool history contains a removal or same-name redeclaration that an
    addition-only transport cannot replay."""
    declared: set[str] = set()
    for message in messages:
        if not _is_system_message(message):
            continue
        if message.tools_removed:
            return True
        for tool in message.tools_added or []:
            if tool.name in declared:
                return True
            declared.add(tool.name)
    return False


@dataclass(slots=True)
class TranscriptTools:
    # Tools sent in the top-level request field.
    request_tools: list[Tool]
    # Whether later system messages carry their own `tools_added` as in-place additions.
    # When False, `request_tools` already holds the complete current tool set.
    anchors_additions: bool


def resolve_transcript_tools(messages: TranscriptMessages, supports_tool_additions: bool) -> TranscriptTools:
    """Split tool declarations between the top-level request field and in-place additions.

    Transports that can anchor additions at a system message keep the initial tools at
    the top and load later ones where they appear; that only works when no tool was
    removed or redeclared, so everything else sends the current tool list.
    """
    anchors_additions = supports_tool_additions and not has_non_additive_tool_changes(messages)
    if anchors_additions:
        initial = get_initial_system_message(messages)
        request_tools = list(initial.tools_added or []) if initial is not None else []
    else:
        request_tools = get_current_tools(messages)
    return TranscriptTools(request_tools=request_tools, anchors_additions=anchors_additions)
