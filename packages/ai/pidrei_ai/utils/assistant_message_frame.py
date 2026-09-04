"""Assistant-message frames (port of pi `ai/src/utils/assistant-message-frame.ts`).

Compact, replayable assistant-message progress: an encoder turns the live
`AssistantMessageEvent` stream into frames, and a reducer replays frames back
into an `AssistantMessage`. Terminal settlement (`done`/`error`) is
intentionally excluded and must be persisted separately.

pi's only consumers are its durable harness runtime, which is not ported
(UPSTREAM_EXPERIMENTAL_RULING.md); the utility lands with its mirrored tests
and no consumer. Landed at its `8b5899dc` state: the burst-safe rewrite
(`5c6655e7`) and the stream-compat restore are folded in; `0fdec07b`'s
`providerThinkingLevel` line follows with that field.

Freeze-at-seam translation: the encoder reads either a producer's live
`AssistantMessageBuilder` partial or a frozen snapshot (the field names are
the same) and emits frozen values; the reducer accumulates through the
mutable builders and freezes the result. `structuredClone` of user-owned
dicts (`arguments`, diagnostic `details`) is `copy.deepcopy`.
"""

import copy
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from pidrei_ai.builders import (
    AssistantMessageBuilder,
    TextContentBuilder,
    ThinkingContentBuilder,
    ToolCallBuilder,
    UsageBuilder,
)
from pidrei_ai.types import AssistantMessage, AssistantMessageEvent, TextContent, ThinkingContent, ToolCall

from .json_parse import parse_streaming_json


# --- frames ---------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class StartFrame:
    partial: AssistantMessage
    type: Literal["start"] = "start"


@dataclass(slots=True, frozen=True)
class TextStartFrame:
    content_index: int
    content: TextContent
    type: Literal["text_start"] = "text_start"


@dataclass(slots=True, frozen=True)
class TextDeltaFrame:
    content_index: int
    delta: str
    type: Literal["text_delta"] = "text_delta"


@dataclass(slots=True, frozen=True)
class TextEndFrame:
    content_index: int
    content: str
    text_signature: str | None = None
    type: Literal["text_end"] = "text_end"


@dataclass(slots=True, frozen=True)
class ThinkingStartFrame:
    content_index: int
    content: ThinkingContent
    type: Literal["thinking_start"] = "thinking_start"


@dataclass(slots=True, frozen=True)
class ThinkingDeltaFrame:
    content_index: int
    delta: str
    type: Literal["thinking_delta"] = "thinking_delta"


@dataclass(slots=True, frozen=True)
class ThinkingEndFrame:
    content_index: int
    content: str
    thinking_signature: str | None = None
    redacted: bool | None = None
    type: Literal["thinking_end"] = "thinking_end"


@dataclass(slots=True, frozen=True)
class ToolCallStartFrame:
    content_index: int
    tool_call: ToolCall
    type: Literal["toolcall_start"] = "toolcall_start"


@dataclass(slots=True, frozen=True)
class ToolCallCheckpointFrame:
    content_index: int
    json: str
    type: Literal["toolcall_checkpoint"] = "toolcall_checkpoint"


@dataclass(slots=True, frozen=True)
class ToolCallDeltaFrame:
    content_index: int
    delta: str
    type: Literal["toolcall_delta"] = "toolcall_delta"


@dataclass(slots=True, frozen=True)
class ToolCallEndFrame:
    content_index: int
    id: str
    name: str
    arguments: dict[str, Any]
    thought_signature: str | None = None
    namespace: str | None = None
    type: Literal["toolcall_end"] = "toolcall_end"


type AssistantMessageFrame = (
    StartFrame
    | TextStartFrame
    | TextDeltaFrame
    | TextEndFrame
    | ThinkingStartFrame
    | ThinkingDeltaFrame
    | ThinkingEndFrame
    | ToolCallStartFrame
    | ToolCallCheckpointFrame
    | ToolCallDeltaFrame
    | ToolCallEndFrame
)


# --- encoder state ----------------------------------------------------------------


@dataclass(slots=True)
class _TextEncoderState:
    kind: Literal["text", "thinking"]
    covered_chars: int
    delta_chars: int


@dataclass(slots=True)
class _ToolCallEncoderState:
    caught_up: bool
    catchup_json: str
    snapshot_arguments: str
    kind: Literal["toolCall"] = "toolCall"


type _EncoderBlockState = _TextEncoderState | _ToolCallEncoderState


@dataclass(slots=True)
class _ReducerBlockState:
    kind: Literal["text", "thinking", "toolCall"]
    ended: bool
    json: str = ""


def _clone_text_content(content) -> TextContent:
    return TextContent(text=content.text, text_signature=content.text_signature)


def _clone_thinking_content(content) -> ThinkingContent:
    return ThinkingContent(
        thinking=content.thinking,
        thinking_signature=content.thinking_signature,
        redacted=content.redacted,
    )


def _clone_tool_call(tool_call) -> ToolCall:
    return ToolCall(
        id=tool_call.id,
        name=tool_call.name,
        arguments=copy.deepcopy(tool_call.arguments),
        thought_signature=tool_call.thought_signature,
        namespace=tool_call.namespace,
    )


def _clone_start_message(message) -> AssistantMessage:
    usage = message.usage
    return AssistantMessage(
        content=[],
        api=message.api,
        provider=message.provider,
        model=message.model,
        response_model=message.response_model,
        response_id=message.response_id,
        diagnostics=copy.deepcopy(message.diagnostics) if message.diagnostics is not None else None,
        usage=usage.freeze() if isinstance(usage, UsageBuilder) else usage,
        stop_reason="pending",
        timestamp=message.timestamp,
    )


def _assert_content_index(content_index: int) -> None:
    if isinstance(content_index, bool) or not isinstance(content_index, int) or content_index < 0:
        raise Exception(f"Invalid assistant message frame contentIndex: {content_index}")


def _event_block(event):
    _assert_content_index(event.content_index)
    content = event.partial.content
    block = content[event.content_index] if event.content_index < len(content) else None
    if block is None:
        raise Exception(f"{event.type} event has no content block at index {event.content_index}")
    return block


def _serialized_arguments(arguments_value: Any) -> str:
    try:
        return json.dumps(arguments_value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise Exception("Tool-call arguments are not JSON-serializable") from error


_EMPTY_PARSED_TOOL_ARGUMENTS = _serialized_arguments(parse_streaming_json(""))


def _is_json_prefix(snapshot: Any, current: Any) -> bool:
    if isinstance(snapshot, str):
        return isinstance(current, str) and current.startswith(snapshot)
    if isinstance(snapshot, list):
        return (
            isinstance(current, list)
            and len(snapshot) <= len(current)
            and all(_is_json_prefix(value, current[index]) for index, value in enumerate(snapshot))
        )
    if not isinstance(snapshot, dict):
        # `Object.is`: JS has one number type, so bool stays distinct from int here.
        if isinstance(snapshot, bool) or isinstance(current, bool):
            return type(snapshot) is type(current) and snapshot == current
        return snapshot == current
    if not isinstance(current, dict):
        return False
    return all(key in current and _is_json_prefix(value, current[key]) for key, value in snapshot.items())


class AssistantMessageFrameEncoder:
    """Encodes one assistant stream.

    `partial` remains a shared live accumulator; the encoder uses per-block
    offsets to avoid replaying deltas already visible when an older queued
    event is consumed.
    """

    def __init__(self) -> None:
        self._started = False
        self._terminal = False
        self._blocks: dict[int, _EncoderBlockState] = {}

    def encode(self, event: AssistantMessageEvent) -> AssistantMessageFrame | None:
        if self._terminal:
            raise Exception(f"Assistant message event {event.type} follows a terminal event")

        event_type = event.type
        if event_type == "start":
            if self._started:
                raise Exception("Assistant message stream contains more than one start event")
            self._started = True
            return StartFrame(partial=_clone_start_message(event.partial))
        if event_type == "done":
            if not self._started:
                raise Exception("Assistant message done event appears before start")
            self._terminal = True
            return None
        if event_type == "error":
            self._terminal = True
            return None

        if not self._started:
            raise Exception(f"Assistant message {event.type} event appears before start")

        if event_type == "text_start":
            content = _event_block(event)
            if content.type != "text":
                raise Exception(f"text_start event points to {content.type} block at index {event.content_index}")
            self._start_block(
                event.content_index, _TextEncoderState(kind="text", covered_chars=len(content.text), delta_chars=0)
            )
            return TextStartFrame(content_index=event.content_index, content=_clone_text_content(content))
        if event_type == "text_delta":
            return self._encode_text_delta(event.content_index, event.delta, "text")
        if event_type == "text_end":
            content = _event_block(event)
            if content.type != "text":
                raise Exception(f"text_end event points to {content.type} block at index {event.content_index}")
            self._end_block(event.content_index, "text")
            return TextEndFrame(
                content_index=event.content_index, content=event.content, text_signature=content.text_signature
            )
        if event_type == "thinking_start":
            content = _event_block(event)
            if content.type != "thinking":
                raise Exception(f"thinking_start event points to {content.type} block at index {event.content_index}")
            self._start_block(
                event.content_index,
                _TextEncoderState(kind="thinking", covered_chars=len(content.thinking), delta_chars=0),
            )
            return ThinkingStartFrame(content_index=event.content_index, content=_clone_thinking_content(content))
        if event_type == "thinking_delta":
            return self._encode_text_delta(event.content_index, event.delta, "thinking")
        if event_type == "thinking_end":
            content = _event_block(event)
            if content.type != "thinking":
                raise Exception(f"thinking_end event points to {content.type} block at index {event.content_index}")
            self._end_block(event.content_index, "thinking")
            return ThinkingEndFrame(
                content_index=event.content_index,
                content=event.content,
                thinking_signature=content.thinking_signature,
                redacted=content.redacted,
            )
        if event_type == "toolcall_start":
            content = _event_block(event)
            if content.type != "toolCall":
                raise Exception(f"toolcall_start event points to {content.type} block at index {event.content_index}")
            snapshot_arguments = _serialized_arguments(content.arguments)
            caught_up = snapshot_arguments == _EMPTY_PARSED_TOOL_ARGUMENTS
            self._start_block(
                event.content_index,
                _ToolCallEncoderState(
                    caught_up=caught_up,
                    catchup_json="",
                    snapshot_arguments="" if caught_up else snapshot_arguments,
                ),
            )
            return ToolCallStartFrame(content_index=event.content_index, tool_call=_clone_tool_call(content))
        if event_type == "toolcall_delta":
            state = self._block(event.content_index, "toolCall")
            if state.kind != "toolCall":
                raise Exception("Unreachable tool-call encoder state")
            if state.caught_up:
                return (
                    None
                    if len(event.delta) == 0
                    else ToolCallDeltaFrame(content_index=event.content_index, delta=event.delta)
                )
            state.catchup_json += event.delta
            arguments_value = parse_streaming_json(state.catchup_json)
            if _serialized_arguments(arguments_value) != state.snapshot_arguments:
                # Legacy grammar calls include the initial input in toolcall_start, but their
                # JSON delta stream still begins at an empty input. Its parsed arguments can
                # therefore extend, rather than exactly reproduce, the start snapshot.
                snapshot_arguments = parse_streaming_json(state.snapshot_arguments)
                if not _is_json_prefix(snapshot_arguments, arguments_value):
                    return None
            state.caught_up = True
            state.snapshot_arguments = ""
            json_text = state.catchup_json
            state.catchup_json = ""
            return (
                None
                if len(json_text) == 0
                else ToolCallCheckpointFrame(content_index=event.content_index, json=json_text)
            )
        if event_type == "toolcall_end":
            content = _event_block(event)
            if content.type != "toolCall":
                raise Exception(f"toolcall_end event points to {content.type} block at index {event.content_index}")
            tool_call = event.tool_call
            if tool_call.type != "toolCall":
                raise Exception(f"toolcall_end event has invalid tool call at index {event.content_index}")
            self._end_block(event.content_index, "toolCall")
            return ToolCallEndFrame(
                content_index=event.content_index,
                id=tool_call.id,
                name=tool_call.name,
                arguments=copy.deepcopy(tool_call.arguments),
                thought_signature=tool_call.thought_signature,
                namespace=tool_call.namespace,
            )
        raise Exception(f"Unknown assistant message event {event_type}")

    def _start_block(self, content_index: int, state: _EncoderBlockState) -> None:
        _assert_content_index(content_index)
        if content_index in self._blocks:
            raise Exception(f"Assistant message block {content_index} starts more than once")
        self._blocks[content_index] = state

    def _block(self, content_index: int, kind: str) -> _EncoderBlockState:
        _assert_content_index(content_index)
        state = self._blocks.get(content_index)
        if state is None:
            raise Exception(f"Assistant message {kind} block {content_index} has not started")
        if state.kind != kind:
            raise Exception(f"Assistant message block {content_index} is {state.kind}, not {kind}")
        return state

    def _end_block(self, content_index: int, kind: str) -> None:
        self._block(content_index, kind)
        del self._blocks[content_index]

    def _encode_text_delta(self, content_index: int, delta: str, kind: str) -> AssistantMessageFrame | None:
        state = self._block(content_index, kind)
        if state.kind == "toolCall":
            raise Exception("Unreachable text encoder state")
        delta_start = state.delta_chars
        state.delta_chars += len(delta)
        covered = max(0, state.covered_chars - delta_start)
        if covered >= len(delta):
            return None
        uncovered = delta if covered == 0 else delta[covered:]
        if kind == "text":
            return TextDeltaFrame(content_index=content_index, delta=uncovered)
        return ThinkingDeltaFrame(content_index=content_index, delta=uncovered)


# --- reducer --------------------------------------------------------------------


def _append_block(
    message: AssistantMessageBuilder,
    states: dict[int, _ReducerBlockState],
    content_index: int,
    block,
    state: _ReducerBlockState,
) -> None:
    _assert_content_index(content_index)
    if content_index != len(message.content):
        reason = "already exists" if content_index < len(message.content) else "would leave a gap"
        raise Exception(f"Cannot start assistant message block at index {content_index}: {reason}")
    message.content.append(block)
    states[content_index] = state


def _active_block(
    message: AssistantMessageBuilder,
    states: dict[int, _ReducerBlockState],
    content_index: int,
    expected_kind: str,
    frame_type: str,
):
    _assert_content_index(content_index)
    state = states.get(content_index)
    block = message.content[content_index] if content_index < len(message.content) else None
    if state is None or block is None:
        raise Exception(f"{frame_type} frame has no started block at index {content_index}")
    if state.kind != expected_kind or block.type != expected_kind:
        raise Exception(
            f"{frame_type} frame expected {expected_kind} block at index {content_index}, found {block.type}"
        )
    if state.ended:
        raise Exception(f"{frame_type} frame follows the end of block at index {content_index}")
    return block, state


def reduce_assistant_message_frames(
    frames: Iterable[AssistantMessageFrame],
) -> AssistantMessage | None:
    """Replay compact frames without mutating them.

    Returns `None` when the iterable contains no start frame.
    """
    message: AssistantMessageBuilder | None = None
    frame_before_start: str | None = None
    states: dict[int, _ReducerBlockState] = {}

    for frame in frames:
        frame_type = frame.type
        if frame_type == "start":
            if message is not None:
                raise Exception("Assistant message frame sequence contains more than one start frame")
            if frame_before_start is not None:
                raise Exception(f"{frame_before_start} frame appears before the start frame")
            partial = frame.partial
            message = AssistantMessageBuilder.from_message(
                partial,
                content=list(partial.content),
                diagnostics=copy.deepcopy(partial.diagnostics) if partial.diagnostics is not None else None,
            )
            continue
        if message is None:
            if frame_before_start is None:
                frame_before_start = frame_type
            continue

        if frame_type == "text_start":
            if frame.content.type != "text":
                raise Exception(f"text_start frame contains {frame.content.type} content")
            _append_block(
                message,
                states,
                frame.content_index,
                TextContentBuilder(text=frame.content.text, text_signature=frame.content.text_signature),
                _ReducerBlockState(kind="text", ended=False),
            )
        elif frame_type == "text_delta":
            block, _ = _active_block(message, states, frame.content_index, "text", frame_type)
            if block.type != "text":
                raise Exception("Unreachable text frame state")
            block.text += frame.delta
        elif frame_type == "text_end":
            block, state = _active_block(message, states, frame.content_index, "text", frame_type)
            if block.type != "text":
                raise Exception("Unreachable text frame state")
            block.text = frame.content
            block.text_signature = frame.text_signature
            state.ended = True
        elif frame_type == "thinking_start":
            if frame.content.type != "thinking":
                raise Exception(f"thinking_start frame contains {frame.content.type} content")
            _append_block(
                message,
                states,
                frame.content_index,
                ThinkingContentBuilder(
                    thinking=frame.content.thinking,
                    thinking_signature=frame.content.thinking_signature,
                    redacted=frame.content.redacted,
                ),
                _ReducerBlockState(kind="thinking", ended=False),
            )
        elif frame_type == "thinking_delta":
            block, _ = _active_block(message, states, frame.content_index, "thinking", frame_type)
            if block.type != "thinking":
                raise Exception("Unreachable thinking frame state")
            block.thinking += frame.delta
        elif frame_type == "thinking_end":
            block, state = _active_block(message, states, frame.content_index, "thinking", frame_type)
            if block.type != "thinking":
                raise Exception("Unreachable thinking frame state")
            block.thinking = frame.content
            block.thinking_signature = frame.thinking_signature
            # pi deletes `redacted` then re-adds it when the frame carries one;
            # the frozen type's default stands in for the deleted field.
            block.redacted = frame.redacted if frame.redacted is not None else False
            state.ended = True
        elif frame_type == "toolcall_start":
            if frame.tool_call.type != "toolCall":
                raise Exception(f"toolcall_start frame contains {frame.tool_call.type} content")
            _append_block(
                message,
                states,
                frame.content_index,
                ToolCallBuilder(
                    id=frame.tool_call.id,
                    name=frame.tool_call.name,
                    arguments=copy.deepcopy(frame.tool_call.arguments),
                    thought_signature=frame.tool_call.thought_signature,
                    namespace=frame.tool_call.namespace,
                ),
                _ReducerBlockState(kind="toolCall", ended=False, json=""),
            )
        elif frame_type == "toolcall_checkpoint":
            block, state = _active_block(message, states, frame.content_index, "toolCall", frame_type)
            if block.type != "toolCall" or state.kind != "toolCall":
                raise Exception("Unreachable tool-call checkpoint state")
            state.json = frame.json
            block.arguments = parse_streaming_json(frame.json)
        elif frame_type == "toolcall_delta":
            block, state = _active_block(message, states, frame.content_index, "toolCall", frame_type)
            if block.type != "toolCall" or state.kind != "toolCall":
                raise Exception("Unreachable tool-call frame state")
            state.json += frame.delta
        elif frame_type == "toolcall_end":
            block, state = _active_block(message, states, frame.content_index, "toolCall", frame_type)
            if block.type != "toolCall":
                raise Exception("Unreachable tool-call frame state")
            block.id = frame.id
            block.name = frame.name
            block.arguments = copy.deepcopy(frame.arguments)
            block.thought_signature = frame.thought_signature
            block.namespace = frame.namespace
            state.ended = True

    if message is None:
        return None
    for content_index, state in states.items():
        if state.kind != "toolCall" or state.ended or len(state.json) == 0:
            continue
        block = message.content[content_index] if content_index < len(message.content) else None
        if block is None or block.type != "toolCall":
            raise Exception("Unreachable tool-call frame state")
        block.arguments = parse_streaming_json(state.json)

    return message.freeze()


__all__ = [
    "AssistantMessageFrame",
    "AssistantMessageFrameEncoder",
    "StartFrame",
    "TextDeltaFrame",
    "TextEndFrame",
    "TextStartFrame",
    "ThinkingDeltaFrame",
    "ThinkingEndFrame",
    "ThinkingStartFrame",
    "ToolCallCheckpointFrame",
    "ToolCallDeltaFrame",
    "ToolCallEndFrame",
    "ToolCallStartFrame",
    "reduce_assistant_message_frames",
]
