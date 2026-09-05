"""Mirror of pi ai/test/assistant-message-frame.test.ts (at its `8b5899dc` state).

Freeze-at-seam translation: pi drives the encoder with a live mutable
`partial`; here that is an `AssistantMessageBuilder`, and where pi replaces a
block in place (`partial.content[0] = {...}`) the test assigns a frozen value
into the builder's content list. Reduced messages are frozen, so content
comparisons are against frozen values.

Dropped: "whitelists public block fields from provider-shaped partials" —
pidrei's blocks are slotted dataclasses, so provider scratch properties cannot
exist on them; the case would be vacuous.
"""

import pytest

from pidrei_ai.api.openai_responses_shared import process_responses_stream
from pidrei_ai.builders import AssistantMessageBuilder, TextContentBuilder, ToolCallBuilder, UsageBuilder
from pidrei_ai.types import (
    AssistantMessageDiagnostic,
    DoneEvent,
    ErrorEvent,
    Model,
    ModelCost,
    StartEvent,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingContent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from pidrei_ai.utils.assistant_message_frame import (
    AssistantMessageFrameEncoder,
    StartFrame,
    TextDeltaFrame,
    TextEndFrame,
    TextStartFrame,
    ThinkingDeltaFrame,
    ThinkingEndFrame,
    ThinkingStartFrame,
    ToolCallCheckpointFrame,
    ToolCallDeltaFrame,
    ToolCallEndFrame,
    ToolCallStartFrame,
    reduce_assistant_message_frames,
)
from pidrei_ai.utils.event_stream import AssistantMessageEventStream


def seed() -> AssistantMessageBuilder:
    return AssistantMessageBuilder(
        content=[],
        api="test-api",
        provider="test-provider",
        model="test-model",
        usage=UsageBuilder(),
        stop_reason="pending",
        timestamp=1,
    )


def frame(encoder: AssistantMessageFrameEncoder, event):
    converted = encoder.encode(event)
    if converted is None:
        raise Exception(f"Expected {event.type} event to produce a frame")
    return converted


def test_uses_authoritative_text_end_content_and_signature():
    partial = seed()
    encoder = AssistantMessageFrameEncoder()
    frames = [frame(encoder, StartEvent(partial=partial))]
    partial.content.append(TextContent(text="Hello "))
    frames.append(frame(encoder, TextStartEvent(content_index=0, partial=partial)))
    partial.content[0] = TextContent(text="Hello world", text_signature="sig-text")
    frames.append(frame(encoder, TextDeltaEvent(content_index=0, delta="incorrect", partial=partial)))
    frames.append(frame(encoder, TextEndEvent(content_index=0, content="Hello world", partial=partial)))

    assert frames[-1] == TextEndFrame(content_index=0, content="Hello world", text_signature="sig-text")
    assert reduce_assistant_message_frames(frames).content == [
        TextContent(text="Hello world", text_signature="sig-text")
    ]


def test_preserves_provider_thinking_level_from_the_stream_start():
    partial = seed()
    partial.provider_thinking_level = "high"
    encoder = AssistantMessageFrameEncoder()
    start = frame(encoder, StartEvent(partial=partial))

    assert start.type == "start"
    assert start.partial.provider_thinking_level == "high"
    assert reduce_assistant_message_frames([start]).provider_thinking_level == "high"


def test_preserves_initial_and_final_thinking_metadata_including_redaction():
    partial = seed()
    encoder = AssistantMessageFrameEncoder()
    frames = [frame(encoder, StartEvent(partial=partial))]
    partial.content.append(ThinkingContent(thinking="[redacted]", thinking_signature="encrypted-start", redacted=True))
    frames.append(frame(encoder, ThinkingStartEvent(content_index=0, partial=partial)))
    partial.content[0] = ThinkingContent(thinking="[redacted]", thinking_signature="encrypted-final", redacted=True)
    frames.append(frame(encoder, ThinkingEndEvent(content_index=0, content="[redacted]", partial=partial)))

    assert frames[-1] == ThinkingEndFrame(
        content_index=0, content="[redacted]", thinking_signature="encrypted-final", redacted=True
    )
    assert reduce_assistant_message_frames(frames).content[0] == ThinkingContent(
        thinking="[redacted]", thinking_signature="encrypted-final", redacted=True
    )


def test_parses_unfinished_tool_json_once_and_uses_authoritative_completed_arguments():
    initial_frames = [
        StartFrame(partial=seed().freeze()),
        ToolCallStartFrame(content_index=0, tool_call=ToolCall(id="initial-id", name="write", arguments={})),
        ToolCallDeltaFrame(content_index=0, delta='{"path":"READ'),
    ]

    reduced = reduce_assistant_message_frames(initial_frames).content[0]
    assert reduced.type == "toolCall"
    assert reduced.arguments == {"path": "READ"}

    complete_frames = [
        *initial_frames,
        ToolCallDeltaFrame(content_index=0, delta='ME.md","lines":[1,2]}'),
        ToolCallEndFrame(
            content_index=0,
            id="final-id",
            name="write_file",
            arguments={"path": "final.md", "lines": [3]},
            thought_signature="thought",
            namespace="files",
        ),
    ]
    assert reduce_assistant_message_frames(complete_frames).content[0] == ToolCall(
        id="final-id",
        name="write_file",
        arguments={"path": "final.md", "lines": [3]},
        thought_signature="thought",
        namespace="files",
    )


class _EncodingStream(AssistantMessageEventStream):
    """pi wraps `stream.push` with the encoder; the encoder sees the live
    partial before the publication seam freezes it."""

    def __init__(self, encoder: AssistantMessageFrameEncoder, frames: list) -> None:
        super().__init__()
        self._encoder = encoder
        self._frames = frames

    def push(self, event) -> None:
        converted = self._encoder.encode(event)
        if converted is not None:
            self._frames.append(converted)
        super().push(event)


@pytest.mark.tonio
async def test_round_trips_openai_responses_content_supplied_only_by_authoritative_end_events():
    output = seed()
    output.api = "openai-responses"
    output.provider = "openai"
    model = Model(
        id=output.model,
        name="Test",
        api="openai-responses",
        provider="openai",
        base_url="https://api.openai.com/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(),
        context_window=1000,
        max_tokens=100,
    )
    events = [
        {
            "type": "response.output_item.added",
            "sequence_number": 0,
            "output_index": 0,
            "item": {"type": "message", "id": "msg", "role": "assistant", "status": "in_progress", "content": []},
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 1,
            "output_index": 0,
            "item": {
                "type": "message",
                "id": "msg",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "final text", "annotations": []}],
            },
        },
        {
            "type": "response.output_item.added",
            "sequence_number": 2,
            "output_index": 1,
            "item": {"type": "function_call", "id": "fc", "call_id": "call", "name": "lookup", "arguments": ""},
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 3,
            "output_index": 1,
            "item": {
                "type": "function_call",
                "id": "fc",
                "call_id": "call",
                "name": "lookup",
                "arguments": '{"query":"pi"}',
            },
        },
        {
            "type": "response.completed",
            "sequence_number": 4,
            "response": {"id": "response", "status": "completed", "output": []},
        },
    ]

    async def source():
        for event in events:
            yield event

    encoder = AssistantMessageFrameEncoder()
    frames = [frame(encoder, StartEvent(partial=output))]
    stream = _EncodingStream(encoder, frames)
    await process_responses_stream(source(), output, stream, model)

    assert reduce_assistant_message_frames(frames).content == output.freeze().content


def test_reconciles_queued_text_events_against_one_advanced_live_partial_without_duplicate_content():
    partial = seed()
    events = [StartEvent(partial=partial)]
    text = TextContentBuilder(text="")
    partial.content.append(text)
    events.append(TextStartEvent(content_index=0, partial=partial))
    for delta in ["Hel", "lo", " ", "world"]:
        text.text += delta
        events.append(TextDeltaEvent(content_index=0, delta=delta, partial=partial))

    encoder = AssistantMessageFrameEncoder()
    frames = [encoded for event in events if (encoded := encoder.encode(event)) is not None]

    assert [item.type for item in frames] == ["start", "text_start"]
    assert frames[0].type == "start"
    assert frames[0].partial.content == []
    assert frames[0].partial.stop_reason == "pending"
    assert reduce_assistant_message_frames(frames).content == [TextContent(text="Hello world")]


def test_trims_only_the_covered_prefix_when_a_start_snapshot_lands_inside_a_delta():
    partial = seed()
    encoder = AssistantMessageFrameEncoder()
    frames = [frame(encoder, StartEvent(partial=partial))]
    text = TextContentBuilder(text="Hel")
    partial.content.append(text)
    frames.append(frame(encoder, TextStartEvent(content_index=0, partial=partial)))
    assert encoder.encode(TextDeltaEvent(content_index=0, delta="He", partial=partial)) is None
    remainder = encoder.encode(TextDeltaEvent(content_index=0, delta="llo", partial=partial))
    if remainder is None:
        raise Exception("Expected uncovered text delta")
    frames.append(remainder)

    assert remainder == TextDeltaFrame(content_index=0, delta="lo")
    assert reduce_assistant_message_frames(frames).content == [TextContent(text="Hello")]


def test_checkpoints_queued_tool_json_without_replaying_covered_deltas():
    partial = seed()
    tool_call = ToolCallBuilder(id="call", name="write", arguments={})
    events = [StartEvent(partial=partial)]
    partial.content.append(tool_call)
    events.append(ToolCallStartEvent(content_index=0, partial=partial))
    tool_call.arguments = {"path": "README.md"}
    events.append(ToolCallDeltaEvent(content_index=0, delta='{"path":"READ', partial=partial))
    events.append(ToolCallDeltaEvent(content_index=0, delta='ME.md"}', partial=partial))

    encoder = AssistantMessageFrameEncoder()
    frames = [encoded for event in events if (encoded := encoder.encode(event)) is not None]
    assert [item.type for item in frames] == ["start", "toolcall_start", "toolcall_checkpoint"]
    assert frames[-1] == ToolCallCheckpointFrame(content_index=0, json='{"path":"README.md"}')
    assert reduce_assistant_message_frames(frames).content == [
        ToolCall(id="call", name="write", arguments={"path": "README.md"})
    ]


def test_resumes_legacy_grammar_tool_json_from_initial_arguments():
    partial = seed()
    encoder = AssistantMessageFrameEncoder()
    frames = [frame(encoder, StartEvent(partial=partial))]
    tool_call = ToolCallBuilder(id="call", name="bash", arguments={"input": "a"})
    partial.content.append(tool_call)
    frames.append(frame(encoder, ToolCallStartEvent(content_index=0, partial=partial)))
    tool_call.arguments = {"input": "ab"}
    frames.append(frame(encoder, ToolCallDeltaEvent(content_index=0, delta='{"input":"ab', partial=partial)))
    tool_call.arguments = {"input": "abc"}
    frames.append(frame(encoder, ToolCallDeltaEvent(content_index=0, delta='c"}', partial=partial)))

    assert frames[2:] == [
        ToolCallCheckpointFrame(content_index=0, json='{"input":"ab'),
        ToolCallDeltaFrame(content_index=0, delta='c"}'),
    ]
    assert reduce_assistant_message_frames(frames).content == [
        ToolCall(id="call", name="bash", arguments={"input": "abc"})
    ]


def test_streams_tool_json_compactly_from_an_empty_argument_start():
    partial = seed()
    encoder = AssistantMessageFrameEncoder()
    frames = [frame(encoder, StartEvent(partial=partial))]
    tool_call = ToolCallBuilder(id="call", name="bash", arguments={})
    partial.content.append(tool_call)
    frames.append(frame(encoder, ToolCallStartEvent(content_index=0, partial=partial)))
    tool_call.arguments = {"command": "ls -la /tmp"}
    frames.append(
        frame(encoder, ToolCallDeltaEvent(content_index=0, delta='{"command":"ls -la /tmp"}', partial=partial))
    )

    assert frames[-1] == ToolCallDeltaFrame(content_index=0, delta='{"command":"ls -la /tmp"}')
    reduced = reduce_assistant_message_frames(frames).content[0]
    assert reduced.type == "toolCall"
    assert reduced.arguments == {"command": "ls -la /tmp"}


def test_accepts_a_pre_generation_error_but_rejects_success_or_updates_before_start():
    failed = seed()
    failed.stop_reason = "error"
    failed.error_message = "setup failed"
    assert AssistantMessageFrameEncoder().encode(ErrorEvent(reason="error", error=failed)) is None

    completed = seed()
    completed.stop_reason = "stop"
    with pytest.raises(Exception, match="done event appears before start"):
        AssistantMessageFrameEncoder().encode(DoneEvent(reason="stop", message=completed))
    with pytest.raises(Exception, match="text_delta event appears before start"):
        AssistantMessageFrameEncoder().encode(TextDeltaEvent(content_index=0, delta="x", partial=seed()))


def test_treats_end_signature_metadata_including_absence_as_authoritative():
    frames = [
        StartFrame(partial=seed().freeze()),
        TextStartFrame(content_index=0, content=TextContent(text="", text_signature="stale-text")),
        TextEndFrame(content_index=0, content=""),
        ThinkingStartFrame(
            content_index=1,
            content=ThinkingContent(thinking="", thinking_signature="stale-thinking", redacted=True),
        ),
        ThinkingEndFrame(content_index=1, content="", thinking_signature="", redacted=False),
        ToolCallStartFrame(
            content_index=2,
            tool_call=ToolCall(
                id="call", name="read", arguments={}, thought_signature="stale-tool", namespace="stale-namespace"
            ),
        ),
        ToolCallEndFrame(content_index=2, id="call", name="read", arguments={}),
    ]

    assert reduce_assistant_message_frames(frames).content == [
        TextContent(text=""),
        ThinkingContent(thinking="", thinking_signature="", redacted=False),
        ToolCall(id="call", name="read", arguments={}),
    ]


def test_stores_authoritative_final_arguments_in_toolcall_end_frames():
    partial = seed()
    tool_call = ToolCall(
        id="call-1", name="read", arguments={"path": "README.md"}, thought_signature="thought", namespace="files"
    )
    partial.content.append(tool_call)

    encoder = AssistantMessageFrameEncoder()
    frame(encoder, StartEvent(partial=partial))
    frame(encoder, ToolCallStartEvent(content_index=0, partial=partial))
    end = frame(encoder, ToolCallEndEvent(content_index=0, tool_call=tool_call, partial=partial))
    assert end == ToolCallEndFrame(
        content_index=0,
        id="call-1",
        name="read",
        arguments={"path": "README.md"},
        thought_signature="thought",
        namespace="files",
    )


def test_supports_interleaved_streams_by_content_index():
    frames = [
        StartFrame(partial=seed().freeze()),
        TextStartFrame(content_index=0, content=TextContent(text="")),
        ToolCallStartFrame(content_index=1, tool_call=ToolCall(id="call", name="lookup", arguments={})),
        ThinkingStartFrame(content_index=2, content=ThinkingContent(thinking="")),
        TextDeltaFrame(content_index=0, delta="answer"),
        ToolCallDeltaFrame(content_index=1, delta='{"query":"pi"}'),
        ThinkingDeltaFrame(content_index=2, delta="check"),
        ToolCallEndFrame(content_index=1, id="call", name="lookup", arguments={"query": "pi"}),
        TextEndFrame(content_index=0, content="answer"),
        ThinkingEndFrame(content_index=2, content="check"),
    ]

    assert reduce_assistant_message_frames(frames).content == [
        TextContent(text="answer"),
        ToolCall(id="call", name="lookup", arguments={"query": "pi"}),
        ThinkingContent(thinking="check"),
    ]


def test_snapshots_mutable_event_data_and_keeps_reduction_pure():
    partial = seed()
    partial.diagnostics = [AssistantMessageDiagnostic(type="test", timestamp=2, details={"value": "original"})]
    encoder = AssistantMessageFrameEncoder()
    start = frame(encoder, StartEvent(partial=partial))
    partial.diagnostics[0].details["value"] = "mutated"
    partial.usage.cost.total = 99

    partial.content.append(ToolCallBuilder(id="call", name="run", arguments={"nested": {"value": "original"}}))
    tool_start = frame(encoder, ToolCallStartEvent(content_index=0, partial=partial))
    source_tool = partial.content[0]
    assert source_tool.type == "toolCall"
    source_tool.arguments["nested"]["value"] = "mutated"

    reduced = reduce_assistant_message_frames([start, tool_start])
    assert reduced.diagnostics[0].details["value"] == "original"
    assert reduced.usage.cost.total == 0
    assert reduced.content[0].arguments == {"nested": {"value": "original"}}

    assert reduced.content[0].type == "toolCall"
    reduced.content[0].arguments["nested"] = "changed-output"
    assert tool_start.type == "toolcall_start"
    assert tool_start.tool_call.arguments["nested"] == {"value": "original"}


def test_omits_terminal_events_because_settlement_is_separate():
    message = seed()
    completed = AssistantMessageFrameEncoder()
    completed.encode(StartEvent(partial=message))
    message.stop_reason = "stop"
    assert completed.encode(DoneEvent(reason="stop", message=message)) is None
    message.stop_reason = "error"
    message.error_message = "failed"
    assert AssistantMessageFrameEncoder().encode(ErrorEvent(reason="error", error=message)) is None


def test_returns_none_when_there_is_no_start_frame():
    assert reduce_assistant_message_frames([]) is None
    assert reduce_assistant_message_frames([TextDeltaFrame(content_index=0, delta="x")]) is None


def test_rejects_frames_before_start_wrong_block_kinds_duplicate_ends_and_index_gaps():
    with pytest.raises(Exception, match="before the start frame"):
        reduce_assistant_message_frames(
            [TextDeltaFrame(content_index=0, delta="x"), StartFrame(partial=seed().freeze())]
        )
    with pytest.raises(Exception, match="expected text block"):
        reduce_assistant_message_frames(
            [
                StartFrame(partial=seed().freeze()),
                ToolCallStartFrame(content_index=0, tool_call=ToolCall(id="call", name="run", arguments={})),
                TextDeltaFrame(content_index=0, delta="wrong"),
            ]
        )
    with pytest.raises(Exception, match="follows the end"):
        reduce_assistant_message_frames(
            [
                StartFrame(partial=seed().freeze()),
                TextStartFrame(content_index=0, content=TextContent(text="")),
                TextEndFrame(content_index=0, content=""),
                TextEndFrame(content_index=0, content=""),
            ]
        )
    with pytest.raises(Exception, match="would leave a gap"):
        reduce_assistant_message_frames(
            [
                StartFrame(partial=seed().freeze()),
                TextStartFrame(content_index=1, content=TextContent(text="")),
            ]
        )


def test_rejects_conversion_events_whose_content_index_points_to_the_wrong_block_kind():
    partial = seed()
    encoder = AssistantMessageFrameEncoder()
    encoder.encode(StartEvent(partial=partial))
    partial.content.append(ThinkingContent(thinking=""))
    with pytest.raises(Exception, match="text_start event points to thinking block"):
        encoder.encode(TextStartEvent(content_index=0, partial=partial))
