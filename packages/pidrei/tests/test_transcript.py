"""Mirror of pi coding-agent test/client/transcript.test.ts."""

from pidrei.client.transcript import (
    apply_transcript_progress,
    apply_transcript_snapshot,
    create_transcript_state,
    select_transcript,
)
from pidrei_protocol import SessionSnapshot


def snapshot(revision: int, text: str = "saved") -> SessionSnapshot:
    return {
        "id": "session-1",
        "cwd": "/workspace",
        "createdAt": 1,
        "updatedAt": revision + 1,
        "phase": "turn",
        "model": {"provider": "faux", "id": "faux-1"},
        "thinkingLevel": "off",
        "attached": True,
        "locked": True,
        "revision": revision,
        "transcript": [
            {
                "id": "assistant-1",
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
                "status": "streaming",
                "model": {"provider": "faux", "id": "faux-1"},
                "timestamp": 1,
            }
        ],
        "queuedSteer": [],
        "queuedSteerCount": 0,
    }


def test_projects_progress_without_mutating_the_authoritative_snapshot():
    state = create_transcript_state(snapshot(1))
    state = apply_transcript_progress(
        state,
        {
            "type": "assistant_delta",
            "messageId": "assistant-1",
            "contentIndex": 0,
            "kind": "text",
            "delta": " response",
        },
    )

    assert state.snapshot["transcript"][0]["content"] == [{"type": "text", "text": "saved"}]
    assert select_transcript(state)[0]["content"] == [{"type": "text", "text": "saved response"}]


def test_applies_streamed_tool_call_argument_deltas():
    state = create_transcript_state(
        {
            **snapshot(1),
            "transcript": [
                {
                    "id": "assistant-1",
                    "role": "assistant",
                    "content": [{"type": "toolCall", "toolCallId": "call-1", "toolName": "bash", "input": None}],
                    "status": "streaming",
                    "model": {"provider": "faux", "id": "faux-1"},
                    "timestamp": 1,
                }
            ],
        }
    )
    state = apply_transcript_progress(
        state,
        {
            "type": "assistant_delta",
            "messageId": "assistant-1",
            "contentIndex": 0,
            "kind": "toolCall",
            "delta": '{"command":',
        },
    )
    assert select_transcript(state)[0]["content"][0]["input"] == '{"command":'

    state = apply_transcript_progress(
        state,
        {
            "type": "item_updated",
            "item": {
                "id": "assistant-1",
                "role": "assistant",
                "content": [{"type": "toolCall", "toolCallId": "call-1", "toolName": "bash", "input": None}],
                "status": "streaming",
                "model": {"provider": "faux", "id": "faux-1"},
                "timestamp": 1,
            },
        },
    )
    state = apply_transcript_progress(
        state,
        {
            "type": "assistant_delta",
            "messageId": "assistant-1",
            "contentIndex": 0,
            "kind": "toolCall",
            "delta": '"pwd"}',
        },
    )
    assert select_transcript(state)[0]["content"][0]["input"] == {"command": "pwd"}


def test_appends_tool_call_deltas_to_a_partial_input_restored_from_a_snapshot():
    state = create_transcript_state(
        {
            **snapshot(1),
            "transcript": [
                {
                    "id": "assistant-1",
                    "role": "assistant",
                    "content": [
                        {"type": "toolCall", "toolCallId": "call-1", "toolName": "bash", "input": '{"command":'}
                    ],
                    "status": "streaming",
                    "model": {"provider": "faux", "id": "faux-1"},
                    "timestamp": 1,
                }
            ],
        }
    )

    state = apply_transcript_progress(
        state,
        {
            "type": "assistant_delta",
            "messageId": "assistant-1",
            "contentIndex": 0,
            "kind": "toolCall",
            "delta": '"pwd"}',
        },
    )

    assert select_transcript(state)[0]["content"][0]["input"] == {"command": "pwd"}


def test_appends_transient_tool_progress_and_replaces_it_by_id():
    state = create_transcript_state(snapshot(1))
    state = apply_transcript_progress(
        state,
        {
            "type": "item_started",
            "item": {
                "id": "tool-call-1",
                "role": "tool",
                "toolCallId": "call-1",
                "toolName": "bash",
                "input": {"command": "printf hi"},
                "content": [],
                "status": "running",
                "isError": False,
                "timestamp": 2,
            },
        },
    )
    started = select_transcript(state)[-1]
    assert started["id"] == "tool-call-1"
    assert started["role"] == "tool"
    assert started["status"] == "running"
    assert started["content"] == []

    state = apply_transcript_progress(
        state,
        {
            "type": "item_updated",
            "item": {
                "id": "tool-call-1",
                "role": "tool",
                "toolCallId": "call-1",
                "toolName": "bash",
                "input": {"command": "printf hi"},
                "content": [{"type": "text", "text": "hi"}],
                "status": "running",
                "isError": False,
                "timestamp": 2,
            },
        },
    )

    transcript = select_transcript(state)
    assert len(transcript) == 2
    assert transcript[1]["role"] == "tool"
    assert transcript[1]["status"] == "running"
    assert transcript[1]["content"] == [{"type": "text", "text": "hi"}]


def test_resets_revision_history_when_the_same_session_runtime_is_reacquired():
    create_transcript_state(snapshot(50, "old runtime"))
    state = create_transcript_state(snapshot(0, "new runtime"))

    assert state.snapshot["revision"] == 0
    assert select_transcript(state)[0]["content"] == [{"type": "text", "text": "new runtime"}]


def test_accepts_a_lower_revision_when_switching_to_a_different_session():
    state = create_transcript_state(snapshot(50, "old session"))
    state = apply_transcript_snapshot(state, {**snapshot(0, "new session"), "id": "session-2"})

    assert state.snapshot["id"] == "session-2"
    assert select_transcript(state)[0]["content"] == [{"type": "text", "text": "new session"}]


def test_renders_accepted_steering_messages_from_authoritative_queued_state():
    state = create_transcript_state(
        {
            **snapshot(2),
            "queuedSteerCount": 1,
            "queuedSteer": [
                {
                    "id": "user-steer",
                    "role": "user",
                    "content": [{"type": "text", "text": "adjust the approach"}],
                    "timestamp": 2,
                }
            ],
        }
    )

    queued = select_transcript(state)[-1]
    assert queued["role"] == "user"
    assert queued["content"] == [{"type": "text", "text": "adjust the approach"}]


def test_a_newer_snapshot_is_authoritative_and_stale_snapshots_are_ignored():
    state = create_transcript_state(snapshot(3, "new"))
    state = apply_transcript_progress(
        state,
        {
            "type": "assistant_delta",
            "messageId": "assistant-1",
            "contentIndex": 0,
            "kind": "text",
            "delta": " transient",
        },
    )
    state = apply_transcript_snapshot(state, snapshot(4, "authoritative"))
    state = apply_transcript_snapshot(state, snapshot(2, "stale"))

    assert state.snapshot["revision"] == 4
    assert select_transcript(state)[0]["content"] == [{"type": "text", "text": "authoritative"}]
