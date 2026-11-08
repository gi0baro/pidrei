"""Mirror of pi's branch-summarization.test.ts.

pi's `streamFn` pushes the done event from a microtask; here the stream is
filled before it is returned, which the awaiting `result()` handles the same
way (see `test_compaction_summary_reasoning.py`).
"""

import time
from dataclasses import replace

import pytest

from pidrei.core.compaction import generate_branch_summary, prepare_branch_entries
from pidrei_ai.providers.faux import faux_assistant_message
from pidrei_ai.types import AssistantMessage, DoneEvent, Model, ModelCost, TextContent, ToolCall, UserMessage
from pidrei_ai.utils.event_stream import AssistantMessageEventStream


MODEL = Model(
    id="test-model",
    name="Test Model",
    api="anthropic-messages",
    provider="anthropic",
    base_url="https://api.anthropic.com",
    reasoning=False,
    input=["text"],
    cost=ModelCost(),
    context_window=200000,
    max_tokens=8192,
)

ENTRIES = [
    {
        "type": "message",
        "id": "branch-user",
        "parentId": None,
        "timestamp": "1970-01-01T00:00:00.001Z",
        "message": {"role": "user", "content": "Abandoned request", "timestamp": 1},
    }
]


def _response(content: list, stop_reason: str = "stop") -> AssistantMessage:
    # Freeze-at-seam relaxation (spec/upstream-sync.md): messages are frozen values now,
    # so the response shape is built by construction instead of mutation.
    return replace(
        faux_assistant_message("", stop_reason=stop_reason, timestamp=int(time.time() * 1000)),
        content=content,
        api=MODEL.api,
        provider=MODEL.provider,
        model=MODEL.id,
    )


def _stream_fn(message: AssistantMessage) -> tuple:
    calls: list = []

    async def stream_fn(_model, _context, options=None):
        calls.append(options)
        stream = AssistantMessageEventStream()
        stream.push(DoneEvent(reason="toolUse" if message.stop_reason == "toolUse" else "stop", message=message))
        return stream

    return stream_fn, calls


@pytest.mark.tonio
async def test_passes_the_thinking_level_only_for_reasoning_capable_models():
    stream_fn, calls = _stream_fn(_response([TextContent(text="summary")]))
    reasoning_model = replace(MODEL, reasoning=True)

    await generate_branch_summary(
        ENTRIES, model=reasoning_model, cancel=None, stream_fn=stream_fn, thinking_level="xhigh"
    )
    await generate_branch_summary(
        ENTRIES, model=reasoning_model, cancel=None, stream_fn=stream_fn, thinking_level="off"
    )
    await generate_branch_summary(ENTRIES, model=MODEL, cancel=None, stream_fn=stream_fn, thinking_level="high")

    assert [options.reasoning for options in calls] == ["xhigh", None, None]


@pytest.mark.tonio
async def test_does_not_override_tool_choice_for_branch_summaries():
    stream_fn, calls = _stream_fn(_response([TextContent(text="summary")]))

    await generate_branch_summary(ENTRIES, model=MODEL, cancel=None, stream_fn=stream_fn)

    assert len(calls) == 1
    assert calls[0].tool_choice is None


@pytest.mark.tonio
async def test_rejects_tool_calls_from_branch_summaries():
    tool_call = ToolCall(id="tool-call-1", name="read", arguments={"path": "README.md"})
    stream_fn, _calls = _stream_fn(_response([tool_call], stop_reason="toolUse"))

    result = await generate_branch_summary(ENTRIES, model=MODEL, cancel=None, stream_fn=stream_fn)

    assert result.error == "Branch summarization attempted to call a tool"


@pytest.mark.tonio
async def test_rejects_length_limited_branch_summaries():
    message = _response([TextContent(text="partial")], stop_reason="length")

    async def stream_fn(_model, _context, options=None):
        stream = AssistantMessageEventStream()
        stream.push(DoneEvent(reason="length", message=message))
        return stream

    result = await generate_branch_summary(ENTRIES, model=MODEL, cancel=None, stream_fn=stream_fn)

    assert result.error == "Branch summarization failed: generation hit the token cap and the summary is incomplete"


def test_applies_context_edits_to_branch_entries():
    def ts(n: int) -> str:
        return f"1970-01-01T00:00:00.00{n}Z"

    branch = [
        {
            "type": "message",
            "id": "u1",
            "parentId": None,
            "timestamp": ts(1),
            "message": UserMessage(content="Investigate login", timestamp=1),
        },
        {
            "type": "message",
            "id": "failed",
            "parentId": "u1",
            "timestamp": ts(2),
            "message": faux_assistant_message("FAILED_ATTEMPT", stop_reason="error", error_message="lost"),
        },
        {
            "type": "context_edit",
            "id": "e1",
            "parentId": "failed",
            "timestamp": ts(3),
            "targetId": "failed",
            "replacement": None,
        },
        {
            "type": "message",
            "id": "ok",
            "parentId": "e1",
            "timestamp": ts(4),
            "message": faux_assistant_message("ORIGINAL_TEXT"),
        },
        {
            "type": "context_edit",
            "id": "e2",
            "parentId": "ok",
            "timestamp": ts(5),
            "targetId": "ok",
            "replacement": {"content": "REPLACED_TEXT"},
        },
    ]

    serialized = repr(prepare_branch_entries(branch).messages)

    assert "Investigate login" in serialized
    assert "FAILED_ATTEMPT" not in serialized
    assert "ORIGINAL_TEXT" not in serialized
    assert "REPLACED_TEXT" in serialized
