"""Tests for the openai-responses adapter.

Mirrors pi's unit suites (message-id, foreign-toolcall-id, partial-json
cleanup, terminal-event semantics, empty tool results, tool-result images —
catalog models replaced by hand-built ones where pi reads deferred providers'
catalogs) plus adapter-level stream/params coverage.
"""

import json
import time

import pytest

from pidrei_ai.api.openai_responses import (
    OpenAIResponsesOptions,
    build_params,
    stream as stream_responses,
)
from pidrei_ai.api.openai_responses_shared import (
    convert_responses_messages,
    process_responses_stream,
)
from pidrei_ai.types import (
    AssistantMessage,
    Context,
    Model,
    ModelCost,
    OpenAIResponsesCompat,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from pidrei_ai.utils.event_stream import AssistantMessageEventStream
from pidrei_ai.utils.hash import short_hash


ALLOWED = {"openai", "openai-codex", "opencode"}

COPILOT_RAW_TOOL_CALL_ID = (
    "call_4VnzVawQXPB9MgYib7CiQFEY|I9b95oN1wD/cHXKTw3PpRkL6KkCtzTJhUxMouMWYwHeTo2j3htzfSk7YPx2vifiIM4g3A8XXyOj8q4"
    "Bt6SLUG7gqY1E3ELkrkVQNHglRfUmWj84lqxJY+Puieb3VKyX0FB+83TUzn91cDMF/4gzt990IzqVrc+nIb9RRscRD070Du16q1glydVjWR0S"
    "BJsE6TbY/esOjFpqplogQqrajm1eI++f3eLi73R6q7hVusY0QbeFySVxABCjhN0lXB04caBe1rzHjYzul6MAXj7uq+0r17VLq+yrtyYhN12wk"
    "mFqHeqTyEei6EFPbMy24Nc+IbJlkP0OCg02W+gOnyBFcbi2ctvJFSOhSjt1CqBdqCnnhwUqXjbWiT0wh3DmLScRgTHmGkaI+oAcQQjfic65nx"
    "j+TnEkReA=="
)


def make_model(provider="openai", **overrides) -> Model:
    defaults: dict = {
        "id": "gpt-5-mini",
        "name": "GPT-5 Mini",
        "api": "openai-responses",
        "provider": provider,
        "base_url": "https://api.openai.com/v1",
        "reasoning": True,
        "input": ["text"],
        "cost": ModelCost(),
        "context_window": 400_000,
        "max_tokens": 128_000,
    }
    defaults.update(overrides)
    return Model(**defaults)


def now_ms() -> int:
    return int(time.time() * 1000)


def make_output(model: Model) -> AssistantMessage:
    return AssistantMessage(
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="pending",
        timestamp=now_ms(),
    )


# --- message conversion mirrors ----------------------------------------------


def test_generates_unique_fallback_message_ids_for_multiple_text_blocks():
    model = make_model(provider="openai-codex", id="gpt-5.5")
    assistant = AssistantMessage(
        content=[ThinkingContent(thinking="private reasoning"), TextContent(text="visible answer")],
        api="anthropic-messages",
        provider="anthropic",
        model="claude-opus-4-8",
        usage=Usage(),
        stop_reason="stop",
        timestamp=now_ms() - 1000,
    )
    context = Context(
        system_prompt="You are concise.",
        messages=[UserMessage(content="hello", timestamp=now_ms() - 2000), assistant],
    )

    converted = convert_responses_messages(model, context, ALLOWED)
    message_ids = [
        item["id"] for item in converted if item.get("type") == "message" and isinstance(item.get("id"), str)
    ]

    assert message_ids == ["msg_pi_1", "msg_pi_1_1"]
    assert len(set(message_ids)) == len(message_ids)


def test_hashes_foreign_copilot_tool_item_ids_into_bounded_fc_shape():
    import re

    model = make_model(provider="openai-codex", id="gpt-5.5")
    assistant = AssistantMessage(
        content=[ToolCall(id=COPILOT_RAW_TOOL_CALL_ID, name="edit", arguments={"path": "src/styles/app.css"})],
        api="openai-responses",
        provider="github-copilot",
        model="gpt-5.5",
        usage=Usage(),
        stop_reason="toolUse",
        timestamp=now_ms() - 2000,
    )
    tool_result = ToolResultMessage(
        tool_call_id=COPILOT_RAW_TOOL_CALL_ID,
        tool_name="edit",
        content=[TextContent(text="ok")],
        is_error=False,
        timestamp=now_ms() - 1000,
    )
    context = Context(
        system_prompt="You are concise.",
        messages=[UserMessage(content="Use the tool.", timestamp=now_ms() - 3000), assistant, tool_result],
    )

    converted = convert_responses_messages(model, context, ALLOWED)
    function_call = next(item for item in converted if item.get("type") == "function_call")

    expected_item_id = f"fc_{short_hash(COPILOT_RAW_TOOL_CALL_ID.split('|')[1])}"
    assert function_call["id"] == expected_item_id
    assert len(function_call["id"]) <= 64
    assert re.match(r"^fc_[A-Za-z0-9]+$", function_call["id"])


def test_empty_tool_result_gets_placeholder_output():
    model = make_model()
    context = Context(
        messages=[
            UserMessage(content="go", timestamp=1),
            ToolResultMessage(tool_call_id="call_1", tool_name="tool", content=[], is_error=False, timestamp=2),
        ]
    )
    converted = convert_responses_messages(model, context, ALLOWED)
    output_item = next(item for item in converted if item.get("type") == "function_call_output")
    assert output_item["output"] == "(no tool output)"


def test_tool_result_images_for_vision_and_non_vision_models():
    from pidrei_ai.types import ImageContent

    image = ImageContent(data="abcd", mime_type="image/png")
    result = ToolResultMessage(
        tool_call_id="call_1",
        tool_name="tool",
        content=[TextContent(text="see this"), image],
        is_error=False,
        timestamp=2,
    )
    context = Context(messages=[UserMessage(content="go", timestamp=1), result])

    vision = convert_responses_messages(make_model(input=["text", "image"]), context, ALLOWED)
    vision_output = next(item for item in vision if item.get("type") == "function_call_output")["output"]
    assert vision_output[0] == {"type": "input_text", "text": "see this"}
    assert vision_output[1]["type"] == "input_image"
    assert vision_output[1]["image_url"] == "data:image/png;base64,abcd"

    # Non-vision models: transform_messages downgrades the image to pi's
    # placeholder text before conversion ever sees it.
    text_only = convert_responses_messages(make_model(input=["text"]), context, ALLOWED)
    text_output = next(item for item in text_only if item.get("type") == "function_call_output")["output"]
    assert text_output == "see this\n(tool image omitted: model does not support images)"


# --- stream processor mirrors -------------------------------------------------


async def events_from(items: list[dict]):
    for item in items:
        yield item


class RecordingStream(AssistantMessageEventStream):
    """pi spies on `stream.push`; iterating would hang since the processor
    never ends the stream (the adapter's caller does)."""

    def __init__(self):
        super().__init__()
        self.pushed = []

    def push(self, event):
        self.pushed.append(event)
        super().push(event)


class StopReasonRecordingStream(RecordingStream):
    """Partials share the mutated output object, so the stop reason must be
    sampled at push time — pi's spy reads `event.partial.stopReason` the same
    way."""

    def __init__(self):
        super().__init__()
        self.observed_stop_reasons = []

    def push(self, event):
        partial = getattr(event, "partial", None)
        if partial is not None:
            self.observed_stop_reasons.append(partial.stop_reason)
        super().push(event)


def function_call_events(arguments_json: str) -> list[dict]:
    return [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "function_call", "id": "fc_test", "call_id": "call_test", "name": "edit", "arguments": ""},
        },
        {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": '{"path":"README.md"'},
        {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": ',"content":"updated"}'},
        {"type": "response.function_call_arguments.done", "output_index": 0, "arguments": arguments_json},
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "fc_test",
                "call_id": "call_test",
                "name": "edit",
                "arguments": arguments_json,
            },
        },
        {"type": "response.completed", "response": {"id": "resp_test", "status": "completed"}},
    ]


@pytest.mark.tonio
async def test_tool_call_blocks_are_clean_at_output_item_done():
    model = make_model()
    output = make_output(model)
    stream = RecordingStream()
    arguments_json = '{"path":"README.md","content":"updated"}'

    await process_responses_stream(events_from(function_call_events(arguments_json)), output, stream, model)

    assert len(output.content) == 1
    persisted = output.content[0]
    assert persisted.type == "toolCall"
    assert persisted.arguments == {"path": "README.md", "content": "updated"}
    assert isinstance(persisted, ToolCall)  # plain dataclass, no scratch attached

    tool_call_end = next(event for event in stream.pushed if event.type == "toolcall_end")
    assert tool_call_end.tool_call is persisted


@pytest.mark.tonio
async def test_stream_without_terminal_event_raises():
    model = make_model()
    output = make_output(model)
    stream = AssistantMessageEventStream()

    with pytest.raises(RuntimeError, match="OpenAI Responses stream ended before a terminal response event"):
        await process_responses_stream(
            events_from([{"type": "response.created", "response": {"id": "resp_1"}}]), output, stream, model
        )


def phased_message_events(phases: tuple[str, str], terminal_status: str = "completed") -> list[dict]:
    events = [
        {
            "type": "response.output_item.added",
            "sequence_number": 0,
            "output_index": 0,
            "item": {
                "type": "message",
                "id": "msg_phase",
                "role": "assistant",
                "status": "in_progress",
                "content": [],
                "phase": phases[0],
            },
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 1,
            "output_index": 0,
            "item": {
                "type": "message",
                "id": "msg_phase",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "answer", "annotations": []}],
                "phase": phases[1],
            },
        },
        {
            "type": f"response.{terminal_status}",
            "sequence_number": 2,
            "response": (
                {"id": "resp_phase", "status": terminal_status, "incomplete_details": {"reason": "max_output_tokens"}}
                if terminal_status == "incomplete"
                else {"id": "resp_phase", "status": terminal_status}
            ),
        },
    ]
    return events


def incomplete_events(reason: str = "max_output_tokens") -> list[dict]:
    return [
        {
            "type": "response.incomplete",
            "sequence_number": 0,
            "response": {
                "id": "resp_incomplete",
                "status": "incomplete",
                "incomplete_details": {"reason": reason},
            },
        }
    ]


@pytest.mark.tonio
@pytest.mark.parametrize(
    ("phases", "expected"),
    [
        (("commentary", "commentary"), ["pending", "pending"]),
        (("final_answer", "final_answer"), ["stop", "stop"]),
        (("commentary", "final_answer"), ["pending", "stop"]),
    ],
)
async def test_tracks_message_phases(phases, expected):
    model = make_model()
    output = make_output(model)
    stream = StopReasonRecordingStream()

    await process_responses_stream(events_from(phased_message_events(phases)), output, stream, model)

    assert stream.observed_stop_reasons == expected
    assert output.stop_reason == "stop"
    assert output.raw_stop_reason == "completed"


@pytest.mark.tonio
async def test_replaces_a_provisional_final_answer_stop_with_an_incomplete_terminal_reason():
    model = make_model()
    output = make_output(model)
    stream = StopReasonRecordingStream()

    await process_responses_stream(
        events_from(phased_message_events(("final_answer", "final_answer"), "incomplete")), output, stream, model
    )

    assert stream.observed_stop_reasons == ["stop", "stop"]
    assert output.stop_reason == "length"
    assert output.raw_stop_reason == "incomplete.max_output_tokens"


@pytest.mark.tonio
async def test_finalizes_content_filtered_incomplete_responses_as_non_retryable_errors():
    model = make_model()
    output = make_output(model)
    stream = AssistantMessageEventStream()

    await process_responses_stream(events_from(incomplete_events("content_filter")), output, stream, model)

    assert output.stop_reason == "error"
    assert output.raw_stop_reason == "incomplete.content_filter"
    assert output.error_message == "Response incomplete: content_filter"


@pytest.mark.tonio
async def test_preserves_unknown_provider_incomplete_reasons_as_non_retryable_errors():
    model = make_model()
    output = make_output(model)
    stream = AssistantMessageEventStream()

    await process_responses_stream(events_from(incomplete_events("max_time_limit")), output, stream, model)

    assert output.stop_reason == "error"
    assert output.raw_stop_reason == "incomplete.max_time_limit"
    assert output.error_message == "Response incomplete: max_time_limit"


@pytest.mark.tonio
async def test_response_failed_raises_with_error_details():
    model = make_model()
    output = make_output(model)
    stream = AssistantMessageEventStream()
    events = [
        {
            "type": "response.failed",
            "response": {"status": "failed", "error": {"code": "server_error", "message": "exploded"}},
        },
    ]

    with pytest.raises(RuntimeError, match="server_error: exploded"):
        await process_responses_stream(events_from(events), output, stream, model)

    assert output.raw_stop_reason == "failed"


@pytest.mark.tonio
async def test_error_events_raise_with_code_and_message():
    model = make_model()
    output = make_output(model)
    stream = AssistantMessageEventStream()
    events = [{"type": "error", "code": "rate_limit", "message": "slow down"}]

    with pytest.raises(RuntimeError, match="Error Code rate_limit: slow down"):
        await process_responses_stream(events_from(events), output, stream, model)


@pytest.mark.tonio
async def test_reasoning_and_text_streaming_with_usage_and_tool_use_promotion():
    model = make_model()
    output = make_output(model)
    stream = AssistantMessageEventStream()
    reasoning_item = {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [{"type": "summary_text", "text": "thought hard"}],
        "encrypted_content": "opaque",
    }
    events = [
        {"type": "response.created", "response": {"id": "resp_1"}},
        {"type": "response.output_item.added", "output_index": 0, "item": {"type": "reasoning", "id": "rs_1"}},
        {"type": "response.reasoning_summary_text.delta", "output_index": 0, "delta": "thought hard"},
        {"type": "response.output_item.done", "output_index": 0, "item": reasoning_item},
        {"type": "response.output_item.added", "output_index": 1, "item": {"type": "message", "id": "msg_1"}},
        {"type": "response.output_text.delta", "output_index": 1, "delta": "hello"},
        {
            "type": "response.output_item.done",
            "output_index": 1,
            "item": {
                "type": "message",
                "id": "msg_1",
                "content": [{"type": "output_text", "text": "hello"}],
            },
        },
        *function_call_events('{"path":"x"}')[:5],
        {
            "type": "response.completed",
            "response": {
                "id": "resp_final",
                "status": "completed",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 30,
                    "total_tokens": 130,
                    "input_tokens_details": {"cached_tokens": 40},
                    "output_tokens_details": {"reasoning_tokens": 10},
                },
            },
        },
    ]

    await process_responses_stream(events_from(events), output, stream, model)

    thinking = output.content[0]
    assert thinking.type == "thinking"
    assert thinking.thinking == "thought hard"
    assert json.loads(thinking.thinking_signature) == reasoning_item

    text = output.content[1]
    assert text.type == "text"
    assert text.text == "hello"
    assert json.loads(text.text_signature) == {"v": 1, "id": "msg_1"}

    assert output.response_id == "resp_final"
    assert output.usage.input == 60  # 100 - 40 cached
    assert output.usage.cache_read == 40
    assert output.usage.reasoning == 10
    # toolCall present + status completed -> promoted to toolUse.
    assert output.stop_reason == "toolUse"


# --- adapter-level ------------------------------------------------------------


class FakeResponse:
    def __init__(self, body: bytes):
        self.status = 200
        self.headers = {"content-type": "text/event-stream"}
        self._body = body

    async def aiter_bytes(self):
        yield self._body


class FakeClient:
    def __init__(self, events: list[dict]):
        self._body = "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()
        self.requests: list[dict] = []
        self._response_factory = FakeResponse

    async def create(self, params, *, timeout_ms, cancel):
        self.requests.append(params)
        return self._response_factory(self._body)


@pytest.mark.tonio
async def test_stream_end_to_end_with_fake_client():
    events = [
        {"type": "response.created", "response": {"id": "resp_1"}},
        {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message", "id": "msg_1"}},
        {"type": "response.output_text.delta", "output_index": 0, "delta": "hi"},
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {"type": "message", "id": "msg_1", "content": [{"type": "output_text", "text": "hi"}]},
        },
        {"type": "response.completed", "response": {"id": "resp_1", "status": "completed"}},
    ]
    client = FakeClient(events)
    result = await stream_responses(
        make_model(),
        Context(messages=[UserMessage(content="q", timestamp=1)]),
        OpenAIResponsesOptions(api_key="k", client=client),
    ).result()

    assert result.stop_reason == "stop"
    assert result.content[0].text == "hi"
    assert len(client.requests) == 1


@pytest.mark.tonio
async def test_rejects_streams_that_end_before_a_terminal_response_event():
    # pi reads `partial.stopReason` off the start event mid-iteration and relies
    # on JS microtask interleaving to observe it before the adapter mutates the
    # shared output; here the body is gated so the start event is deterministically
    # observed while the stream is still open.
    import tonio.colored as tonio

    start_seen = tonio.Event()

    class GatedResponse(FakeResponse):
        async def aiter_bytes(self):
            yield self._body
            await start_seen.wait()

    client = FakeClient([{"type": "response.created", "response": {"id": "resp_1"}}])
    client._response_factory = GatedResponse
    stream = stream_responses(
        make_model(),
        Context(messages=[UserMessage(content="q", timestamp=1)]),
        OpenAIResponsesOptions(api_key="k", client=client),
    )

    initial_stop_reason = None
    events = []
    async for event in stream:
        if event.type == "start":
            initial_stop_reason = event.partial.stop_reason
            start_seen.set()
        events.append(event)

    result = await stream.result()
    assert initial_stop_reason == "pending"
    assert events[-1].type == "error"
    assert result.stop_reason == "error"
    assert result.error_message == "OpenAI Responses stream ended before a terminal response event"


def test_build_params_defaults_and_reasoning():
    model = make_model()
    params = build_params(
        model,
        Context(messages=[UserMessage(content="q", timestamp=1)]),
        OpenAIResponsesOptions(max_tokens=5, session_id="sess", reasoning_effort="high"),
    )

    assert params["store"] is False
    assert params["prompt_cache_key"] == "sess"
    assert params["max_output_tokens"] == 16  # clamped to the API minimum
    assert params["reasoning"] == {"effort": "high", "summary": "auto"}
    assert params["include"] == ["reasoning.encrypted_content"]


def test_build_params_reasoning_off_variants():
    default_off = build_params(
        make_model(), Context(messages=[UserMessage(content="q", timestamp=1)]), OpenAIResponsesOptions()
    )
    assert default_off["reasoning"] == {"effort": "none"}

    mapped_off = build_params(
        make_model(thinking_level_map={"off": "minimal"}),
        Context(messages=[UserMessage(content="q", timestamp=1)]),
        OpenAIResponsesOptions(),
    )
    assert mapped_off["reasoning"] == {"effort": "minimal"}

    null_off = build_params(
        make_model(thinking_level_map={"off": None}),
        Context(messages=[UserMessage(content="q", timestamp=1)]),
        OpenAIResponsesOptions(),
    )
    assert "reasoning" not in null_off


def test_build_params_explicit_prompt_cache_mode():
    model = make_model(compat=OpenAIResponsesCompat(supports_explicit_prompt_cache_mode=True))
    params = build_params(
        model,
        Context(messages=[UserMessage(content="q", timestamp=1)]),
        OpenAIResponsesOptions(cache_retention="none"),
    )
    assert params["prompt_cache_options"] == {"mode": "explicit"}
    assert "prompt_cache_key" not in params


def test_build_params_long_retention_and_xai_include():
    params = build_params(
        make_model(),
        Context(messages=[UserMessage(content="q", timestamp=1)]),
        OpenAIResponsesOptions(session_id="s", cache_retention="long"),
    )
    assert params["prompt_cache_retention"] == "24h"

    xai = build_params(
        make_model(provider="xai", base_url="https://api.x.ai/v1"),
        Context(messages=[UserMessage(content="q", timestamp=1)]),
        OpenAIResponsesOptions(),
    )
    assert xai["include"] == ["reasoning.encrypted_content"]
