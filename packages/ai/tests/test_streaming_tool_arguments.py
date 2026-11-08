"""Mirror of pi's streaming-tool-arguments.test.ts (#9265: every provider
re-parsed the accumulated tool-call arguments on each delta).

Where it differs:

- The `pi-messages` path is not mirrored: that API is not ported.
- pi also checks that no stream state (`partialJson`, `index`, ...) reaches a
  partial block. Content blocks here are slotted dataclasses whose fields are
  fixed, and adapters keep stream state in their own maps, so that check
  could never fail and is not mirrored.
- "allocates linearly in the argument text" measures V8 heap allocation;
  here it measures the wall time of the whole stream, best of two runs, for
  arguments ten times longer than the other's.
- "releases the stream, its partial messages and argument snapshots once the
  caller drops them" is not mirrored: dicts cannot be weakly referenced, and
  CPython frees them by reference count.
"""

import json
import math
import time
from collections.abc import AsyncIterator, Callable

import pytest

from pidrei_ai.api.anthropic_messages import stream as stream_anthropic
from pidrei_ai.api.mistral_conversations import stream as stream_mistral
from pidrei_ai.api.openai_completions import stream as stream_openai_completions
from pidrei_ai.api.openai_responses_shared import process_responses_stream
from pidrei_ai.builders import AssistantMessageBuilder, UsageBuilder
from pidrei_ai.providers.faux import create_faux_core, faux_assistant_message, faux_tool_call
from pidrei_ai.types import (
    AssistantMessage,
    Context,
    DoneEvent,
    Model,
    ModelCost,
    StartEvent,
    StreamOptions,
    ToolCall,
    UserMessage,
)
from pidrei_ai.utils.event_stream import AssistantMessageEventStream
from pidrei_ai.utils.transcript import normalize_context
from pidrei_http import http


CONTENT = "".join(f'const line{line} = "value {line}"; // \\ done\n' for line in range(20_000))


def write_call(size: int) -> tuple[dict, str]:
    """`write` arguments with `size` characters of file content, and their JSON text."""
    args = {"path": "/src/file.ts", "content": CONTENT[:size]}
    return args, json.dumps(args)


def deltas_of(text: str, size: int) -> list[str]:
    return [text[index : index + size] for index in range(0, len(text), size)]


def sse_fetch(body: Callable[[], str]):
    """A fetch whose response streams the server-sent events `body` renders."""

    async def fetch(request, *, env=None):
        return http.Response(
            200, headers={"content-type": "text/event-stream"}, content=body().encode(), request=request
        )

    return fetch


def sse(data: object, event: str | None = None) -> str:
    return f"{f'event: {event}\n' if event else ''}data: {json.dumps(data)}\n\n"


def make_model(model_id: str, api: str, provider: str, base_url: str) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api=api,
        provider=provider,
        base_url=base_url,
        reasoning=False,
        input=["text"],
        cost=ModelCost(),
        context_window=200_000,
        max_tokens=64_000,
    )


CONTEXT = normalize_context(Context(messages=[UserMessage(content="write", timestamp=0)]))


def anthropic_events(deltas: list[str]) -> list[str]:
    return [
        sse(
            {
                "type": "message_start",
                "message": {"id": "msg_1", "model": "claude-test", "usage": {"input_tokens": 1, "output_tokens": 1}},
            },
            "message_start",
        ),
        sse(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_1", "name": "write", "input": {}},
            },
            "content_block_start",
        ),
        *(
            sse(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": delta},
                },
                "content_block_delta",
            )
            for delta in deltas
        ),
        sse({"type": "content_block_stop", "index": 0}, "content_block_stop"),
        sse(
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 10}},
            "message_delta",
        ),
        sse({"type": "message_stop"}, "message_stop"),
    ]


def chat_events(deltas: list[str], with_id_every_chunk: bool) -> list[str]:
    base = {"id": "chat_1", "object": "chat.completion.chunk", "created": 1, "model": "test"}

    def call(fields: dict) -> dict:
        return {**base, "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, **fields}]}}]}

    return [
        sse(call({"id": "call_1", "type": "function", "function": {"name": "write", "arguments": ""}})),
        *(
            sse(
                call(
                    {"id": "call_1", "function": {"name": "write", "arguments": delta}}
                    if with_id_every_chunk
                    else {"function": {"arguments": delta}}
                )
            )
            for delta in deltas
        ),
        sse(
            {
                **base,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ),
        "data: [DONE]\n\n",
    ]


def responses_events(deltas: list[str], text: str) -> list[dict]:
    item = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "write",
        "arguments": "",
        "status": "in_progress",
    }
    return [
        {"type": "response.output_item.added", "output_index": 0, "sequence_number": 0, "item": item},
        *(
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "item_id": "fc_1",
                "sequence_number": 0,
                "delta": delta,
            }
            for delta in deltas
        ),
        {
            "type": "response.function_call_arguments.done",
            "output_index": 0,
            "item_id": "fc_1",
            "sequence_number": 0,
            "arguments": text,
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "sequence_number": 0,
            "item": {**item, "arguments": text, "status": "completed"},
        },
        {
            "type": "response.completed",
            "sequence_number": 0,
            "response": {
                "id": "resp_1",
                "status": "completed",
                "output": [],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
        },
    ]


def stream_anthropic_path(deltas: list[str], _args: dict, _text: str) -> AssistantMessageEventStream:
    return stream_anthropic(
        make_model("claude-test", "anthropic-messages", "anthropic", "https://api.anthropic.com"),
        CONTEXT,
        StreamOptions(api_key="sk-ant-test", fetch=sse_fetch(lambda: "".join(anthropic_events(deltas)))),
    )


def stream_openai_completions_path(deltas: list[str], _args: dict, _text: str) -> AssistantMessageEventStream:
    return stream_openai_completions(
        make_model("gpt-test", "openai-completions", "openai", "https://api.openai.com/v1"),
        CONTEXT,
        StreamOptions(api_key="sk-test", fetch=sse_fetch(lambda: "".join(chat_events(deltas, False)))),
    )


def stream_openai_responses_path(deltas: list[str], _args: dict, text: str) -> AssistantMessageEventStream:
    output = AssistantMessageBuilder(
        content=[],
        api="openai-responses",
        provider="openai",
        model="gpt-test",
        usage=UsageBuilder(),
        stop_reason="pending",
        timestamp=0,
    )
    events = AssistantMessageEventStream()
    events.partial = output
    model = make_model("gpt-test", "openai-responses", "openai", "https://api.openai.com/v1")

    async def provider_events() -> AsyncIterator[dict]:
        for event in responses_events(deltas, text):
            yield event

    async def produce() -> None:
        events.push(StartEvent(partial=output))
        await process_responses_stream(provider_events(), output, events, model)
        events.push(DoneEvent(reason="toolUse", message=output))
        events.end()

    events.spawn_producer(produce())
    return events


def stream_mistral_path(deltas: list[str], _args: dict, _text: str) -> AssistantMessageEventStream:
    return stream_mistral(
        make_model("devstral", "mistral-conversations", "mistral", "https://api.mistral.ai"),
        CONTEXT,
        StreamOptions(api_key="test", fetch=sse_fetch(lambda: "".join(chat_events(deltas, True)))),
    )


def stream_faux_path(_deltas: list[str], args: dict, _text: str) -> AssistantMessageEventStream:
    faux = create_faux_core(token_size_min=8, token_size_max=8)
    faux.set_responses([faux_assistant_message([faux_tool_call("write", args, id="call_1")], stop_reason="toolUse")])
    return faux.stream(faux.get_model(), CONTEXT, StreamOptions())


PATHS = {
    "anthropic-messages": stream_anthropic_path,
    "openai-completions": stream_openai_completions_path,
    "openai-responses": stream_openai_responses_path,
    "mistral-conversations": stream_mistral_path,
    "faux": stream_faux_path,
}


async def consume(events: AssistantMessageEventStream, on_event=None) -> AssistantMessage:
    async for event in events:
        if on_event is not None:
            on_event(event)
    return await events.result()


@pytest.mark.tonio
@pytest.mark.parametrize("path", PATHS)
async def test_shows_growing_partial_arguments_and_finishes_with_the_exact_arguments(path):
    args, text = write_call(5_000)
    shown = -1
    partials = 0

    def on_event(event) -> None:
        nonlocal shown, partials
        if event.type != "toolcall_delta":
            return
        call = event.partial.content[event.content_index]
        shown_content = call.arguments.get("content") if call.type == "toolCall" else None
        if not isinstance(shown_content, str):
            return
        partials += 1
        assert args["content"].startswith(shown_content)
        assert len(shown_content) >= shown
        shown = len(shown_content)

    message = await consume(PATHS[path](deltas_of(text, 24), args, text), on_event)
    assert message.stop_reason == "toolUse", message.error_message
    assert partials > 100
    assert len(message.content) == 1
    call = message.content[0]
    assert isinstance(call, ToolCall)
    assert (call.name, call.arguments, call.arguments_error) == ("write", args, None)


async def _wall_seconds(path: str, size: int) -> float:
    best = math.inf
    for _ in range(2):
        args, text = write_call(size)
        deltas = deltas_of(text, 32)
        started = time.perf_counter()
        message = await consume(PATHS[path](deltas, args, text))
        best = min(best, time.perf_counter() - started)
        assert message.content[0].arguments == args
    return best


@pytest.mark.tonio
@pytest.mark.parametrize("path", PATHS)
async def test_takes_time_linear_in_the_argument_text(path):
    small = await _wall_seconds(path, 25_000)
    large = await _wall_seconds(path, 250_000)
    # Ten times the text: linear growth is about 10x, re-parsing the accumulated text about 100x.
    assert large / small < 25


@pytest.mark.tonio
async def test_keeps_the_partial_arguments_of_a_stream_cut_off_mid_call():
    _, text = write_call(2_000)
    cut = deltas_of(text, 16)[:20]
    # Drop the block stop and the message end, as a dropped connection does.
    events = anthropic_events(cut)[:-3]
    message = await consume(
        stream_anthropic(
            make_model("claude-test", "anthropic-messages", "anthropic", "https://api.anthropic.com"),
            CONTEXT,
            StreamOptions(api_key="sk-ant-test", fetch=sse_fetch(lambda: "".join(events))),
        )
    )
    assert message.stop_reason == "error"
    call = message.content[0]
    assert isinstance(call, ToolCall)
    assert call.arguments["path"] == "/src/file.ts"
