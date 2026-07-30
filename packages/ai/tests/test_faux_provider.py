"""Mirror of pi's faux-provider.test.ts.

pi drives these through the deprecated compat api-registry (`registerFauxProvider`
plus global `stream`/`complete`); pidrei uses the explicit `faux_provider()`
handle — either called directly or registered on a `Models` collection. The
registry-specific "unregisters the provider" case has no pidrei equivalent
(deregistration is `Models.delete_provider`, covered by the registry tests).
"""

import json
import time

import pytest

from pidrei_ai.providers.faux import (
    FauxModelDefinition,
    faux_assistant_message,
    faux_provider,
    faux_text,
    faux_thinking,
    faux_tool_call,
)
from pidrei_ai.registry import create_models
from pidrei_ai.types import Context, StreamOptions, TextContent, ThinkingContent, Tool, UserMessage
from pidrei_ai.utils.cancel import CancelToken
from pidrei_ai.utils.estimate import _tool_json_shape


def now_ms() -> int:
    return int(time.time() * 1000)


def user_context(text: str = "hi") -> Context:
    return Context(messages=[UserMessage(content=text, timestamp=now_ms())])


async def collect_events(stream) -> list:
    return [event async for event in stream]


async def complete(faux, model, context, options=None):
    return await faux.provider.stream(model, context, options).result()


@pytest.mark.tonio
async def test_registers_a_custom_provider_and_estimates_usage():
    faux = faux_provider()
    models = create_models()
    models.set_provider(faux.provider)
    faux.set_responses([faux_assistant_message("hello world")])

    context = Context(
        system_prompt="Be concise.",
        messages=[UserMessage(content="hi there", timestamp=now_ms())],
    )

    # Through the Models collection: auth resolution + provider dispatch.
    response = await models.complete(faux.get_model(), context)
    assert response.content == [TextContent(text="hello world")]
    assert response.usage.input > 0
    assert response.usage.output > 0
    assert response.usage.total_tokens == response.usage.input + response.usage.output
    assert faux.state.call_count == 1


@pytest.mark.tonio
async def test_supports_helper_blocks_for_text_thinking_and_tool_calls():
    faux = faux_provider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_thinking("think"), faux_tool_call("echo", {"text": "hi"}), faux_text("done")],
                stop_reason="toolUse",
            )
        ]
    )

    response = await complete(faux, faux.get_model(), user_context())

    assert response.content[0] == ThinkingContent(thinking="think")
    assert response.content[1].type == "toolCall"
    assert response.content[1].name == "echo"
    assert response.content[1].arguments == {"text": "hi"}
    assert response.content[2] == TextContent(text="done")
    assert response.stop_reason == "toolUse"


@pytest.mark.tonio
async def test_supports_multiple_models_with_model_aware_factories():
    faux = faux_provider(
        models=[
            FauxModelDefinition(id="faux-fast", name="Faux Fast", reasoning=False),
            FauxModelDefinition(id="faux-thinker", name="Faux Thinker", reasoning=True),
        ]
    )

    async def factory(_context, _options, _state, model):
        return faux_assistant_message(f"{model.id}:{'true' if model.reasoning else 'false'}")

    faux.set_responses([factory, factory])

    assert [model.id for model in faux.models] == ["faux-fast", "faux-thinker"]
    assert faux.get_model() is faux.models[0]
    assert faux.get_model("faux-fast").reasoning is False
    assert faux.get_model("faux-thinker").reasoning is True

    fast = await complete(faux, faux.get_model("faux-fast"), user_context())
    thinker = await complete(faux, faux.get_model("faux-thinker"), user_context())

    assert fast.content == [TextContent(text="faux-fast:false")]
    assert thinker.content == [TextContent(text="faux-thinker:true")]


@pytest.mark.tonio
async def test_rewrites_api_provider_and_model_on_returned_messages():
    faux = faux_provider(api="faux:test", provider="faux-provider", models=[FauxModelDefinition(id="faux-model")])
    faux.set_responses([faux_assistant_message("hello")])

    response = await complete(faux, faux.get_model(), user_context())

    assert response.api == "faux:test"
    assert response.provider == "faux-provider"
    assert response.model == "faux-model"


@pytest.mark.tonio
async def test_consumes_queued_responses_in_order_and_errors_when_exhausted():
    faux = faux_provider()
    faux.set_responses([faux_assistant_message("first"), faux_assistant_message("second")])
    context = user_context()

    first = await complete(faux, faux.get_model(), context)
    second = await complete(faux, faux.get_model(), context)
    exhausted = await complete(faux, faux.get_model(), context)

    assert first.content == [TextContent(text="first")]
    assert second.content == [TextContent(text="second")]
    assert exhausted.stop_reason == "error"
    assert exhausted.error_message == "No more faux responses queued"
    assert faux.get_pending_response_count() == 0
    assert faux.state.call_count == 3


@pytest.mark.tonio
async def test_can_replace_and_append_queued_responses():
    faux = faux_provider()
    faux.set_responses([faux_assistant_message("first")])
    context = user_context()

    assert (await complete(faux, faux.get_model(), context)).content == [TextContent(text="first")]
    assert faux.get_pending_response_count() == 0

    faux.set_responses([faux_assistant_message("second")])
    assert faux.get_pending_response_count() == 1
    assert (await complete(faux, faux.get_model(), context)).content == [TextContent(text="second")]

    faux.append_responses([faux_assistant_message("third"), faux_assistant_message("fourth")])
    assert faux.get_pending_response_count() == 2
    assert (await complete(faux, faux.get_model(), context)).content == [TextContent(text="third")]
    assert (await complete(faux, faux.get_model(), context)).content == [TextContent(text="fourth")]
    assert faux.get_pending_response_count() == 0


@pytest.mark.tonio
async def test_supports_async_response_factories():
    faux = faux_provider()

    async def factory(context, _options, state, _model):
        return faux_assistant_message(f"{len(context.messages)}:{state.call_count}")

    faux.set_responses([factory])
    response = await complete(faux, faux.get_model(), user_context())

    assert response.content == [TextContent(text="1:1")]


@pytest.mark.tonio
async def test_emits_an_error_when_a_response_factory_throws():
    faux = faux_provider()

    async def factory(_context, _options, _state, _model):
        raise RuntimeError("boom")

    faux.set_responses([factory])

    events = await collect_events(faux.provider.stream(faux.get_model(), user_context()))

    assert len(events) == 1
    assert events[0].type == "error"
    assert events[0].error.stop_reason == "error"
    assert events[0].error.error_message == "boom"


@pytest.mark.tonio
async def test_rejects_a_queued_response_without_a_terminal_stop_reason():
    faux = faux_provider()
    faux.set_responses([faux_assistant_message("partial", stop_reason="pending")])

    events = await collect_events(faux.provider.stream(faux.get_model(), user_context()))

    assert all(event.type != "done" for event in events)
    terminal = events[-1]
    assert terminal.type == "error"
    assert terminal.error.stop_reason == "error"
    assert terminal.error.error_message == "Faux response ended without a stop reason"


@pytest.mark.tonio
async def test_estimates_prompt_and_output_tokens_from_serialized_context():
    faux = faux_provider()
    faux.set_responses([faux_assistant_message("done")])

    tool = Tool(
        name="echo",
        description="Echo back text",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    )
    from pidrei_ai.types import ImageContent, ToolResultMessage

    context = Context(
        system_prompt="sys",
        messages=[
            UserMessage(
                content=[TextContent(text="hello"), ImageContent(data="abcd", mime_type="image/png")], timestamp=1
            ),
            faux_assistant_message("prior"),
            ToolResultMessage(
                tool_call_id="tool-1",
                tool_name="echo",
                content=[TextContent(text="tool out")],
                is_error=False,
                timestamp=2,
            ),
        ],
        tools=[tool],
    )

    response = await complete(faux, faux.get_model(), context)
    tools_json = json.dumps([_tool_json_shape(tool)], separators=(",", ":"), ensure_ascii=False)
    prompt_text = "\n\n".join(
        [
            "system:sys",
            "user:hello\n[image:image/png:4]",
            "assistant:prior",
            "toolResult:echo\ntool out",
            f"tools:{tools_json}",
        ]
    )
    expected_prompt_tokens = -(-len(prompt_text) // 4)
    expected_output_tokens = -(-len("done") // 4)

    assert response.usage.input == expected_prompt_tokens
    assert response.usage.output == expected_output_tokens
    assert response.usage.cache_read == 0
    assert response.usage.cache_write == 0
    assert response.usage.total_tokens == expected_prompt_tokens + expected_output_tokens


@pytest.mark.tonio
async def test_does_not_share_cache_across_sessions_or_requests_without_session_id():
    faux = faux_provider()
    faux.set_responses(
        [faux_assistant_message("first"), faux_assistant_message("second"), faux_assistant_message("third")]
    )
    context = Context(messages=[UserMessage(content="hello", timestamp=now_ms())])

    first = await complete(
        faux, faux.get_model(), context, StreamOptions(session_id="session-1", cache_retention="short")
    )
    assert first.usage.cache_write > 0
    context.messages.append(first)
    context.messages.append(UserMessage(content="follow up", timestamp=now_ms() + 1))

    second = await complete(
        faux, faux.get_model(), context, StreamOptions(session_id="session-2", cache_retention="short")
    )
    assert second.usage.cache_read == 0
    assert second.usage.cache_write > 0

    third = await complete(faux, faux.get_model(), context)
    assert third.usage.cache_read == 0
    assert third.usage.cache_write == 0


@pytest.mark.tonio
async def test_simulates_prompt_caching_per_session_id():
    faux = faux_provider()
    faux.set_responses([faux_assistant_message("first"), faux_assistant_message("second")])
    context = Context(
        system_prompt="Be concise.",
        messages=[UserMessage(content="hello", timestamp=now_ms())],
    )

    first = await complete(
        faux, faux.get_model(), context, StreamOptions(session_id="session-1", cache_retention="short")
    )
    assert first.usage.cache_read == 0
    assert first.usage.cache_write > 0

    context.messages.append(first)
    context.messages.append(UserMessage(content="follow up", timestamp=now_ms() + 1))

    second = await complete(
        faux, faux.get_model(), context, StreamOptions(session_id="session-1", cache_retention="short")
    )
    assert second.usage.cache_read > 0
    assert second.usage.input + second.usage.cache_read > second.usage.input


@pytest.mark.tonio
async def test_does_not_simulate_caching_when_cache_retention_is_none():
    faux = faux_provider()
    faux.set_responses([faux_assistant_message("first"), faux_assistant_message("second")])
    context = Context(messages=[UserMessage(content="hello", timestamp=now_ms())])

    await complete(faux, faux.get_model(), context, StreamOptions(session_id="session-1", cache_retention="none"))
    context.messages.append(faux_assistant_message("first"))
    context.messages.append(UserMessage(content="follow up", timestamp=now_ms() + 1))
    second = await complete(
        faux, faux.get_model(), context, StreamOptions(session_id="session-1", cache_retention="none")
    )
    assert second.usage.cache_read == 0
    assert second.usage.cache_write == 0


@pytest.mark.tonio
async def test_streams_thinking_text_and_partial_tool_call_deltas():
    faux = faux_provider()
    faux.set_responses(
        [
            faux_assistant_message(
                [
                    faux_thinking("thinking text"),
                    faux_text("answer text"),
                    faux_tool_call("echo", {"text": "hi", "count": 12}, id="tool-1"),
                ],
                stop_reason="toolUse",
            )
        ]
    )

    events: list[str] = []
    tool_call_deltas: list[str] = []
    stream = faux.provider.stream(faux.get_model(), user_context())
    async for event in stream:
        events.append(event.type)
        if event.type == "toolcall_delta":
            tool_call_deltas.append(event.delta)

    for expected in (
        "thinking_start",
        "thinking_delta",
        "text_start",
        "text_delta",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_end",
    ):
        assert expected in events
    assert len(tool_call_deltas) > 1
    assert json.loads("".join(tool_call_deltas)) == {"text": "hi", "count": 12}


@pytest.mark.tonio
async def test_streams_an_exact_event_order_for_fixed_size_chunks():
    faux = faux_provider(token_size_min=1, token_size_max=1)
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_thinking("go"), faux_text("ok"), faux_tool_call("echo", {}, id="tool-1")],
                stop_reason="toolUse",
            )
        ]
    )

    events = await collect_events(faux.provider.stream(faux.get_model(), user_context()))

    assert events[0].type == "start"
    assert events[0].partial.stop_reason == "pending"
    assert [event.type for event in events] == [
        "start",
        "thinking_start",
        "thinking_delta",
        "thinking_end",
        "text_start",
        "text_delta",
        "text_end",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_end",
        "done",
    ]


@pytest.mark.tonio
async def test_streams_multiple_tool_calls_in_one_message():
    faux = faux_provider()
    faux.set_responses(
        [
            faux_assistant_message(
                [
                    faux_tool_call("echo", {"text": "one"}, id="tool-1"),
                    faux_tool_call("echo", {"text": "two"}, id="tool-2"),
                ],
                stop_reason="toolUse",
            )
        ]
    )

    events = await collect_events(faux.provider.stream(faux.get_model(), user_context()))

    assert len([event for event in events if event.type == "toolcall_start"]) == 2
    assert len([event for event in events if event.type == "toolcall_end"]) == 2


@pytest.mark.tonio
async def test_streams_an_explicit_assistant_error_message_as_a_terminal_error():
    faux = faux_provider(token_size_min=2, token_size_max=2)
    faux.set_responses([faux_assistant_message("partial", stop_reason="error", error_message="upstream failed")])

    events = await collect_events(faux.provider.stream(faux.get_model(), user_context()))

    assert [event.type for event in events] == ["start", "text_start", "text_delta", "text_end", "error"]
    terminal = events[-1]
    assert terminal.reason == "error"
    assert terminal.error.stop_reason == "error"
    assert terminal.error.error_message == "upstream failed"


@pytest.mark.tonio
async def test_streams_an_explicit_assistant_aborted_message_as_a_terminal_error():
    faux = faux_provider(token_size_min=2, token_size_max=2)
    faux.set_responses([faux_assistant_message("partial", stop_reason="aborted", error_message="Request was aborted")])

    events = await collect_events(faux.provider.stream(faux.get_model(), user_context()))

    assert [event.type for event in events] == ["start", "text_start", "text_delta", "text_end", "error"]
    terminal = events[-1]
    assert terminal.reason == "aborted"
    assert terminal.error.stop_reason == "aborted"
    assert terminal.error.error_message == "Request was aborted"


@pytest.mark.tonio
async def test_supports_aborting_before_the_first_chunk():
    faux = faux_provider(tokens_per_second=50, token_size_min=3, token_size_max=3)
    faux.set_responses([faux_assistant_message("abcdefghijklmnopqrstuvwxyz")])

    cancel = CancelToken()
    cancel.cancel()
    events = await collect_events(faux.provider.stream(faux.get_model(), user_context(), StreamOptions(cancel=cancel)))

    assert len(events) == 1
    assert events[0].type == "error"
    assert events[0].reason == "aborted"
    assert events[0].error.stop_reason == "aborted"


@pytest.mark.tonio
async def test_supports_aborting_mid_text_stream_when_paced():
    faux = faux_provider(tokens_per_second=100, token_size_min=3, token_size_max=3)
    faux.set_responses([faux_assistant_message("abcdefghijklmnopqrstuvwxyz")])

    cancel = CancelToken()
    events: list[str] = []
    text_delta_count = 0
    stream = faux.provider.stream(faux.get_model(), user_context(), StreamOptions(cancel=cancel))
    async for event in stream:
        events.append(event.type)
        if event.type == "text_delta":
            text_delta_count += 1
            cancel.cancel()

    assert text_delta_count == 1
    assert "text_start" in events
    assert "text_delta" in events
    assert "error" in events
    assert "text_end" not in events


@pytest.mark.tonio
async def test_supports_aborting_mid_thinking_stream_when_paced():
    faux = faux_provider(tokens_per_second=100, token_size_min=3, token_size_max=3)
    faux.set_responses([faux_assistant_message([faux_thinking("abcdefghijklmnopqrstuvwxyz")])])

    cancel = CancelToken()
    events: list[str] = []
    thinking_delta_count = 0
    stream = faux.provider.stream(faux.get_model(), user_context(), StreamOptions(cancel=cancel))
    async for event in stream:
        events.append(event.type)
        if event.type == "thinking_delta":
            thinking_delta_count += 1
            cancel.cancel()

    assert thinking_delta_count == 1
    assert "thinking_start" in events
    assert "thinking_delta" in events
    assert "error" in events
    assert "thinking_end" not in events


@pytest.mark.tonio
async def test_supports_aborting_mid_toolcall_stream_when_paced():
    faux = faux_provider(tokens_per_second=100, token_size_min=3, token_size_max=3)
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("echo", {"text": "abcdefghijklmnopqrstuvwxyz", "count": 123456789}, id="tool-1")],
                stop_reason="toolUse",
            )
        ]
    )

    cancel = CancelToken()
    events: list[str] = []
    tool_call_delta_count = 0
    stream = faux.provider.stream(faux.get_model(), user_context(), StreamOptions(cancel=cancel))
    async for event in stream:
        events.append(event.type)
        if event.type == "toolcall_delta":
            tool_call_delta_count += 1
            cancel.cancel()

    assert tool_call_delta_count == 1
    assert "toolcall_start" in events
    assert "toolcall_delta" in events
    assert "error" in events
    assert "toolcall_end" not in events
