"""Mirror of pi agent/test/agent-loop.test.ts."""

import time
from dataclasses import dataclass, replace

import pytest
import tonio.colored as tonio

from pidrei_agent.agent_loop import agent_loop, agent_loop_continue, run_agent_loop
from pidrei_agent.stream_fn import set_default_stream_fn
from pidrei_agent.types import (
    AfterToolCallResult,
    AgentContext,
    AgentLoopConfig,
    AgentLoopTurnUpdate,
    AgentRequestUpdate,
    AgentTool,
    AgentToolResult,
    AgentTurnDecision,
    BeforeToolCallResult,
)
from pidrei_ai.types import (
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    Model,
    ModelCost,
    SystemMessage,
    TextContent,
    ToolCall,
    TranscriptContext,
    Usage,
    UsageCost,
    UserMessage,
)
from pidrei_ai.utils.event_stream import AssistantMessageEventStream


VALUE_SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "string"}},
    "required": ["value"],
}


class FnTool(AgentTool):
    def __init__(
        self,
        name: str,
        label: str,
        description: str,
        parameters: dict,
        execute,
        execution_mode=None,
        prepare_arguments=None,
    ):
        self.name = name
        self.label = label
        self.description = description
        self.parameters = parameters
        self.execution_mode = execution_mode
        self.prepare_arguments = prepare_arguments
        self._execute = execute

    async def execute(self, tool_call_id, params, cancel, on_update):
        return await self._execute(tool_call_id, params)


def create_model() -> Model:
    return Model(
        id="mock",
        name="mock",
        api="openai-responses",
        provider="openai",
        base_url="https://example.invalid",
        reasoning=False,
        input=["text"],
        cost=ModelCost(),
        context_window=8192,
        max_tokens=2048,
    )


def create_assistant_message(content, stop_reason="stop") -> AssistantMessage:
    return AssistantMessage(
        content=content,
        api="openai-responses",
        provider="openai",
        model="mock",
        usage=Usage(),
        stop_reason=stop_reason,
        timestamp=int(time.time() * 1000),
    )


def create_user_message(text: str) -> UserMessage:
    return UserMessage(content=text, timestamp=int(time.time() * 1000))


async def identity_converter(messages):
    """Simple identity converter for tests - passes through standard messages."""
    return [m for m in messages if getattr(m, "role", None) in ("system", "user", "assistant", "toolResult")]


def done_stream(message: AssistantMessage, reason: str = "stop") -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()
    stream.push(DoneEvent(reason=reason, message=message))
    return stream


@pytest.mark.tonio
async def test_uses_the_configured_default_when_a_legacy_caller_omits_stream_fn():
    calls = 0

    async def default_fn(_model, _context, _options):
        nonlocal calls
        calls += 1
        return done_stream(create_assistant_message([TextContent(text="fallback")]))

    set_default_stream_fn(default_fn)
    try:
        context = AgentContext(messages=[], tools=[])
        config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter)
        stream = agent_loop([create_user_message("Hello")], context, config, None)

        await stream.result()
        assert calls == 1
    finally:
        set_default_stream_fn(None)


@pytest.mark.tonio
async def test_should_emit_events_with_agent_message_types():
    context = AgentContext(messages=[], tools=[])
    user_prompt = create_user_message("Hello")
    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter)

    async def stream_fn(_model, _context, _options):
        return done_stream(create_assistant_message([TextContent(text="Hi there!")]))

    events = []
    stream = agent_loop([user_prompt], context, config, None, stream_fn)
    async for event in stream:
        events.append(event)
    messages = await stream.result()

    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"

    event_types = [event.type for event in events]
    for expected in ("agent_start", "turn_start", "message_start", "message_end", "turn_end", "agent_end"):
        assert expected in event_types


@pytest.mark.tonio
async def test_should_build_provider_context_exclusively_from_transcript_messages():
    initial_system = SystemMessage(content="Transcript prompt", tools_added=[], timestamp=1)
    context = AgentContext(messages=[], tools=[])
    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter)

    provider_contexts = []

    async def stream_fn(_model, provider_context, _options):
        provider_contexts.append(provider_context)
        return done_stream(create_assistant_message([TextContent(text="done")]))

    stream = agent_loop([initial_system, create_user_message("Hello")], context, config, None, stream_fn)

    await stream.result()
    # The provider receives a transcript: no top-level prompt or tool fields.
    assert len(provider_contexts) == 1
    assert type(provider_contexts[0]) is TranscriptContext
    assert provider_contexts[0].messages[0] is initial_system


@dataclass
class CustomNotification:
    role: str
    text: str
    timestamp: int


@pytest.mark.tonio
async def test_should_handle_custom_message_types_via_convert_to_llm():
    notification = CustomNotification(role="notification", text="This is a notification", timestamp=int(time.time()))
    context = AgentContext(messages=[notification], tools=[])
    user_prompt = create_user_message("Hello")

    converted_messages = []

    async def convert(messages):
        nonlocal converted_messages
        converted_messages = [
            m
            for m in messages
            if getattr(m, "role", None) != "notification"
            and getattr(m, "role", None) in ("user", "assistant", "toolResult")
        ]
        return converted_messages

    config = AgentLoopConfig(model=create_model(), convert_to_llm=convert)

    async def stream_fn(_model, _context, _options):
        return done_stream(create_assistant_message([TextContent(text="Response")]))

    stream = agent_loop([user_prompt], context, config, None, stream_fn)
    async for _event in stream:
        pass

    # The notification should have been filtered out in convert_to_llm.
    assert len(converted_messages) == 1
    assert converted_messages[0].role == "user"


@pytest.mark.tonio
async def test_should_apply_transform_context_before_convert_to_llm():
    context = AgentContext(
        messages=[
            create_user_message("old message 1"),
            create_assistant_message([TextContent(text="old response 1")]),
            create_user_message("old message 2"),
            create_assistant_message([TextContent(text="old response 2")]),
        ],
        tools=[],
    )
    user_prompt = create_user_message("new message")

    transformed_messages = []
    converted_messages = []

    async def transform_context(messages, _cancel):
        nonlocal transformed_messages
        transformed_messages = messages[-2:]
        return transformed_messages

    async def convert(messages):
        nonlocal converted_messages
        converted_messages = [m for m in messages if getattr(m, "role", None) in ("user", "assistant", "toolResult")]
        return converted_messages

    config = AgentLoopConfig(model=create_model(), convert_to_llm=convert, transform_context=transform_context)

    async def stream_fn(_model, _context, _options):
        return done_stream(create_assistant_message([TextContent(text="Response")]))

    stream = agent_loop([user_prompt], context, config, None, stream_fn)
    async for _event in stream:
        pass

    assert len(transformed_messages) == 2
    assert len(converted_messages) == 2


@pytest.mark.tonio
async def test_should_handle_tool_calls_and_results():
    executed = []
    tool_usage = Usage(
        input=1,
        output=2,
        cache_read=3,
        cache_write=4,
        total_tokens=10,
        cost=UsageCost(input=0.1, output=0.2, cache_read=0.3, cache_write=0.4, total=1),
    )
    patched_tool_usage = Usage(
        input=5,
        output=6,
        cache_read=7,
        cache_write=8,
        total_tokens=26,
        cost=UsageCost(input=0.5, output=0.6, cache_read=0.7, cache_write=0.8, total=2.6),
    )
    observed_tool_usage = None

    async def execute(_tool_call_id, params):
        executed.append(params["value"])
        return AgentToolResult(
            content=[TextContent(text=f"echoed: {params['value']}")],
            details={"value": params["value"]},
            usage=tool_usage,
        )

    tool = FnTool("echo", "Echo", "Echo tool", VALUE_SCHEMA, execute)
    context = AgentContext(messages=[], tools=[tool])
    user_prompt = create_user_message("echo something")

    async def after_tool_call(ctx, _cancel):
        nonlocal observed_tool_usage
        observed_tool_usage = ctx.result.usage
        return AfterToolCallResult(usage=patched_tool_usage)

    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter, after_tool_call=after_tool_call)

    call_index = 0

    async def stream_fn(_model, _context, _options):
        nonlocal call_index
        if call_index == 0:
            message = create_assistant_message(
                [ToolCall(id="tool-1", name="echo", arguments={"value": "hello"})], "toolUse"
            )
            stream = done_stream(message, "toolUse")
        else:
            stream = done_stream(create_assistant_message([TextContent(text="done")]))
        call_index += 1
        return stream

    events = []
    stream = agent_loop([user_prompt], context, config, None, stream_fn)
    async for event in stream:
        events.append(event)

    assert executed == ["hello"]

    tool_start = next((e for e in events if e.type == "tool_execution_start"), None)
    tool_end = next((e for e in events if e.type == "tool_execution_end"), None)
    assert tool_start is not None
    assert tool_end is not None
    assert tool_end.is_error is False
    assert observed_tool_usage == tool_usage
    messages = await stream.result()
    tool_result = next((m for m in messages if getattr(m, "role", None) == "toolResult"), None)
    assert tool_result is not None
    assert tool_result.usage == patched_tool_usage


@pytest.mark.tonio
async def test_should_not_execute_tool_calls_from_a_length_truncated_assistant_message():
    executed = []

    async def execute(_tool_call_id, params):
        executed.append(params["value"])
        return AgentToolResult(
            content=[TextContent(text=f"echoed: {params['value']}")], details={"value": params["value"]}
        )

    tool = FnTool("echo", "Echo", "Echo tool", VALUE_SCHEMA, execute)
    context = AgentContext(messages=[], tools=[tool])
    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter)

    call_index = 0

    async def stream_fn(_model, _context, _options):
        nonlocal call_index
        if call_index == 0:
            # Output hit the token limit mid tool call: nothing in this message may execute.
            message = create_assistant_message(
                [ToolCall(id="tool-1", name="echo", arguments={"value": "hel"})], "length"
            )
            stream = done_stream(message, "length")
        else:
            stream = done_stream(create_assistant_message([TextContent(text="done")]))
        call_index += 1
        return stream

    events = []
    stream = agent_loop([create_user_message("echo something")], context, config, None, stream_fn)
    async for event in stream:
        events.append(event)

    # The tool must never execute with potentially truncated arguments.
    assert executed == []

    tool_end = next((e for e in events if e.type == "tool_execution_end"), None)
    assert tool_end is not None
    assert tool_end.is_error is True
    text = next((c for c in tool_end.result.content if c.type == "text"), None)
    assert text is not None
    assert "output token limit" in text.text

    # The loop continues so the model can re-issue the tool call.
    assert call_index == 2
    messages = await stream.result()
    assert messages[-1].role == "assistant"


@pytest.mark.tonio
async def test_should_execute_mutated_before_tool_call_args_without_revalidation():
    executed = []

    async def execute(_tool_call_id, params):
        executed.append(params["value"])
        return AgentToolResult(
            content=[TextContent(text=f"echoed: {params['value']}")], details={"value": params["value"]}
        )

    tool = FnTool("echo", "Echo", "Echo tool", VALUE_SCHEMA, execute)
    context = AgentContext(messages=[], tools=[tool])
    user_prompt = create_user_message("echo something")

    async def before_tool_call(ctx, _cancel):
        ctx.args["value"] = 123

    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter, before_tool_call=before_tool_call)

    call_index = 0

    async def stream_fn(_model, _context, _options):
        nonlocal call_index
        if call_index == 0:
            message = create_assistant_message(
                [ToolCall(id="tool-1", name="echo", arguments={"value": "hello"})], "toolUse"
            )
            stream = done_stream(message, "toolUse")
        else:
            stream = done_stream(create_assistant_message([TextContent(text="done")]))
        call_index += 1
        return stream

    stream = agent_loop([user_prompt], context, config, None, stream_fn)
    async for _event in stream:
        pass

    assert executed == [123]


@pytest.mark.tonio
async def test_should_prepare_tool_arguments_for_validation():
    edit_schema = {
        "type": "object",
        "properties": {
            "edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"oldText": {"type": "string"}, "newText": {"type": "string"}},
                    "required": ["oldText", "newText"],
                },
            }
        },
        "required": ["edits"],
    }
    executed = []

    def prepare_arguments(args):
        if not isinstance(args, dict):
            return args
        if not isinstance(args.get("oldText"), str) or not isinstance(args.get("newText"), str):
            return args
        return {"edits": [*args.get("edits", []), {"oldText": args["oldText"], "newText": args["newText"]}]}

    async def execute(_tool_call_id, params):
        executed.append(params["edits"])
        return AgentToolResult(
            content=[TextContent(text=f"edited {len(params['edits'])}")], details={"count": len(params["edits"])}
        )

    tool = FnTool("edit", "Edit", "Edit tool", edit_schema, execute, prepare_arguments=prepare_arguments)
    context = AgentContext(messages=[], tools=[tool])
    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter)

    call_index = 0

    async def stream_fn(_model, _context, _options):
        nonlocal call_index
        if call_index == 0:
            message = create_assistant_message(
                [ToolCall(id="tool-1", name="edit", arguments={"oldText": "before", "newText": "after"})],
                "toolUse",
            )
            stream = done_stream(message, "toolUse")
        else:
            stream = done_stream(create_assistant_message([TextContent(text="done")]))
        call_index += 1
        return stream

    stream = agent_loop([create_user_message("edit something")], context, config, None, stream_fn)
    async for _event in stream:
        pass

    assert executed == [[{"oldText": "before", "newText": "after"}]]


@pytest.mark.tonio
async def test_should_emit_tool_execution_end_in_completion_order_but_persist_results_in_source_order():
    first_resolved = False
    parallel_observed = False
    first_done = tonio.Event()

    async def execute(_tool_call_id, params):
        nonlocal first_resolved, parallel_observed
        if params["value"] == "first":
            # Bounded: a sequential regression never releases the gate.
            await first_done.wait(5)
            first_resolved = True
        if params["value"] == "second" and not first_resolved:
            parallel_observed = True
        return AgentToolResult(
            content=[TextContent(text=f"echoed: {params['value']}")], details={"value": params["value"]}
        )

    tool = FnTool("echo", "Echo", "Echo tool", VALUE_SCHEMA, execute)
    context = AgentContext(messages=[], tools=[tool])
    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter, tool_execution="parallel")

    call_index = 0

    async def stream_fn(_model, _context, _options):
        nonlocal call_index
        if call_index == 0:
            message = create_assistant_message(
                [
                    ToolCall(id="tool-1", name="echo", arguments={"value": "first"}),
                    ToolCall(id="tool-2", name="echo", arguments={"value": "second"}),
                ],
                "toolUse",
            )
            stream = done_stream(message, "toolUse")
        else:
            stream = done_stream(create_assistant_message([TextContent(text="done")]))
        call_index += 1
        return stream

    # pidrei: pi releases the first tool after a 20 ms timer. Here it is
    # released once the second tool's end has been emitted, which is the
    # completion order the test asserts, without a wall-clock race.
    events = []
    stream = agent_loop([create_user_message("echo both")], context, config, None, stream_fn)
    async for event in stream:
        events.append(event)
        if event.type == "tool_execution_end" and event.tool_call_id == "tool-2":
            first_done.set()

    tool_execution_end_ids = [e.tool_call_id for e in events if e.type == "tool_execution_end"]
    tool_result_ids = [
        e.message.tool_call_id
        for e in events
        if e.type == "message_end" and getattr(e.message, "role", None) == "toolResult"
    ]
    turn_tool_result_ids = [
        tool_result.tool_call_id for e in events if e.type == "turn_end" for tool_result in e.tool_results
    ]

    assert parallel_observed is True
    assert tool_execution_end_ids == ["tool-2", "tool-1"]
    assert tool_result_ids == ["tool-1", "tool-2"]
    assert turn_tool_result_ids == ["tool-1", "tool-2"]


@pytest.mark.tonio
async def test_should_inject_queued_messages_after_all_tool_calls_complete():
    executed = []

    async def execute(_tool_call_id, params):
        executed.append(params["value"])
        return AgentToolResult(content=[TextContent(text=f"ok:{params['value']}")], details={"value": params["value"]})

    tool = FnTool("echo", "Echo", "Echo tool", VALUE_SCHEMA, execute)
    context = AgentContext(messages=[], tools=[tool])
    user_prompt = create_user_message("start")
    queued_user_message = create_user_message("interrupt")

    queued_delivered = False
    call_index = 0
    saw_interrupt_in_context = False

    async def get_steering_messages():
        nonlocal queued_delivered
        # Return steering message after tool execution has started.
        if len(executed) >= 1 and not queued_delivered:
            queued_delivered = True
            return [queued_user_message]
        return []

    config = AgentLoopConfig(
        model=create_model(),
        convert_to_llm=identity_converter,
        tool_execution="sequential",
        get_steering_messages=get_steering_messages,
    )

    async def stream_fn(_model, ctx, _options):
        nonlocal call_index, saw_interrupt_in_context
        # Check if interrupt message is in context on second call.
        if call_index == 1:
            saw_interrupt_in_context = any(
                getattr(m, "role", None) == "user" and isinstance(m.content, str) and m.content == "interrupt"
                for m in ctx.messages
            )
        if call_index == 0:
            message = create_assistant_message(
                [
                    ToolCall(id="tool-1", name="echo", arguments={"value": "first"}),
                    ToolCall(id="tool-2", name="echo", arguments={"value": "second"}),
                ],
                "toolUse",
            )
            stream = done_stream(message, "toolUse")
        else:
            stream = done_stream(create_assistant_message([TextContent(text="done")]))
        call_index += 1
        return stream

    events = []
    stream = agent_loop([user_prompt], context, config, None, stream_fn)
    async for event in stream:
        events.append(event)

    # Both tools should execute before steering is injected.
    assert executed == ["first", "second"]

    tool_ends = [e for e in events if e.type == "tool_execution_end"]
    assert len(tool_ends) == 2
    assert tool_ends[0].is_error is False
    assert tool_ends[1].is_error is False

    # Queued message should appear in events after both tool result messages.
    event_sequence = []
    for event in events:
        if event.type != "message_start":
            continue
        if getattr(event.message, "role", None) == "toolResult":
            event_sequence.append(f"tool:{event.message.tool_call_id}")
        elif getattr(event.message, "role", None) == "user" and isinstance(event.message.content, str):
            event_sequence.append(event.message.content)
    assert "interrupt" in event_sequence
    assert event_sequence.index("tool:tool-1") < event_sequence.index("interrupt")
    assert event_sequence.index("tool:tool-2") < event_sequence.index("interrupt")

    # Interrupt message should be in context when second LLM call is made.
    assert saw_interrupt_in_context is True


@pytest.mark.tonio
async def test_should_force_sequential_when_a_tool_has_sequential_mode_with_default_parallel_config():
    first_resolved = False
    parallel_observed = False
    first_started = tonio.Event()
    first_done = tonio.Event()

    async def execute(_tool_call_id, params):
        nonlocal first_resolved, parallel_observed
        if params["value"] == "first":
            first_started.set()
            await first_done.wait(None)
            first_resolved = True
        if params["value"] == "second" and not first_resolved:
            parallel_observed = True
        return AgentToolResult(
            content=[TextContent(text=f"slow: {params['value']}")], details={"value": params["value"]}
        )

    slow_tool = FnTool("slow", "Slow", "Slow tool", VALUE_SCHEMA, execute, execution_mode="sequential")
    context = AgentContext(messages=[], tools=[slow_tool])
    # config is parallel (default), but the tool forces sequential.
    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter)

    # pidrei: pi releases the first tool after a 20 ms timer; here once it is
    # parked. `parallel_observed` alone would need a timing window to catch a
    # parallel regression, so the event order below is the deterministic check.
    async def release_first():
        await first_started.wait(None)
        first_done.set()

    call_index = 0

    async def stream_fn(_model, _context, _options):
        nonlocal call_index
        if call_index == 0:
            message = create_assistant_message(
                [
                    ToolCall(id="tool-1", name="slow", arguments={"value": "first"}),
                    ToolCall(id="tool-2", name="slow", arguments={"value": "second"}),
                ],
                "toolUse",
            )
            stream = done_stream(message, "toolUse")
            tonio.spawn.without_tracking(release_first())
        else:
            stream = done_stream(create_assistant_message([TextContent(text="done")]))
        call_index += 1
        return stream

    events = []
    stream = agent_loop([create_user_message("run both")], context, config, None, stream_fn)
    async for event in stream:
        events.append(event)

    # With sequential execution, second tool should NOT start before first finishes.
    assert parallel_observed is False
    tool_lifecycle = [(e.type, e.tool_call_id) for e in events if e.type.startswith("tool_execution_")]
    assert tool_lifecycle == [
        ("tool_execution_start", "tool-1"),
        ("tool_execution_end", "tool-1"),
        ("tool_execution_start", "tool-2"),
        ("tool_execution_end", "tool-2"),
    ]

    tool_result_ids = [
        e.message.tool_call_id
        for e in events
        if e.type == "message_end" and getattr(e.message, "role", None) == "toolResult"
    ]
    assert tool_result_ids == ["tool-1", "tool-2"]


@pytest.mark.tonio
async def test_should_force_sequential_when_one_of_multiple_tools_has_sequential_mode():
    execution_order = []
    slow_started = tonio.Event()
    slow_done = tonio.Event()

    async def execute_slow(_tool_call_id, params):
        execution_order.append(f"slow:{params['value']}")
        if params["value"] == "a":
            slow_started.set()
            await slow_done.wait(None)
        return AgentToolResult(
            content=[TextContent(text=f"slow: {params['value']}")], details={"value": params["value"]}
        )

    async def execute_fast(_tool_call_id, params):
        execution_order.append(f"fast:{params['value']}")
        return AgentToolResult(
            content=[TextContent(text=f"fast: {params['value']}")], details={"value": params["value"]}
        )

    slow_tool = FnTool("slow", "Slow", "Slow tool", VALUE_SCHEMA, execute_slow, execution_mode="sequential")
    fast_tool = FnTool("fast", "Fast", "Fast tool", VALUE_SCHEMA, execute_fast)
    context = AgentContext(messages=[], tools=[slow_tool, fast_tool])
    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter)

    # pidrei: released once parked instead of after pi's 20 ms timer; the
    # event order below is the deterministic check against parallel dispatch.
    async def release_slow():
        await slow_started.wait(None)
        slow_done.set()

    call_index = 0

    async def stream_fn(_model, _context, _options):
        nonlocal call_index
        if call_index == 0:
            message = create_assistant_message(
                [
                    ToolCall(id="tool-1", name="slow", arguments={"value": "a"}),
                    ToolCall(id="tool-2", name="fast", arguments={"value": "b"}),
                ],
                "toolUse",
            )
            stream = done_stream(message, "toolUse")
            tonio.spawn.without_tracking(release_slow())
        else:
            stream = done_stream(create_assistant_message([TextContent(text="done")]))
        call_index += 1
        return stream

    events = []
    stream = agent_loop([create_user_message("run both")], context, config, None, stream_fn)
    async for event in stream:
        events.append(event)

    # Fast tool should NOT run before slow tool finishes.
    assert execution_order[0] == "slow:a"
    assert "fast:b" in execution_order
    tool_lifecycle = [(e.type, e.tool_call_id) for e in events if e.type.startswith("tool_execution_")]
    assert tool_lifecycle == [
        ("tool_execution_start", "tool-1"),
        ("tool_execution_end", "tool-1"),
        ("tool_execution_start", "tool-2"),
        ("tool_execution_end", "tool-2"),
    ]


@pytest.mark.tonio
async def test_should_allow_parallel_execution_when_all_tools_have_parallel_mode():
    first_resolved = False
    parallel_observed = False
    first_done = tonio.Event()

    async def execute(_tool_call_id, params):
        nonlocal first_resolved, parallel_observed
        if params["value"] == "first":
            # Bounded: a sequential regression never releases the gate.
            await first_done.wait(5)
            first_resolved = True
        if params["value"] == "second" and not first_resolved:
            parallel_observed = True
        return AgentToolResult(
            content=[TextContent(text=f"echoed: {params['value']}")], details={"value": params["value"]}
        )

    tool = FnTool("echo", "Echo", "Echo tool", VALUE_SCHEMA, execute, execution_mode="parallel")
    context = AgentContext(messages=[], tools=[tool])
    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter)

    call_index = 0

    async def stream_fn(_model, _context, _options):
        nonlocal call_index
        if call_index == 0:
            message = create_assistant_message(
                [
                    ToolCall(id="tool-1", name="echo", arguments={"value": "first"}),
                    ToolCall(id="tool-2", name="echo", arguments={"value": "second"}),
                ],
                "toolUse",
            )
            stream = done_stream(message, "toolUse")
        else:
            stream = done_stream(create_assistant_message([TextContent(text="done")]))
        call_index += 1
        return stream

    # pidrei: the first tool is released once the second has ended instead of
    # after pi's 20 ms timer, so `parallel_observed` does not race the clock.
    stream = agent_loop([create_user_message("echo both")], context, config, None, stream_fn)
    async for event in stream:
        if event.type == "tool_execution_end" and event.tool_call_id == "tool-2":
            first_done.set()

    # With execution_mode="parallel", second tool should start before first finishes.
    assert parallel_observed is True


@pytest.mark.tonio
async def test_should_use_prepare_next_turn_snapshot_before_continuing():
    async def execute(_tool_call_id, params):
        return AgentToolResult(
            content=[TextContent(text=f"echoed: {params['value']}")], details={"value": params["value"]}
        )

    tool = FnTool("echo", "Echo", "Echo tool", VALUE_SCHEMA, execute)
    context = AgentContext(messages=[], tools=[tool])
    converted_second_turn_has_update = False
    prepare_calls = 0
    prepared = False

    async def prepare_next_turn(ctx):
        nonlocal prepare_calls, prepared
        prepare_calls += 1
        if prepared:
            return None
        prepared = True
        return AgentLoopTurnUpdate(
            context=AgentContext(messages=list(ctx.context.messages), tools=ctx.context.tools),
            messages=[SystemMessage(content="updated guidance", timestamp=1)],
        )

    config = AgentLoopConfig(
        model=create_model(), convert_to_llm=identity_converter, prepare_next_turn=prepare_next_turn
    )

    llm_calls = 0

    async def stream_fn(_model, ctx, _options):
        nonlocal llm_calls, converted_second_turn_has_update
        llm_calls += 1
        if llm_calls == 2:
            converted_second_turn_has_update = any(
                message.role == "system" and message.content == "updated guidance" for message in ctx.messages
            )
        if llm_calls == 1:
            return done_stream(
                create_assistant_message([ToolCall(id="tool-1", name="echo", arguments={"value": "hello"})], "toolUse"),
                "toolUse",
            )
        return done_stream(create_assistant_message([TextContent(text="done")]))

    stream = agent_loop([create_user_message("echo something")], context, config, None, stream_fn)
    async for _event in stream:
        pass

    assert llm_calls == 2
    assert prepare_calls == 1
    assert converted_second_turn_has_update is True


def _noop_tool() -> FnTool:
    async def execute(_tool_call_id, _params):
        return AgentToolResult(content=[TextContent(text="done")], details=None)

    return FnTool("noop", "Noop", "Noop tool", {"type": "object", "properties": {}}, execute)


def _noop_call_then_text(provider_calls: int) -> AssistantMessageEventStream:
    if provider_calls == 1:
        return done_stream(
            create_assistant_message([ToolCall(id="tool-1", name="noop", arguments={})], "toolUse"), "toolUse"
        )
    return done_stream(create_assistant_message([TextContent(text="done")]))


@pytest.mark.tonio
async def test_runs_finish_turn_after_tool_result_messages_and_before_turn_end():
    async def execute(_tool_call_id, params):
        return AgentToolResult(
            content=[TextContent(text=params["value"])], details={"value": params["value"]}, terminate=True
        )

    tool = FnTool("echo", "Echo", "Echo tool", VALUE_SCHEMA, execute)
    ordering: list[str] = []

    async def finish_turn(turn, _cancel):
        ordering.append("finishTurn")
        assert len(turn.tool_results) == 1
        assert turn.context.messages[-1].role == "toolResult"

    async def emit(event):
        if event.type == "message_end":
            ordering.append(f"message_end:{event.message.role}")
        if event.type == "turn_end":
            ordering.append("turn_end")

    async def stream_fn(_model, _context, _options):
        return done_stream(
            create_assistant_message([ToolCall(id="tool-1", name="echo", arguments={"value": "hello"})], "toolUse"),
            "toolUse",
        )

    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter, finish_turn=finish_turn)
    await run_agent_loop(
        [create_user_message("echo")], AgentContext(messages=[], tools=[tool]), config, emit, None, stream_fn
    )

    assert ordering[-3:] == ["message_end:toolResult", "finishTurn", "turn_end"]


@pytest.mark.tonio
@pytest.mark.parametrize("reason", ["error", "aborted"])
async def test_runs_finish_turn_for_a_failed_assistant_before_turn_end_without_changing_the_hard_exit(reason):
    ordering: list[str] = []
    provider_calls = 0
    steering_polls = 0
    follow_up_polls = 0

    async def finish_turn(turn, _cancel):
        assert turn.message.stop_reason == reason
        ordering.append("finishTurn")
        return AgentTurnDecision("continue")

    async def get_steering_messages():
        nonlocal steering_polls
        steering_polls += 1
        return []

    async def get_follow_up_messages():
        nonlocal follow_up_polls
        follow_up_polls += 1
        return [create_user_message("queued")]

    async def emit(event):
        if event.type == "turn_end":
            ordering.append("turn_end")

    async def stream_fn(_model, _context, _options):
        nonlocal provider_calls
        provider_calls += 1
        stream = AssistantMessageEventStream()
        failed = replace(create_assistant_message([], reason), error_message=reason)
        stream.push(ErrorEvent(reason=reason, error=failed))
        return stream

    config = AgentLoopConfig(
        model=create_model(),
        convert_to_llm=identity_converter,
        finish_turn=finish_turn,
        get_steering_messages=get_steering_messages,
        get_follow_up_messages=get_follow_up_messages,
    )
    await run_agent_loop(
        [create_user_message("run")], AgentContext(messages=[], tools=[]), config, emit, None, stream_fn
    )

    assert ordering == ["finishTurn", "turn_end"]
    assert provider_calls == 1
    assert steering_polls == 1
    assert follow_up_polls == 0


@pytest.mark.tonio
async def test_action_end_skips_queue_polling_and_next_turn_preparation():
    provider_calls = 0
    steering_polls = 0
    follow_up_polls = 0
    prepare_next_turn_calls = 0

    async def finish_turn(_turn, _cancel):
        return AgentTurnDecision("end")

    async def prepare_next_turn(_context):
        nonlocal prepare_next_turn_calls
        prepare_next_turn_calls += 1

    async def get_steering_messages():
        nonlocal steering_polls
        steering_polls += 1
        return []

    async def get_follow_up_messages():
        nonlocal follow_up_polls
        follow_up_polls += 1
        return [create_user_message("queued")]

    async def stream_fn(_model, _context, _options):
        nonlocal provider_calls
        provider_calls += 1
        return done_stream(
            create_assistant_message([ToolCall(id="tool-1", name="noop", arguments={})], "toolUse"), "toolUse"
        )

    config = AgentLoopConfig(
        model=create_model(),
        convert_to_llm=identity_converter,
        finish_turn=finish_turn,
        prepare_next_turn=prepare_next_turn,
        get_steering_messages=get_steering_messages,
        get_follow_up_messages=get_follow_up_messages,
    )
    stream = agent_loop(
        [create_user_message("run")], AgentContext(messages=[], tools=[_noop_tool()]), config, None, stream_fn
    )
    await stream.result()

    assert provider_calls == 1
    assert steering_polls == 1
    assert follow_up_polls == 0
    assert prepare_next_turn_calls == 0


@pytest.mark.tonio
async def test_makes_exactly_one_context_only_request_when_no_natural_request_satisfies_continuation():
    provider_calls = 0
    finish_calls = 0

    async def finish_turn(_turn, _cancel):
        nonlocal finish_calls
        finish_calls += 1
        return AgentTurnDecision("continue") if finish_calls == 1 else None

    async def stream_fn(_model, _context, _options):
        nonlocal provider_calls
        provider_calls += 1
        return done_stream(create_assistant_message([TextContent(text=f"response {provider_calls}")]))

    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter, finish_turn=finish_turn)
    stream = agent_loop([create_user_message("run")], AgentContext(messages=[], tools=[]), config, None, stream_fn)
    await stream.result()

    assert provider_calls == 2
    assert finish_calls == 2


@pytest.mark.tonio
async def test_lets_a_natural_tool_result_request_satisfy_continuation():
    provider_calls = 0
    finish_calls = 0

    async def finish_turn(_turn, _cancel):
        nonlocal finish_calls
        finish_calls += 1
        return AgentTurnDecision("continue") if finish_calls == 1 else None

    async def stream_fn(_model, _context, _options):
        nonlocal provider_calls
        provider_calls += 1
        return _noop_call_then_text(provider_calls)

    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter, finish_turn=finish_turn)
    stream = agent_loop(
        [create_user_message("run")], AgentContext(messages=[], tools=[_noop_tool()]), config, None, stream_fn
    )
    await stream.result()

    assert provider_calls == 2
    assert finish_calls == 2


@pytest.mark.tonio
@pytest.mark.parametrize("queue_kind", ["steering", "follow-up"])
async def test_lets_a_natural_queued_request_satisfy_continuation(queue_kind):
    queued_message = create_user_message(queue_kind)
    provider_calls = 0
    finish_calls = 0
    steering_polls = 0
    follow_up_delivered = False
    second_request_users: list[str] = []

    async def finish_turn(_turn, _cancel):
        nonlocal finish_calls
        finish_calls += 1
        return AgentTurnDecision("continue") if finish_calls == 1 else None

    async def get_steering_messages():
        nonlocal steering_polls
        steering_polls += 1
        return [queued_message] if queue_kind == "steering" and steering_polls == 2 else []

    async def get_follow_up_messages():
        nonlocal follow_up_delivered
        if queue_kind != "follow-up" or follow_up_delivered:
            return []
        follow_up_delivered = True
        return [queued_message]

    async def stream_fn(_model, context, _options):
        nonlocal provider_calls
        provider_calls += 1
        if provider_calls == 2:
            second_request_users.extend(
                message.content
                for message in context.messages
                if message.role == "user" and isinstance(message.content, str)
            )
        return done_stream(create_assistant_message([TextContent(text="done")]))

    config = AgentLoopConfig(
        model=create_model(),
        convert_to_llm=identity_converter,
        finish_turn=finish_turn,
        get_steering_messages=get_steering_messages,
        get_follow_up_messages=get_follow_up_messages,
    )
    stream = agent_loop([create_user_message("run")], AgentContext(messages=[], tools=[]), config, None, stream_fn)
    await stream.result()

    assert provider_calls == 2
    assert finish_calls == 2
    assert queue_kind in second_request_users


@pytest.mark.tonio
async def test_prepares_the_initial_request_after_pending_messages_and_can_replace_request_state():
    replacement_model = replace(create_model(), id="replacement", name="replacement")
    canonical_message = create_user_message("canonical projection")
    steering_message = create_user_message("steering")
    completed_messages: list = []
    steering_delivered = False
    prepare_calls = 0

    async def get_steering_messages():
        nonlocal steering_delivered
        if steering_delivered:
            return []
        steering_delivered = True
        return [steering_message]

    async def prepare_request(request, _cancel):
        nonlocal prepare_calls
        prepare_calls += 1
        assert any(message is steering_message for message in completed_messages)
        assert any(message is steering_message for message in request.context.messages)
        return AgentRequestUpdate(
            context=replace(request.context, messages=[canonical_message]),
            model=replacement_model,
            thinking_level="high",
        )

    async def emit(event):
        if event.type == "message_end":
            completed_messages.append(event.message)

    seen: list[tuple] = []

    async def stream_fn(model, context, options):
        seen.append((model, list(context.messages), options.reasoning))
        return done_stream(create_assistant_message([TextContent(text="done")]))

    config = AgentLoopConfig(
        model=create_model(),
        convert_to_llm=identity_converter,
        get_steering_messages=get_steering_messages,
        prepare_request=prepare_request,
    )
    await run_agent_loop(
        [create_user_message("prompt")], AgentContext(messages=[], tools=[]), config, emit, None, stream_fn
    )

    assert prepare_calls == 1
    # Assertions inside a stream fn would be swallowed into an error message, so check afterwards.
    [(model, messages, reasoning)] = seen
    assert model is replacement_model
    assert messages == [canonical_message]
    assert reasoning == "high"


@pytest.mark.tonio
async def test_does_not_poll_steering_after_prepare_request():
    queued: list = []
    late_steering = create_user_message("late steering")
    request_included_steering: list[bool] = []
    request_preparations = 0
    steering_polls = 0

    async def get_steering_messages():
        nonlocal steering_polls
        steering_polls += 1
        drained = list(queued)
        queued.clear()
        return drained

    async def prepare_request(_request, _cancel):
        nonlocal request_preparations
        request_preparations += 1
        if request_preparations == 1:
            queued.append(late_steering)

    async def stream_fn(_model, context, _options):
        request_included_steering.append(any(message is late_steering for message in context.messages))
        return done_stream(create_assistant_message([TextContent(text="done")]))

    config = AgentLoopConfig(
        model=create_model(),
        convert_to_llm=identity_converter,
        get_steering_messages=get_steering_messages,
        prepare_request=prepare_request,
    )
    stream = agent_loop([create_user_message("run")], AgentContext(messages=[], tools=[]), config, None, stream_fn)
    await stream.result()

    assert request_included_steering == [False, True]
    assert request_preparations == 2
    # Startup, post-turn delivery, then the final natural-stop check.
    assert steering_polls == 3


@pytest.mark.tonio
async def test_picks_up_steering_queued_during_prepare_next_turn_before_the_next_request():
    queued: list = []
    late_steering = create_user_message("late steering")
    provider_calls = 0
    second_request_included_steering = False

    async def prepare_next_turn(_context):
        queued.append(late_steering)

    async def get_steering_messages():
        drained = list(queued)
        queued.clear()
        return drained

    async def stream_fn(_model, context, _options):
        nonlocal provider_calls, second_request_included_steering
        provider_calls += 1
        if provider_calls == 2:
            second_request_included_steering = any(message is late_steering for message in context.messages)
        return _noop_call_then_text(provider_calls)

    config = AgentLoopConfig(
        model=create_model(),
        convert_to_llm=identity_converter,
        prepare_next_turn=prepare_next_turn,
        get_steering_messages=get_steering_messages,
    )
    stream = agent_loop(
        [create_user_message("run")], AgentContext(messages=[], tools=[_noop_tool()]), config, None, stream_fn
    )
    await stream.result()

    assert provider_calls == 2
    assert second_request_included_steering is True


@pytest.mark.tonio
async def test_action_end_receives_finalized_turn_context_and_stops_before_queue_polling():
    executed = []

    async def execute(_tool_call_id, params):
        executed.append(params["value"])
        return AgentToolResult(
            content=[TextContent(text=f"echoed: {params['value']}")], details={"value": params["value"]}
        )

    tool = FnTool("echo", "Echo", "Echo tool", VALUE_SCHEMA, execute)
    context = AgentContext(messages=[], tools=[tool])

    steering_polls = 0
    follow_up_polls = 0
    callback_tool_result_ids = []
    callback_context_roles = []

    async def get_steering_messages():
        nonlocal steering_polls
        steering_polls += 1
        return []

    async def get_follow_up_messages():
        nonlocal follow_up_polls
        follow_up_polls += 1
        return [create_user_message("follow up should stay queued")]

    async def finish_turn(ctx, _cancel):
        nonlocal callback_tool_result_ids, callback_context_roles
        assert ctx.message.role == "assistant"
        callback_tool_result_ids = [tool_result.tool_call_id for tool_result in ctx.tool_results]
        callback_context_roles = [getattr(m, "role", None) for m in ctx.context.messages]
        return AgentTurnDecision("end")

    config = AgentLoopConfig(
        model=create_model(),
        convert_to_llm=identity_converter,
        finish_turn=finish_turn,
        get_steering_messages=get_steering_messages,
        get_follow_up_messages=get_follow_up_messages,
    )

    llm_calls = 0

    async def stream_fn(_model, _context, _options):
        nonlocal llm_calls
        llm_calls += 1
        if llm_calls == 1:
            return done_stream(
                create_assistant_message([ToolCall(id="tool-1", name="echo", arguments={"value": "hello"})], "toolUse"),
                "toolUse",
            )
        return done_stream(create_assistant_message([TextContent(text="should not run")]))

    events = []
    stream = agent_loop([create_user_message("echo something")], context, config, None, stream_fn)
    async for event in stream:
        events.append(event)

    messages = await stream.result()
    assert llm_calls == 1
    assert executed == ["hello"]
    assert steering_polls == 1
    assert follow_up_polls == 0
    assert callback_tool_result_ids == ["tool-1"]
    assert callback_context_roles == ["system", "user", "assistant", "toolResult"]
    # The context declares no tools, so the loop announces the loadout with a system message.
    assert [getattr(m, "role", None) for m in messages] == ["system", "user", "assistant", "toolResult"]
    assert [event.type for event in events] == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "message_start",
        "message_end",
        "message_start",
        "message_end",
        "tool_execution_start",
        "tool_execution_end",
        "message_start",
        "message_end",
        "turn_end",
        "agent_end",
    ]


@pytest.mark.tonio
async def test_should_stop_after_a_tool_batch_when_every_tool_result_sets_terminate_true():
    async def execute(_tool_call_id, params):
        return AgentToolResult(
            content=[TextContent(text=f"echoed: {params['value']}")],
            details={"value": params["value"]},
            terminate=True,
        )

    tool = FnTool("echo", "Echo", "Echo tool", VALUE_SCHEMA, execute)
    context = AgentContext(messages=[], tools=[tool])
    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter)

    llm_calls = 0

    async def stream_fn(_model, _context, _options):
        nonlocal llm_calls
        llm_calls += 1
        return done_stream(
            create_assistant_message([ToolCall(id="tool-1", name="echo", arguments={"value": "hello"})], "toolUse"),
            "toolUse",
        )

    events = []
    stream = agent_loop([create_user_message("echo something")], context, config, None, stream_fn)
    async for event in stream:
        events.append(event)

    messages = await stream.result()
    assert llm_calls == 1
    assert [getattr(m, "role", None) for m in messages] == ["system", "user", "assistant", "toolResult"]
    assert len([event for event in events if event.type == "turn_end"]) == 1


@pytest.mark.tonio
async def test_should_stop_after_a_blocked_tool_call_when_before_tool_call_sets_terminate_true():
    executed = False

    async def execute(_tool_call_id, params):
        nonlocal executed
        executed = True
        return AgentToolResult(content=[TextContent(text="should not execute")], details={"value": "unexpected"})

    tool = FnTool("echo", "Echo", "Echo tool", VALUE_SCHEMA, execute)
    context = AgentContext(messages=[], tools=[tool])

    async def before_tool_call(_context, _cancel):
        return BeforeToolCallResult(block=True, reason="Blocked by policy", terminate=True)

    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter, before_tool_call=before_tool_call)

    llm_calls = 0

    async def stream_fn(_model, _context, _options):
        nonlocal llm_calls
        llm_calls += 1
        if llm_calls == 1:
            return done_stream(
                create_assistant_message([ToolCall(id="tool-1", name="echo", arguments={"value": "hello"})], "toolUse"),
                "toolUse",
            )
        return done_stream(create_assistant_message([TextContent(text="should not run")]))

    stream = agent_loop([create_user_message("echo something")], context, config, None, stream_fn)
    async for _event in stream:
        pass

    messages = await stream.result()
    tool_result = next((m for m in messages if getattr(m, "role", None) == "toolResult"), None)
    assert executed is False
    assert llm_calls == 1
    assert tool_result is not None and tool_result.is_error is True
    assert TextContent(text="Blocked by policy") in tool_result.content


@pytest.mark.tonio
async def test_should_continue_after_a_mixed_batch_with_one_terminating_blocked_call():
    executed = []

    async def execute(_tool_call_id, params):
        executed.append(params["value"])
        return AgentToolResult(
            content=[TextContent(text=f"echoed: {params['value']}")], details={"value": params["value"]}
        )

    tool = FnTool("echo", "Echo", "Echo tool", VALUE_SCHEMA, execute)
    context = AgentContext(messages=[], tools=[tool])

    async def before_tool_call(before_context, _cancel):
        if before_context.args["value"] == "first":
            return BeforeToolCallResult(block=True, reason="Blocked first", terminate=True)
        return None

    config = AgentLoopConfig(
        model=create_model(),
        convert_to_llm=identity_converter,
        tool_execution="parallel",
        before_tool_call=before_tool_call,
    )

    llm_calls = 0

    async def stream_fn(_model, _context, _options):
        nonlocal llm_calls
        llm_calls += 1
        if llm_calls == 1:
            return done_stream(
                create_assistant_message(
                    [
                        ToolCall(id="tool-1", name="echo", arguments={"value": "first"}),
                        ToolCall(id="tool-2", name="echo", arguments={"value": "second"}),
                    ],
                    "toolUse",
                ),
                "toolUse",
            )
        return done_stream(create_assistant_message([TextContent(text="done")]))

    stream = agent_loop([create_user_message("echo both")], context, config, None, stream_fn)
    async for _event in stream:
        pass

    await stream.result()
    assert executed == ["second"]
    assert llm_calls == 2


@pytest.mark.tonio
async def test_should_continue_after_parallel_tool_calls_when_not_all_tool_results_terminate():
    async def execute(_tool_call_id, params):
        return AgentToolResult(
            content=[TextContent(text=f"echoed: {params['value']}")],
            details={"value": params["value"]},
            terminate=params["value"] == "first",
        )

    tool = FnTool("echo", "Echo", "Echo tool", VALUE_SCHEMA, execute)
    context = AgentContext(messages=[], tools=[tool])
    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter, tool_execution="parallel")

    call_index = 0

    async def stream_fn(_model, _context, _options):
        nonlocal call_index
        if call_index == 0:
            message = create_assistant_message(
                [
                    ToolCall(id="tool-1", name="echo", arguments={"value": "first"}),
                    ToolCall(id="tool-2", name="echo", arguments={"value": "second"}),
                ],
                "toolUse",
            )
            stream = done_stream(message, "toolUse")
        else:
            stream = done_stream(create_assistant_message([TextContent(text="done")]))
        call_index += 1
        return stream

    stream = agent_loop([create_user_message("echo both")], context, config, None, stream_fn)
    async for _event in stream:
        pass

    messages = await stream.result()
    assert call_index == 2
    assert [getattr(m, "role", None) for m in messages] == [
        "system",
        "user",
        "assistant",
        "toolResult",
        "toolResult",
        "assistant",
    ]


@pytest.mark.tonio
async def test_should_allow_after_tool_call_to_mark_a_tool_batch_as_terminating():
    async def execute(_tool_call_id, params):
        return AgentToolResult(
            content=[TextContent(text=f"echoed: {params['value']}")], details={"value": params["value"]}
        )

    tool = FnTool("echo", "Echo", "Echo tool", VALUE_SCHEMA, execute)
    context = AgentContext(messages=[], tools=[tool])

    async def after_tool_call(_ctx, _cancel):
        return AfterToolCallResult(terminate=True)

    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter, after_tool_call=after_tool_call)

    llm_calls = 0

    async def stream_fn(_model, _context, _options):
        nonlocal llm_calls
        llm_calls += 1
        return done_stream(
            create_assistant_message([ToolCall(id="tool-1", name="echo", arguments={"value": "hello"})], "toolUse"),
            "toolUse",
        )

    stream = agent_loop([create_user_message("echo something")], context, config, None, stream_fn)
    async for _event in stream:
        pass

    assert llm_calls == 1


@pytest.mark.tonio
async def test_continue_should_throw_when_context_has_no_messages():
    context = AgentContext(messages=[], tools=[])
    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter)

    async def stream_fn(_model, _context, _options):
        raise Exception("Unexpected stream call")

    with pytest.raises(Exception, match="Cannot continue: no messages in context"):
        agent_loop_continue(context, config, None, stream_fn)


@pytest.mark.tonio
async def test_continue_from_existing_context_without_emitting_user_message_events():
    user_message = create_user_message("Hello")
    context = AgentContext(messages=[user_message], tools=[])
    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter)

    async def stream_fn(_model, _context, _options):
        return done_stream(create_assistant_message([TextContent(text="Response")]))

    events = []
    stream = agent_loop_continue(context, config, None, stream_fn)
    async for event in stream:
        events.append(event)

    messages = await stream.result()

    # Should only return the new assistant message (not the existing user message).
    assert len(messages) == 1
    assert messages[0].role == "assistant"

    # Should NOT have user message events (the key difference from agent_loop).
    message_end_events = [e for e in events if e.type == "message_end"]
    assert len(message_end_events) == 1
    assert message_end_events[0].message.role == "assistant"


@dataclass
class CustomMessage:
    role: str
    text: str
    timestamp: int


@pytest.mark.tonio
async def test_continue_should_allow_custom_message_types_as_last_message():
    custom_message = CustomMessage(role="custom", text="Hook content", timestamp=int(time.time()))
    context = AgentContext(messages=[custom_message], tools=[])

    async def convert(messages):
        out = []
        for m in messages:
            if getattr(m, "role", None) == "custom":
                out.append(UserMessage(content=m.text, timestamp=m.timestamp))
            elif getattr(m, "role", None) in ("user", "assistant", "toolResult"):
                out.append(m)
        return out

    config = AgentLoopConfig(model=create_model(), convert_to_llm=convert)

    async def stream_fn(_model, _context, _options):
        return done_stream(create_assistant_message([TextContent(text="Response to custom message")]))

    # Should not raise - the custom message will be converted to a user message.
    stream = agent_loop_continue(context, config, None, stream_fn)

    events = []
    async for event in stream:
        events.append(event)

    messages = await stream.result()
    assert len(messages) == 1
    assert messages[0].role == "assistant"
