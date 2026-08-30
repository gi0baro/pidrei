"""Mirror of pi agent/test/agent-loop.test.ts."""

import time
from dataclasses import dataclass

import pytest
import tonio.colored as tonio

from pidrei_agent.agent_loop import agent_loop, agent_loop_continue
from pidrei_agent.stream_fn import set_default_stream_fn
from pidrei_agent.types import (
    AfterToolCallResult,
    AgentContext,
    AgentLoopConfig,
    AgentLoopTurnUpdate,
    AgentTool,
    AgentToolResult,
    BeforeToolCallResult,
)
from pidrei_ai.types import (
    AssistantMessage,
    DoneEvent,
    Model,
    ModelCost,
    TextContent,
    ToolCall,
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
    return [m for m in messages if getattr(m, "role", None) in ("user", "assistant", "toolResult")]


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
        context = AgentContext(system_prompt="", messages=[], tools=[])
        config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter)
        stream = agent_loop([create_user_message("Hello")], context, config, None)

        await stream.result()
        assert calls == 1
    finally:
        set_default_stream_fn(None)


@pytest.mark.tonio
async def test_should_emit_events_with_agent_message_types():
    context = AgentContext(system_prompt="You are helpful.", messages=[], tools=[])
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


@dataclass
class CustomNotification:
    role: str
    text: str
    timestamp: int


@pytest.mark.tonio
async def test_should_handle_custom_message_types_via_convert_to_llm():
    notification = CustomNotification(role="notification", text="This is a notification", timestamp=int(time.time()))
    context = AgentContext(system_prompt="You are helpful.", messages=[notification], tools=[])
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
        system_prompt="You are helpful.",
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
    context = AgentContext(system_prompt="", messages=[], tools=[tool])
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
    context = AgentContext(system_prompt="", messages=[], tools=[tool])
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
    context = AgentContext(system_prompt="", messages=[], tools=[tool])
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
    context = AgentContext(system_prompt="", messages=[], tools=[tool])
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
            await first_done.wait(None)
            first_resolved = True
        if params["value"] == "second" and not first_resolved:
            parallel_observed = True
        return AgentToolResult(
            content=[TextContent(text=f"echoed: {params['value']}")], details={"value": params["value"]}
        )

    tool = FnTool("echo", "Echo", "Echo tool", VALUE_SCHEMA, execute)
    context = AgentContext(system_prompt="", messages=[], tools=[tool])
    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter, tool_execution="parallel")

    async def release_first():
        await tonio.sleep(0.02)
        first_done.set()

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
            tonio.spawn.without_tracking(release_first())
        else:
            stream = done_stream(create_assistant_message([TextContent(text="done")]))
        call_index += 1
        return stream

    events = []
    stream = agent_loop([create_user_message("echo both")], context, config, None, stream_fn)
    async for event in stream:
        events.append(event)

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
    context = AgentContext(system_prompt="", messages=[], tools=[tool])
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
    first_done = tonio.Event()

    async def execute(_tool_call_id, params):
        nonlocal first_resolved, parallel_observed
        if params["value"] == "first":
            await first_done.wait(None)
            first_resolved = True
        if params["value"] == "second" and not first_resolved:
            parallel_observed = True
        return AgentToolResult(
            content=[TextContent(text=f"slow: {params['value']}")], details={"value": params["value"]}
        )

    slow_tool = FnTool("slow", "Slow", "Slow tool", VALUE_SCHEMA, execute, execution_mode="sequential")
    context = AgentContext(system_prompt="", messages=[], tools=[slow_tool])
    # config is parallel (default), but the tool forces sequential.
    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter)

    async def release_first():
        await tonio.sleep(0.02)
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

    tool_result_ids = [
        e.message.tool_call_id
        for e in events
        if e.type == "message_end" and getattr(e.message, "role", None) == "toolResult"
    ]
    assert tool_result_ids == ["tool-1", "tool-2"]


@pytest.mark.tonio
async def test_should_force_sequential_when_one_of_multiple_tools_has_sequential_mode():
    execution_order = []
    slow_done = tonio.Event()

    async def execute_slow(_tool_call_id, params):
        execution_order.append(f"slow:{params['value']}")
        if params["value"] == "a":
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
    context = AgentContext(system_prompt="", messages=[], tools=[slow_tool, fast_tool])
    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter)

    async def release_slow():
        await tonio.sleep(0.02)
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

    stream = agent_loop([create_user_message("run both")], context, config, None, stream_fn)
    async for _event in stream:
        pass

    # Fast tool should NOT run before slow tool finishes.
    assert execution_order[0] == "slow:a"
    assert "fast:b" in execution_order


@pytest.mark.tonio
async def test_should_allow_parallel_execution_when_all_tools_have_parallel_mode():
    first_resolved = False
    parallel_observed = False
    first_done = tonio.Event()

    async def execute(_tool_call_id, params):
        nonlocal first_resolved, parallel_observed
        if params["value"] == "first":
            await first_done.wait(None)
            first_resolved = True
        if params["value"] == "second" and not first_resolved:
            parallel_observed = True
        return AgentToolResult(
            content=[TextContent(text=f"echoed: {params['value']}")], details={"value": params["value"]}
        )

    tool = FnTool("echo", "Echo", "Echo tool", VALUE_SCHEMA, execute, execution_mode="parallel")
    context = AgentContext(system_prompt="", messages=[], tools=[tool])
    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter)

    async def release_first():
        await tonio.sleep(0.02)
        first_done.set()

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
            tonio.spawn.without_tracking(release_first())
        else:
            stream = done_stream(create_assistant_message([TextContent(text="done")]))
        call_index += 1
        return stream

    stream = agent_loop([create_user_message("echo both")], context, config, None, stream_fn)
    async for _event in stream:
        pass

    # With execution_mode="parallel", second tool should start before first finishes.
    assert parallel_observed is True


@pytest.mark.tonio
async def test_should_use_prepare_next_turn_snapshot_before_continuing():
    async def execute(_tool_call_id, params):
        return AgentToolResult(
            content=[TextContent(text=f"echoed: {params['value']}")], details={"value": params["value"]}
        )

    tool = FnTool("echo", "Echo", "Echo tool", VALUE_SCHEMA, execute)
    context = AgentContext(system_prompt="first prompt", messages=[], tools=[tool])
    converted_second_turn_system_prompt = ""
    prepare_calls = 0
    prepared = False

    async def prepare_next_turn(ctx):
        nonlocal prepare_calls, prepared
        prepare_calls += 1
        if prepared:
            return None
        prepared = True
        return AgentLoopTurnUpdate(
            context=AgentContext(
                system_prompt="second prompt",
                messages=list(ctx.context.messages),
                tools=ctx.context.tools,
            )
        )

    config = AgentLoopConfig(
        model=create_model(), convert_to_llm=identity_converter, prepare_next_turn=prepare_next_turn
    )

    llm_calls = 0

    async def stream_fn(_model, ctx, _options):
        nonlocal llm_calls, converted_second_turn_system_prompt
        llm_calls += 1
        if llm_calls == 2:
            converted_second_turn_system_prompt = ctx.system_prompt or ""
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
    assert converted_second_turn_system_prompt == "second prompt"


@pytest.mark.tonio
async def test_should_stop_after_the_current_turn_when_should_stop_after_turn_returns_true():
    executed = []

    async def execute(_tool_call_id, params):
        executed.append(params["value"])
        return AgentToolResult(
            content=[TextContent(text=f"echoed: {params['value']}")], details={"value": params["value"]}
        )

    tool = FnTool("echo", "Echo", "Echo tool", VALUE_SCHEMA, execute)
    context = AgentContext(system_prompt="", messages=[], tools=[tool])

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

    async def should_stop_after_turn(ctx):
        nonlocal callback_tool_result_ids, callback_context_roles
        assert ctx.message.role == "assistant"
        callback_tool_result_ids = [tool_result.tool_call_id for tool_result in ctx.tool_results]
        callback_context_roles = [getattr(m, "role", None) for m in ctx.context.messages]
        return True

    config = AgentLoopConfig(
        model=create_model(),
        convert_to_llm=identity_converter,
        get_steering_messages=get_steering_messages,
        get_follow_up_messages=get_follow_up_messages,
        should_stop_after_turn=should_stop_after_turn,
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
    assert callback_context_roles == ["user", "assistant", "toolResult"]
    assert [getattr(m, "role", None) for m in messages] == ["user", "assistant", "toolResult"]
    assert [event.type for event in events] == [
        "agent_start",
        "turn_start",
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
    context = AgentContext(system_prompt="", messages=[], tools=[tool])
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
    assert [getattr(m, "role", None) for m in messages] == ["user", "assistant", "toolResult"]
    assert len([event for event in events if event.type == "turn_end"]) == 1


@pytest.mark.tonio
async def test_should_stop_after_a_blocked_tool_call_when_before_tool_call_sets_terminate_true():
    executed = False

    async def execute(_tool_call_id, params):
        nonlocal executed
        executed = True
        return AgentToolResult(content=[TextContent(text="should not execute")], details={"value": "unexpected"})

    tool = FnTool("echo", "Echo", "Echo tool", VALUE_SCHEMA, execute)
    context = AgentContext(system_prompt="", messages=[], tools=[tool])

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
    context = AgentContext(system_prompt="", messages=[], tools=[tool])

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
    context = AgentContext(system_prompt="", messages=[], tools=[tool])
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
    context = AgentContext(system_prompt="", messages=[], tools=[tool])

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
    context = AgentContext(system_prompt="You are helpful.", messages=[], tools=[])
    config = AgentLoopConfig(model=create_model(), convert_to_llm=identity_converter)

    async def stream_fn(_model, _context, _options):
        raise Exception("Unexpected stream call")

    with pytest.raises(Exception, match="Cannot continue: no messages in context"):
        agent_loop_continue(context, config, None, stream_fn)


@pytest.mark.tonio
async def test_continue_from_existing_context_without_emitting_user_message_events():
    user_message = create_user_message("Hello")
    context = AgentContext(system_prompt="You are helpful.", messages=[user_message], tools=[])
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
    context = AgentContext(system_prompt="You are helpful.", messages=[custom_message], tools=[])

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
