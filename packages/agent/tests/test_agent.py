"""Mirror of pi agent/test/agent.test.ts."""

import time

import pytest
import tonio.colored as tonio

from pidrei_agent.agent import Agent, AgentInitialState
from pidrei_agent.stream_fn import set_default_stream_fn
from pidrei_agent.types import AgentTool, AgentToolResult
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import (
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    Model,
    ModelCost,
    StartEvent,
    TextContent,
    ToolCall,
    Usage,
    UserMessage,
)
from pidrei_ai.utils.cancel import CancelToken
from pidrei_ai.utils.event_stream import AssistantMessageEventStream


EMPTY_SCHEMA = {"type": "object", "properties": {}}


class FnTool(AgentTool):
    def __init__(self, name, label, description, parameters, execute, execution_mode=None):
        self.name = name
        self.label = label
        self.description = description
        self.parameters = parameters
        self.execution_mode = execution_mode
        self.prepare_arguments = None
        self._execute = execute

    async def execute(self, tool_call_id, params, cancel, on_update):
        return await self._execute(tool_call_id, params, cancel, on_update)


def create_assistant_message(text: str) -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text=text)],
        api="openai-responses",
        provider="openai",
        model="mock",
        usage=Usage(),
        stop_reason="stop",
        timestamp=int(time.time() * 1000),
    )


def create_assistant_tool_use_message(content: list[ToolCall]) -> AssistantMessage:
    return AssistantMessage(
        content=content,
        api="openai-responses",
        provider="openai",
        model="mock",
        usage=Usage(),
        stop_reason="toolUse",
        timestamp=int(time.time() * 1000),
    )


def unused_stream_fn(_model, _context, _options):
    raise Exception("Unexpected stream call")


def done_stream(message: AssistantMessage, reason: str = "stop") -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()
    stream.push(DoneEvent(reason=reason, message=message))
    return stream


def abort_responsive_stream_fn(_model, _context, options):
    """Stream that starts a partial and errors once the run's cancel fires."""
    stream = AssistantMessageEventStream()

    async def driver():
        stream.push(StartEvent(partial=create_assistant_message("")))
        while options is None or options.cancel is None or not options.cancel.cancelled:
            await tonio.sleep(0.005)
        stream.push(ErrorEvent(reason="aborted", error=create_assistant_message("Aborted")))

    tonio.spawn.without_tracking(driver())
    return stream


def custom_model(model_id: str) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="openai-responses",
        provider="custom",
        base_url="https://example.invalid",
        reasoning=False,
        input=["text"],
        cost=ModelCost(),
        context_window=8192,
        max_tokens=2048,
    )


@pytest.mark.tonio
async def test_uses_the_configured_default_when_a_legacy_caller_omits_stream_fn():
    calls = 0

    def default_fn(_model, _context, _options):
        nonlocal calls
        calls += 1
        return done_stream(create_assistant_message("fallback"))

    set_default_stream_fn(default_fn)
    try:
        agent = Agent()
        await agent.prompt("Hello")
        assert calls == 1
    finally:
        set_default_stream_fn(None)


@pytest.mark.tonio
async def test_should_create_an_agent_instance_with_default_state():
    agent = Agent(stream_fn=unused_stream_fn)

    assert agent.state is not None
    assert agent.state.system_prompt == ""
    assert agent.state.model is not None
    assert agent.state.thinking_level == "off"
    assert agent.state.tools == []
    assert agent.state.messages == []
    assert agent.state.is_streaming is False
    assert agent.state.streaming_message is None
    assert agent.state.pending_tool_calls == set()
    assert agent.state.error_message is None


@pytest.mark.tonio
async def test_should_create_an_agent_instance_with_custom_initial_state():
    model = get_builtin_model("openai", "gpt-4o-mini")
    assert model is not None
    agent = Agent(
        stream_fn=unused_stream_fn,
        initial_state=AgentInitialState(
            system_prompt="You are a helpful assistant.",
            model=model,
            thinking_level="low",
        ),
    )

    assert agent.state.system_prompt == "You are a helpful assistant."
    assert agent.state.model is model
    assert agent.state.thinking_level == "low"


@pytest.mark.tonio
async def test_should_subscribe_to_events():
    agent = Agent(stream_fn=unused_stream_fn)

    event_count = 0

    def listener(_event, _signal):
        nonlocal event_count
        event_count += 1

    unsubscribe = agent.subscribe(listener)

    # No initial event on subscribe.
    assert event_count == 0

    # State mutators don't emit events.
    agent.state.system_prompt = "Test prompt"
    assert event_count == 0
    assert agent.state.system_prompt == "Test prompt"

    # Unsubscribe should work.
    unsubscribe()
    agent.state.system_prompt = "Another prompt"
    assert event_count == 0


@pytest.mark.tonio
async def test_emits_full_lifecycle_events_for_thrown_run_failures():
    def exploding_stream_fn(_model, _context, _options):
        raise Exception("provider exploded")

    agent = Agent(stream_fn=exploding_stream_fn)
    events = []
    agent.subscribe(lambda event, _signal: events.append(event.type))

    await agent.prompt("hello")

    assert events == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "message_start",
        "message_end",
        "turn_end",
        "agent_end",
    ]
    last_message = agent.state.messages[-1]
    assert last_message.role == "assistant"
    assert last_message.stop_reason == "error"
    assert last_message.error_message == "provider exploded"
    assert agent.state.error_message == "provider exploded"


@pytest.mark.tonio
async def test_should_await_async_subscribers_before_prompt_resolves():
    barrier = tonio.Event()
    agent = Agent(stream_fn=lambda _model, _context, _options: done_stream(create_assistant_message("ok")))

    listener_finished = False

    async def listener(event, _signal):
        nonlocal listener_finished
        if event.type == "agent_end":
            await barrier.wait(None)
            listener_finished = True

    agent.subscribe(listener)

    prompt_resolved = False

    async def run_prompt():
        nonlocal prompt_resolved
        await agent.prompt("hello")
        prompt_resolved = True

    handle = tonio.spawn(run_prompt())

    await tonio.sleep(0.01)
    assert prompt_resolved is False
    assert listener_finished is False
    assert agent.state.is_streaming is True

    barrier.set()
    await handle

    assert listener_finished is True
    assert prompt_resolved is True
    assert agent.state.is_streaming is False


@pytest.mark.tonio
async def test_wait_for_idle_should_wait_for_async_subscribers():
    barrier = tonio.Event()
    agent = Agent(stream_fn=lambda _model, _context, _options: done_stream(create_assistant_message("ok")))

    async def listener(event, _signal):
        if event.type == "message_end" and getattr(event.message, "role", None) == "assistant":
            await barrier.wait(None)

    agent.subscribe(listener)

    prompt_handle = tonio.spawn(agent.prompt("hello"))
    idle_resolved = False

    async def run_idle():
        nonlocal idle_resolved
        # Give the prompt a beat to register the active run first.
        await tonio.sleep(0.001)
        await agent.wait_for_idle()
        idle_resolved = True

    idle_handle = tonio.spawn(run_idle())

    await tonio.sleep(0.01)
    assert idle_resolved is False
    assert agent.state.is_streaming is True

    barrier.set()
    await prompt_handle
    await idle_handle

    assert idle_resolved is True
    assert agent.state.is_streaming is False


@pytest.mark.tonio
async def test_should_pass_the_active_abort_signal_to_subscribers():
    received_signal = None
    agent = Agent(stream_fn=abort_responsive_stream_fn)

    def listener(event, signal):
        nonlocal received_signal
        if event.type == "agent_start":
            received_signal = signal

    agent.subscribe(listener)

    handle = tonio.spawn(agent.prompt("hello"))
    await tonio.sleep(0.01)

    assert received_signal is not None
    assert received_signal.cancelled is False

    agent.abort()
    await handle

    assert received_signal.cancelled is True


@pytest.mark.tonio
async def test_should_ignore_tool_updates_after_the_tool_execution_settles():
    delayed_update = None
    events = []

    async def execute(_tool_call_id, _params, _cancel, on_update):
        nonlocal delayed_update
        delayed_update = on_update
        on_update(AgentToolResult(content=[TextContent(text="running")], details={"status": "running"}))
        return AgentToolResult(content=[TextContent(text="ok")], details={"status": "done"}, terminate=True)

    tool = FnTool("delayed_tool", "Delayed Tool", "Captures progress callbacks", EMPTY_SCHEMA, execute)

    def stream_fn(_model, _context, _options):
        return done_stream(
            create_assistant_tool_use_message([ToolCall(id="call-1", name="delayed_tool", arguments={})]),
            "toolUse",
        )

    agent = Agent(initial_state=AgentInitialState(tools=[tool]), stream_fn=stream_fn)
    agent.subscribe(lambda event, _signal: events.append(event))

    await agent.prompt("run tool")
    event_count_after_prompt = len(events)

    delayed_update(AgentToolResult(content=[TextContent(text="late")], details={"status": "late"}))
    await tonio.sleep(0.005)

    assert len([event for event in events if event.type == "tool_execution_update"]) == 1
    assert len(events) == event_count_after_prompt


@pytest.mark.tonio
async def test_should_ignore_a_settled_parallel_tool_update_while_another_tool_is_still_running():
    slow_started = tonio.Event()
    settled_tool_ended = tonio.Event()
    release_slow = tonio.Event()
    settled_tool_update = None
    events = []

    async def execute_settled(_tool_call_id, _params, _cancel, on_update):
        nonlocal settled_tool_update
        settled_tool_update = on_update
        return AgentToolResult(content=[TextContent(text="done")], details={"status": "done"}, terminate=True)

    async def execute_slow(_tool_call_id, _params, _cancel, _on_update):
        slow_started.set()
        await release_slow.wait(None)
        return AgentToolResult(content=[TextContent(text="done")], details={"status": "done"}, terminate=True)

    settled_tool = FnTool("settled_tool", "Settled Tool", "Captures progress callbacks", EMPTY_SCHEMA, execute_settled)
    slow_tool = FnTool("slow_tool", "Slow Tool", "Keeps the agent run active", EMPTY_SCHEMA, execute_slow)

    def stream_fn(_model, _context, _options):
        return done_stream(
            create_assistant_tool_use_message(
                [
                    ToolCall(id="call-1", name="settled_tool", arguments={}),
                    ToolCall(id="call-2", name="slow_tool", arguments={}),
                ]
            ),
            "toolUse",
        )

    agent = Agent(initial_state=AgentInitialState(tools=[settled_tool, slow_tool]), stream_fn=stream_fn)

    def listener(event, _signal):
        events.append(event)
        if event.type == "tool_execution_end" and event.tool_call_id == "call-1":
            settled_tool_ended.set()

    agent.subscribe(listener)

    handle = tonio.spawn(agent.prompt("run tools"))
    await slow_started.wait(None)
    await settled_tool_ended.wait(None)
    event_count_before_late_update = len(events)

    settled_tool_update(AgentToolResult(content=[TextContent(text="late")], details={"status": "late"}))
    await tonio.sleep(0.005)
    assert len(events) == event_count_before_late_update

    release_slow.set()
    await handle
    assert len([event for event in events if event.type == "tool_execution_update"]) == 0


@pytest.mark.tonio
async def test_should_update_state_with_mutators():
    agent = Agent(stream_fn=unused_stream_fn)

    agent.state.system_prompt = "Custom prompt"
    assert agent.state.system_prompt == "Custom prompt"

    new_model = custom_model("gemini-2.5-flash")
    agent.state.model = new_model
    assert agent.state.model is new_model

    agent.state.thinking_level = "high"
    assert agent.state.thinking_level == "high"

    tools = [FnTool("test", "test", "test tool", EMPTY_SCHEMA, None)]
    agent.state.tools = tools
    assert agent.state.tools == tools
    assert agent.state.tools is not tools  # Should be a copy.

    messages = [UserMessage(content="Hello", timestamp=int(time.time() * 1000))]
    agent.state.messages = messages
    assert agent.state.messages == messages
    assert agent.state.messages is not messages  # Should be a copy.

    new_message = create_assistant_message("Hi")
    agent.state.messages.append(new_message)
    assert len(agent.state.messages) == 2
    assert agent.state.messages[1] is new_message

    agent.state.messages = []
    assert agent.state.messages == []


@pytest.mark.tonio
async def test_should_support_steering_message_queue():
    agent = Agent(stream_fn=unused_stream_fn)

    message = UserMessage(content="Steering message", timestamp=int(time.time() * 1000))
    agent.steer(message)

    # The message is queued but not yet in state.messages.
    assert message not in agent.state.messages


@pytest.mark.tonio
async def test_should_support_follow_up_message_queue():
    agent = Agent(stream_fn=unused_stream_fn)

    message = UserMessage(content="Follow-up message", timestamp=int(time.time() * 1000))
    agent.follow_up(message)

    assert message not in agent.state.messages


@pytest.mark.tonio
async def test_should_handle_abort_controller():
    agent = Agent(stream_fn=unused_stream_fn)

    # Should not raise even if nothing is running.
    agent.abort()


@pytest.mark.tonio
async def test_should_throw_when_prompt_called_while_streaming():
    agent = Agent(stream_fn=abort_responsive_stream_fn)

    handle = tonio.spawn(agent.prompt("First message"))

    await tonio.sleep(0.01)
    assert agent.state.is_streaming is True

    with pytest.raises(Exception, match="Agent is already processing a prompt"):
        await agent.prompt("Second message")

    agent.abort()
    await handle


@pytest.mark.tonio
async def test_should_throw_when_continue_called_while_streaming():
    agent = Agent(stream_fn=abort_responsive_stream_fn)

    handle = tonio.spawn(agent.prompt("First message"))
    await tonio.sleep(0.01)
    assert agent.state.is_streaming is True

    with pytest.raises(Exception, match="Agent is already processing. Wait for completion before continuing."):
        await agent.continue_()

    agent.abort()
    await handle


@pytest.mark.tonio
async def test_continue_should_process_queued_follow_up_messages_after_an_assistant_turn():
    agent = Agent(stream_fn=lambda _model, _context, _options: done_stream(create_assistant_message("Processed")))

    agent.state.messages = [
        UserMessage(content=[TextContent(text="Initial")], timestamp=int(time.time() * 1000) - 10),
        create_assistant_message("Initial response"),
    ]

    agent.follow_up(UserMessage(content=[TextContent(text="Queued follow-up")], timestamp=int(time.time() * 1000)))

    await agent.continue_()

    has_queued_follow_up = any(
        getattr(message, "role", None) == "user"
        and not isinstance(message.content, str)
        and any(part.type == "text" and part.text == "Queued follow-up" for part in message.content)
        for message in agent.state.messages
    )

    assert has_queued_follow_up is True
    assert agent.state.messages[-1].role == "assistant"


@pytest.mark.tonio
async def test_continue_should_keep_one_at_a_time_steering_semantics_from_assistant_tail():
    response_count = 0

    def stream_fn(_model, _context, _options):
        nonlocal response_count
        response_count += 1
        return done_stream(create_assistant_message(f"Processed {response_count}"))

    agent = Agent(stream_fn=stream_fn)

    agent.state.messages = [
        UserMessage(content=[TextContent(text="Initial")], timestamp=int(time.time() * 1000) - 10),
        create_assistant_message("Initial response"),
    ]

    agent.steer(UserMessage(content=[TextContent(text="Steering 1")], timestamp=int(time.time() * 1000)))
    agent.steer(UserMessage(content=[TextContent(text="Steering 2")], timestamp=int(time.time() * 1000) + 1))

    await agent.continue_()

    recent_messages = agent.state.messages[-4:]
    assert [getattr(m, "role", None) for m in recent_messages] == ["user", "assistant", "user", "assistant"]
    assert response_count == 2


@pytest.mark.tonio
async def test_keeps_legacy_prepare_next_turn_signal_callback_behavior():
    async def execute(_tool_call_id, _params, _cancel, _on_update):
        return AgentToolResult(content=[TextContent(text="ok")], details={})

    tool = FnTool("noop", "Noop", "Noop tool", EMPTY_SCHEMA, execute)
    request_count = 0
    saw_cancel_token = False

    def stream_fn(_model, _context, _options):
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            return done_stream(
                create_assistant_tool_use_message([ToolCall(id="tool-1", name="noop", arguments={})]), "toolUse"
            )
        return done_stream(create_assistant_message("done"))

    async def prepare_next_turn(signal):
        nonlocal saw_cancel_token
        saw_cancel_token = isinstance(signal, CancelToken)

    agent = Agent(
        initial_state=AgentInitialState(tools=[tool]),
        prepare_next_turn=prepare_next_turn,
        stream_fn=stream_fn,
    )

    await agent.prompt("start")

    assert request_count == 2
    assert saw_cancel_token is True


@pytest.mark.tonio
async def test_forwards_session_id_to_stream_function_options():
    received_session_id = None

    def stream_fn(_model, _context, options):
        nonlocal received_session_id
        received_session_id = options.session_id if options is not None else None
        return done_stream(create_assistant_message("ok"))

    agent = Agent(session_id="session-abc", stream_fn=stream_fn)

    await agent.prompt("hello")
    assert received_session_id == "session-abc"

    # Test setter.
    agent.session_id = "session-def"
    assert agent.session_id == "session-def"

    await agent.prompt("hello again")
    assert received_session_id == "session-def"
