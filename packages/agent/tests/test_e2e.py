"""Mirror of pi agent/test/e2e.test.ts (Agent + faux provider through the full stack).

pi drives the compat global `streamSimple`; pidrei passes the explicit
`models.stream_simple` bound method as the agent's stream function.
"""

import time

import pytest
import tonio.colored as tonio

from pidrei_agent.agent import Agent, AgentInitialState
from pidrei_agent.types import AgentTool
from pidrei_ai.providers.faux import (
    FauxModelDefinition,
    faux_assistant_message,
    faux_provider,
    faux_text,
    faux_thinking,
    faux_tool_call,
)
from pidrei_ai.registry import create_models
from pidrei_ai.types import (
    AssistantMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from tests.harness_tool_fixtures import CALCULATE_SCHEMA, calculate


models = create_models()
_faux_count = 0


def new_faux(**options):
    global _faux_count
    _faux_count += 1
    faux = faux_provider(provider=f"faux-e2e-{_faux_count}", **options)
    models.set_provider(faux.provider)
    return faux


class CalculateAgentTool(AgentTool):
    """Agent-level calculate tool (4-arg execute; mirror of test/utils/calculate.ts)."""

    name = "calculate"
    label = "Calculator"
    description = "Evaluate mathematical expressions"
    parameters = CALCULATE_SCHEMA

    async def execute(self, tool_call_id, params, cancel, on_update):
        return calculate(params["expression"])


calculate_tool = CalculateAgentTool()


def get_text_content(message) -> str:
    return "\n".join(block.text for block in message.content if getattr(block, "type", None) == "text")


async def stream_fn(model, context, options=None):
    return models.stream_simple(model, context, options)


def create_agent(model, system_prompt: str, tools=None, thinking_level="off") -> Agent:
    return Agent(
        stream_fn=stream_fn,
        initial_state=AgentInitialState(
            system_prompt=system_prompt,
            model=model,
            thinking_level=thinking_level,
            tools=tools if tools is not None else [],
        ),
    )


@pytest.mark.tonio
async def test_handles_a_basic_text_prompt():
    faux = new_faux()
    faux.set_responses([faux_assistant_message("4")])
    agent = create_agent(faux.get_model(), "You are a helpful assistant. Keep your responses concise.")

    await agent.prompt("What is 2+2? Answer with just the number.")

    assert agent.state.is_streaming is False
    assert len(agent.state.messages) == 2
    assert agent.state.messages[0].role == "user"
    assert agent.state.messages[1].role == "assistant"
    assert "4" in get_text_content(agent.state.messages[1])


@pytest.mark.tonio
async def test_executes_tools_and_tracks_pending_tool_calls():
    faux = new_faux()
    faux.set_responses(
        [
            faux_assistant_message(
                [
                    faux_text("Let me calculate that."),
                    faux_tool_call("calculate", {"expression": "123 * 456"}, id="calc-1"),
                ],
                stop_reason="toolUse",
            ),
            faux_assistant_message("The result is 56088."),
        ]
    )
    agent = create_agent(
        faux.get_model(), "You are a helpful assistant. Always use the calculator tool for math.", [calculate_tool]
    )

    pending_during_events: list[tuple[str, list[str]]] = []

    async def listener(event, _signal):
        if event.type in ("tool_execution_start", "tool_execution_end"):
            pending_during_events.append((event.type, sorted(agent.state.pending_tool_calls)))

    agent.subscribe(listener)

    await agent.prompt("Calculate 123 * 456 using the calculator tool.")

    assert agent.state.is_streaming is False
    assert len(agent.state.messages) >= 4
    tool_result_msg = next((m for m in agent.state.messages if getattr(m, "role", None) == "toolResult"), None)
    assert tool_result_msg is not None
    assert "123 * 456 = 56088" in get_text_content(tool_result_msg)

    final_message = agent.state.messages[-1]
    assert final_message.role == "assistant"
    assert "56088" in get_text_content(final_message)
    assert len(agent.state.pending_tool_calls) == 0
    assert pending_during_events == [("tool_execution_start", ["calc-1"]), ("tool_execution_end", [])]


@pytest.mark.tonio
async def test_handles_abort_during_streaming():
    faux = new_faux(tokens_per_second=20, token_size_min=2, token_size_max=2)
    faux.set_responses(
        [
            faux_assistant_message(
                "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen"
            )
        ]
    )
    agent = create_agent(faux.get_model(), "You are a helpful assistant.")

    prompt_handle = tonio.spawn(agent.prompt("Count slowly from 1 to 20."))

    async def abort_later():
        await tonio.sleep(0.03)
        agent.abort()

    tonio.spawn.without_tracking(abort_later())

    await prompt_handle

    assert agent.state.is_streaming is False
    assert len(agent.state.messages) >= 2

    last_message = agent.state.messages[-1]
    assert last_message.role == "assistant"
    assert last_message.stop_reason == "aborted"
    assert last_message.error_message is not None
    assert agent.state.error_message == last_message.error_message


@pytest.mark.tonio
async def test_emits_lifecycle_updates_while_streaming():
    faux = new_faux(token_size_min=1, token_size_max=1)
    faux.set_responses([faux_assistant_message("1 2 3 4 5")])
    agent = create_agent(faux.get_model(), "You are a helpful assistant.")

    events: list[str] = []

    async def record_event(event, _signal):
        events.append(event.type)

    agent.subscribe(record_event)

    await agent.prompt("Count from 1 to 5.")

    for expected in (
        "agent_start",
        "turn_start",
        "message_start",
        "message_update",
        "message_end",
        "turn_end",
        "agent_end",
    ):
        assert expected in events
    assert events.index("agent_start") < events.index("message_start")
    assert events.index("message_start") < events.index("message_end")
    assert events.index("message_end") < (len(events) - 1 - events[::-1].index("agent_end"))

    assert agent.state.is_streaming is False
    assert len(agent.state.messages) == 2


@pytest.mark.tonio
async def test_maintains_context_across_multiple_turns():
    faux = new_faux()

    async def second_response(context, _options, _state, _model):
        has_alice = any(
            getattr(message, "role", None) == "user"
            and (
                "Alice" in message.content
                if isinstance(message.content, str)
                else any(getattr(block, "type", None) == "text" and "Alice" in block.text for block in message.content)
            )
            for message in context.messages
        )
        return faux_assistant_message("Your name is Alice." if has_alice else "I do not know your name.")

    faux.set_responses([faux_assistant_message("Nice to meet you, Alice."), second_response])
    agent = create_agent(faux.get_model(), "You are a helpful assistant.")

    await agent.prompt("My name is Alice.")
    assert len(agent.state.messages) == 2

    await agent.prompt("What is my name?")
    assert len(agent.state.messages) == 4

    last_message = agent.state.messages[3]
    assert last_message.role == "assistant"
    assert "alice" in get_text_content(last_message).lower()


@pytest.mark.tonio
async def test_preserves_thinking_content_blocks():
    faux = new_faux(models=[FauxModelDefinition(id="faux-reasoning", reasoning=True)])
    faux.set_responses([faux_assistant_message([faux_thinking("step by step"), faux_text("4")])])

    agent = create_agent(faux.get_model(), "You are a helpful assistant.", thinking_level="low")

    await agent.prompt("What is 2+2?")

    assistant_message = agent.state.messages[1]
    assert assistant_message.role == "assistant"
    assert assistant_message.content == [ThinkingContent(thinking="step by step"), TextContent(text="4")]


@pytest.mark.tonio
async def test_continue_throws_when_no_messages_in_context():
    faux = new_faux()
    agent = create_agent(faux.get_model(), "Test")

    with pytest.raises(Exception, match="No messages to continue from"):
        await agent.continue_()


@pytest.mark.tonio
async def test_continue_throws_when_last_message_is_assistant():
    faux = new_faux()
    model = faux.get_model()
    agent = create_agent(model, "Test")

    assistant_message = AssistantMessage(
        content=[TextContent(text="Hello")],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="stop",
        timestamp=int(time.time() * 1000),
    )
    agent.state.messages = [assistant_message]

    with pytest.raises(Exception, match="Cannot continue from message role: assistant"):
        await agent.continue_()


@pytest.mark.tonio
async def test_continue_continues_and_gets_a_response_when_last_message_is_user():
    faux = new_faux()
    faux.set_responses([faux_assistant_message("HELLO WORLD")])
    agent = create_agent(faux.get_model(), "You are a helpful assistant. Follow instructions exactly.")

    user_message = UserMessage(
        content=[TextContent(text="Say exactly: HELLO WORLD")], timestamp=int(time.time() * 1000)
    )
    agent.state.messages = [user_message]

    await agent.continue_()

    assert agent.state.is_streaming is False
    assert len(agent.state.messages) == 2
    assert agent.state.messages[0].role == "user"
    assert agent.state.messages[1].role == "assistant"
    assert "HELLO WORLD" in get_text_content(agent.state.messages[1]).upper()


@pytest.mark.tonio
async def test_continue_continues_and_processes_tool_results():
    faux = new_faux()
    model = faux.get_model()
    faux.set_responses([faux_assistant_message("The answer is 8.")])
    agent = create_agent(
        model,
        "You are a helpful assistant. After getting a calculation result, state the answer clearly.",
        [calculate_tool],
    )

    user_message = UserMessage(content=[TextContent(text="What is 5 + 3?")], timestamp=int(time.time() * 1000))

    assistant_message = AssistantMessage(
        content=[
            TextContent(text="Let me calculate that."),
            ToolCall(id="calc-1", name="calculate", arguments={"expression": "5 + 3"}),
        ],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="toolUse",
        timestamp=int(time.time() * 1000),
    )

    tool_result = ToolResultMessage(
        tool_call_id="calc-1",
        tool_name="calculate",
        content=[TextContent(text="5 + 3 = 8")],
        is_error=False,
        timestamp=int(time.time() * 1000),
    )

    agent.state.messages = [user_message, assistant_message, tool_result]

    await agent.continue_()

    assert agent.state.is_streaming is False
    assert len(agent.state.messages) >= 4

    last_message = agent.state.messages[-1]
    assert last_message.role == "assistant"
    assert "8" in get_text_content(last_message)
