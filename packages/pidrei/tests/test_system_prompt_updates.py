"""Mirror of pi coding-agent test/system-prompt-updates.test.ts.

pi's faux response callables take `(context)`; pidrei's take
`(context, options, state, model)`, so each one absorbs the extra arguments.
"""

import json
import os

import pytest

from pidrei.core.extensions import ToolDefinition
from pidrei.core.sdk import CreateAgentSessionOptions, create_agent_session
from pidrei.core.session_manager import SessionManager
from pidrei.core.settings_manager import SettingsManager
from pidrei.core.system_prompt import (
    BuildSystemPromptOptions,
    SystemPromptState,
    build_system_prompt_sections,
    build_system_prompt_state,
    diff_system_prompt_sections,
)
from pidrei_agent.harness.session.serde import parse_message, serialize_message
from pidrei_agent.types import AgentToolResult
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.providers.faux import faux_assistant_message, faux_tool_call
from pidrei_ai.types import SystemMessage, TextContent, Tool, ToolReference, UserMessage
from pidrei_ai.utils.text import get_system_message_text
from pidrei_ai.utils.transcript import get_current_system_message, get_current_system_prompt

from .harness import create_harness


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


def _recording(requests: list, response):
    async def respond(context, *_rest):
        requests.append(context)
        return response

    return respond


def _system_messages(context) -> list[SystemMessage]:
    return [message for message in context.messages if message.role == "system"]


@pytest.mark.tonio
async def test_declares_the_prompt_and_tools_once_and_reuses_them_across_resume(harnesses):
    harness = await create_harness()
    harnesses.append(harness)
    harness.set_responses([faux_assistant_message("first"), faux_assistant_message("second")])
    await harness.session.prompt("one")
    await harness.session.prompt("two")

    system_entries = [
        entry
        for entry in harness.session_manager.get_entries()
        if entry["type"] == "message" and entry["message"].role == "system"
    ]
    assert len(system_entries) == 1
    assert [message.role for message in harness.session.messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    head = harness.session.messages[0]
    assert head.role == "system"
    assert head.content == ""
    assert list(head.sections or {}) == ["preamble", "tools", "rules", "docs", "cwd"]
    assert [tool.name for tool in head.tools_added or []] == ["read", "bash", "edit", "write"]
    assert get_system_message_text(head) == harness.session.system_prompt


@pytest.mark.tonio
async def test_opens_a_transcript_without_a_system_message_and_declares_the_prompt_on_the_first_request(tmp_path):
    session_manager = SessionManager.in_memory(str(tmp_path))
    await session_manager.append_message(UserMessage(content="existing", timestamp=1))
    created = await create_agent_session(
        CreateAgentSessionOptions(
            cwd=str(tmp_path),
            agent_dir=os.path.join(str(tmp_path), "agent"),
            model=get_builtin_model("anthropic", "claude-sonnet-4-5"),
            settings_manager=SettingsManager.in_memory(),
            session_manager=session_manager,
            no_tools="all",
        )
    )
    try:
        # Nothing is synthesized or persisted until a request needs it.
        assert [message.role for message in created.session.messages] == ["user"]
        assert [message.role for message in session_manager.build_session_context().messages] == ["user"]
        assert get_current_system_message(created.session.messages) is None
    finally:
        created.session.dispose()


def test_diffs_sections_into_a_patch():
    previous = build_system_prompt_sections(BuildSystemPromptOptions(cwd="/tmp", sections={"plan_mode": "Plan only."}))
    current = build_system_prompt_sections(
        BuildSystemPromptOptions(cwd="/tmp", sections={"plan_mode": "Implementation allowed."})
    )
    assert diff_system_prompt_sections(previous, current) == {
        "plan_mode": "<plan_mode>\nImplementation allowed.\n</plan_mode>"
    }
    assert diff_system_prompt_sections(previous, previous) is None
    assert diff_system_prompt_sections(
        previous, build_system_prompt_sections(BuildSystemPromptOptions(cwd="/tmp"))
    ) == {"plan_mode": None}


def test_keeps_the_preamble_untagged_and_replaces_it_like_any_section():
    previous = build_system_prompt_sections(BuildSystemPromptOptions(custom_prompt="You are A.", cwd="/tmp"))
    current = build_system_prompt_sections(BuildSystemPromptOptions(custom_prompt="You are B.", cwd="/tmp"))
    assert previous["preamble"] == "You are A."
    assert diff_system_prompt_sections(previous, current) == {"preamble": "You are B."}

    assert build_system_prompt_state(
        BuildSystemPromptOptions(force_system_prompt="Exact prompt.", cwd="/tmp")
    ) == SystemPromptState(content="Exact prompt.")
    assert build_system_prompt_state(BuildSystemPromptOptions(cwd="/tmp")) == SystemPromptState(
        content="", sections=build_system_prompt_sections(BuildSystemPromptOptions(cwd="/tmp"))
    )
    with pytest.raises(Exception, match="Invalid system prompt section name"):
        build_system_prompt_sections(BuildSystemPromptOptions(cwd="/tmp", sections={"preamble": "x"}))


@pytest.mark.tonio
async def test_a_forced_prompt_is_sent_as_the_leading_prompt_for_the_run_and_never_recorded(harnesses):
    turn = 0

    async def extension(pi) -> None:
        async def on_before_agent_start(event, _ctx):
            nonlocal turn
            turn += 1
            if turn == 3:
                event["systemPromptOptions"].sections["plan_mode"] = "Plan only."
            return {"systemPrompt": "Exact prompt."} if turn in (2, 3) else None

        pi.on("before_agent_start", on_before_agent_start)

    harness = await create_harness(extension_factories=[extension])
    harnesses.append(harness)
    requests: list = []
    harness.set_responses(
        [_recording(requests, faux_assistant_message(text)) for text in ("one", "two", "three", "four")]
    )
    for text in ("one", "two", "three", "four"):
        await harness.session.prompt(text)

    system_messages = [_system_messages(request) for request in requests]
    # Forced turns collapse to one leading message; the unforced fourth turn passes the
    # recorded head and both plan_mode patches through.
    assert [len(messages) for messages in system_messages] == [1, 1, 1, 3]

    forced = system_messages[1][-1]
    assert forced == SystemMessage(
        content="Exact prompt.",
        tools_added=system_messages[0][0].tools_added,
        timestamp=system_messages[0][0].timestamp,
    )
    assert system_messages[2][-1] == forced
    assert get_current_system_prompt(requests[2].messages) == "Exact prompt."
    assert [message.role for message in requests[2].messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]

    # The transcript only records the structured sections, never the forced text.
    recorded = [message.sections for message in harness.session.messages if message.role == "system"]
    assert recorded == [
        system_messages[0][0].sections,
        {"plan_mode": "<plan_mode>\nPlan only.\n</plan_mode>"},
        {"plan_mode": None},
    ]
    assert get_current_system_prompt(harness.session.messages) == harness.session.system_prompt


def _register_named_tools(pi, *, with_prompt: bool) -> None:
    for name in ("first", "second"):

        async def execute(*_args, name=name):
            return AgentToolResult(content=[TextContent(text=name)], details={})

        pi.register_tool(
            ToolDefinition(
                name=name,
                label=name,
                description=f"{name} description",
                prompt_snippet=f"{name} prompt snippet" if with_prompt else None,
                prompt_guidelines=[f"Use {name} carefully."] if with_prompt else None,
                parameters={"type": "object", "properties": {}},
                execute=execute,
            )
        )


@pytest.mark.tonio
async def test_set_active_tools_emits_prompt_sections_and_tool_changes_before_the_next_request(harnesses):
    async def extension(pi) -> None:
        _register_named_tools(pi, with_prompt=True)

    harness = await create_harness(extension_factories=[extension], initial_active_tool_names=["first"])
    harnesses.append(harness)
    # Faux response callbacks swallow raised assertions, so capture and assert afterwards.
    requests: list = []
    harness.set_responses(
        [
            _recording(requests, faux_assistant_message("first")),
            _recording(requests, faux_assistant_message([faux_tool_call("first", {})], stop_reason="toolUse")),
            _recording(requests, faux_assistant_message("second")),
        ]
    )
    await harness.session.prompt("first")
    harness.session.set_active_tools_by_name(["second"])
    await harness.session.prompt("second")
    assert len(requests) == 3

    initial = requests[0].messages[0]
    assert initial.role == "system"
    assert [value.name for value in initial.tools_added or []] == ["first", "second"]
    assert "first prompt snippet" in (initial.sections or {})["tools"]

    update = _system_messages(requests[1])[-1]
    assert update.content == ""
    assert list(update.sections or {}) == ["tools", "rules"]
    assert "second prompt snippet" in update.sections["tools"]
    assert "first prompt snippet" not in update.sections["tools"]
    assert "Use first carefully." not in update.sections["rules"]
    assert update.tools_added is None
    assert update.tools_removed == [ToolReference(name="first")]
    assert isinstance(update.timestamp, int)

    result = [message for message in requests[2].messages if message.role == "toolResult"][-1]
    assert result.tool_name == "first"
    assert result.is_error is True

    current = get_current_system_message(harness.session.messages)
    assert current is not None
    assert [value.name for value in current.tools_added or []] == ["second"]
    assert get_system_message_text(current) == harness.session.system_prompt


@pytest.mark.tonio
async def test_set_active_tools_in_before_agent_start_controls_the_same_request(harnesses):
    async def extension(pi) -> None:
        _register_named_tools(pi, with_prompt=False)
        turn = 0

        async def on_before_agent_start(_event, _ctx):
            nonlocal turn
            if turn == 1:
                pi.set_active_tools(["second"])
            turn += 1

        pi.on("before_agent_start", on_before_agent_start)

    harness = await create_harness(extension_factories=[extension], initial_active_tool_names=["first"])
    harnesses.append(harness)
    requests: list = []
    harness.set_responses(
        [_recording(requests, faux_assistant_message("first")), _recording(requests, faux_assistant_message("second"))]
    )
    await harness.session.prompt("first")
    await harness.session.prompt("second")
    assert len(requests) == 2
    update = _system_messages(requests[1])[-1]
    assert update.tools_removed == [ToolReference(name="first")]
    assert update.tools_added is None
    assert harness.session.get_active_tool_names() == ["second"]


@pytest.mark.tonio
async def test_keeps_tool_declarations_stable_across_a_session_json_round_trip(harnesses):
    async def execute(*_args):
        return AgentToolResult(content=[], details={})

    executable_tool = ToolDefinition(
        name="plain",
        label="Plain",
        description="Plain tool",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )
    harness = await create_harness(tools=[executable_tool], initial_active_tool_names=["plain"])
    harnesses.append(harness)
    harness.set_responses([faux_assistant_message("first"), faux_assistant_message("second")])
    await harness.session.prompt("one")
    head = harness.session.messages[0]
    assert head.role == "system"
    declaration = (head.tools_added or [None])[0]
    assert declaration is not None
    # A plain declaration: no constrained sampling and nothing executable.
    assert type(declaration) is Tool
    assert declaration.constrained_sampling is None

    # Simulate a resume: the persisted JSON must replay to the same declarations.
    harness.session.agent.state.messages = [
        parse_message(json.loads(json.dumps(serialize_message(message)))) for message in harness.session.messages
    ]
    await harness.session.prompt("two")
    assert len([message for message in harness.session.messages if message.role == "system"]) == 1
