"""Mirror of pi's suite/regressions/9789-context-handler-system-messages.test.ts."""

import dataclasses

import pytest

from pidrei.core.compaction import CompactionResult
from pidrei_ai.providers.faux import faux_assistant_message
from pidrei_ai.types import SystemMessage, TextContent, UserMessage
from pidrei_ai.utils.transcript import get_current_system_prompt, get_current_tools

from .harness import create_harness


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


def compact_via_hook(pi) -> None:
    """Compaction supplied by an extension hook, as in the reported sessions."""

    async def on_before_compact(event, _ctx):
        preparation = event["preparation"]
        return {
            "compaction": CompactionResult(
                summary="extension summary",
                first_kept_entry_id=preparation.first_kept_entry_id,
                tokens_before=preparation.tokens_before,
                details={"source": "test"},
            )
        }

    pi.on("session_before_compact", on_before_compact)


def capture_request(harness, text: str):
    captured: list = []

    async def respond(context, *_rest):
        captured.append(context)
        return faux_assistant_message(text)

    harness.set_responses([respond])

    def get_request():
        if not captured:
            raise Exception("expected a provider request")
        return captured[0]

    return get_request


def tool_names(context) -> list[str]:
    return [tool.name for tool in get_current_tools(context.messages)]


async def compact_session(harness) -> None:
    harness.settings_manager.apply_overrides({"compaction": {"keepRecentTokens": 1}})
    harness.set_responses([faux_assistant_message("one"), faux_assistant_message("two")])
    await harness.session.prompt("first")
    await harness.session.prompt("second")
    await harness.session.compact()
    assert [message.role for message in harness.session.messages][:2] == ["system", "compactionSummary"]


def _slice_from_summary(messages: list) -> list:
    summary = next((index for index, message in enumerate(messages) if message.role == "compactionSummary"), -1)
    return messages[summary:]


# -- context handlers and system messages -----------------------------------------------


# Regression #9789, #9822: pruning from the compaction summary dropped the prompt and tool checkpoint.
@pytest.mark.tonio
async def test_keeps_the_prompt_and_tools_when_a_handler_slices_from_the_compaction_summary(harnesses):
    seen: list[list] = []

    def factory(pi) -> None:
        async def on_context(event, _ctx):
            seen.append(event["messages"])
            return {"messages": _slice_from_summary(event["messages"])}

        pi.on("context", on_context)

    harness = await create_harness(extension_factories=[compact_via_hook, factory])
    harnesses.append(harness)
    await compact_session(harness)
    get_request = capture_request(harness, "after compaction")

    await harness.session.prompt("third")

    request = get_request()
    assert not any(message.role == "system" for message in seen[-1])
    assert request.messages[0].role == "system"
    assert tool_names(request) == harness.session.get_active_tool_names()
    assert get_current_system_prompt(request.messages) == harness.session.system_prompt
    assert len([message for message in request.messages if message.role == "system"]) == 1


@pytest.mark.tonio
async def test_keeps_mid_conversation_system_messages_in_place_when_a_handler_leaves_the_conversation_unchanged(
    harnesses,
):
    turn = 0

    def factory(pi) -> None:
        async def on_before_agent_start(event, _ctx):
            nonlocal turn
            turn += 1
            if turn == 2:
                event["systemPromptOptions"].sections["plan_mode"] = "Plan only."

        async def on_context(event, _ctx):
            return {"messages": event["messages"]}

        pi.on("before_agent_start", on_before_agent_start)
        pi.on("context", on_context)

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)
    harness.set_responses([faux_assistant_message("one")])
    await harness.session.prompt("first")
    get_request = capture_request(harness, "two")

    await harness.session.prompt("second")

    system_messages = [message for message in get_request().messages if message.role == "system"]
    assert len(system_messages) == 2
    assert system_messages[1].sections == {"plan_mode": "<plan_mode>\nPlan only.\n</plan_mode>"}


@pytest.mark.tonio
async def test_applies_in_place_edits_to_event_messages_without_a_return_value(harnesses):
    def factory(pi) -> None:
        async def on_context(event, _ctx):
            event["messages"].insert(0, UserMessage(content=[TextContent(text="injected")], timestamp=0))

        pi.on("context", on_context)

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)
    get_request = capture_request(harness, "done")

    await harness.session.prompt("hello")

    request = get_request()
    assert [message.role for message in request.messages] == ["system", "user", "user"]
    assert tool_names(request) == harness.session.get_active_tool_names()


@pytest.mark.tonio
async def test_keeps_system_messages_a_handler_adds_after_the_replayed_head(harnesses):
    def factory(pi) -> None:
        async def on_context(event, _ctx):
            return {"messages": [SystemMessage(content="ephemeral reminder", timestamp=0), *event["messages"]]}

        pi.on("context", on_context)

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)
    get_request = capture_request(harness, "done")

    await harness.session.prompt("hello")

    request = get_request()
    assert [message.role for message in request.messages] == ["system", "system", "user"]
    assert tool_names(request) == harness.session.get_active_tool_names()
    assert harness.session.system_prompt in get_current_system_prompt(request.messages)
    assert "ephemeral reminder" in get_current_system_prompt(request.messages)


# -- context_with_system handlers -------------------------------------------------------


@pytest.mark.tonio
async def test_runs_after_context_handlers_on_the_restored_transcript_and_sends_its_output_verbatim(harnesses):
    seen: list[list] = []

    def factory(pi) -> None:
        async def on_context_with_system(event, _ctx):
            seen.append(event["messages"])
            return {
                "messages": [
                    dataclasses.replace(
                        message, tools_added=[tool for tool in message.tools_added if tool.name != "bash"]
                    )
                    if message.role == "system" and message.tools_added
                    else message
                    for message in event["messages"]
                ]
            }

        async def on_context(event, _ctx):
            return {"messages": _slice_from_summary(event["messages"])}

        pi.on("context_with_system", on_context_with_system)
        # Registered after, but runs first: context handlers precede context_with_system.
        pi.on("context", on_context)

    harness = await create_harness(extension_factories=[compact_via_hook, factory])
    harnesses.append(harness)
    await compact_session(harness)
    get_request = capture_request(harness, "after compaction")

    await harness.session.prompt("third")

    received = seen[-1]
    assert received[0].role == "system"
    assert received[1].role == "compactionSummary"
    assert "bash" in harness.session.get_active_tool_names()
    assert tool_names(get_request()) == [name for name in harness.session.get_active_tool_names() if name != "bash"]


@pytest.mark.tonio
async def test_reports_a_handler_that_drops_the_leading_system_message_but_honors_its_output(harnesses):
    def factory(pi) -> None:
        async def on_context_with_system(event, _ctx):
            return {"messages": [message for message in event["messages"] if message.role != "system"]}

        pi.on("context_with_system", on_context_with_system)

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)
    errors: list[str] = []
    harness.session.extension_runner.on_error(lambda error: errors.append(f"{error.event}: {error.error}"))
    get_request = capture_request(harness, "done")

    await harness.session.prompt("hello")

    assert [message.role for message in get_request().messages] == ["user"]
    assert len(errors) == 1
    assert errors[0].startswith("context_with_system: Handler removed the leading system message")
