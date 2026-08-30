"""Mirror of the queued-prompt case pi added to suite/agent-session-compaction.test.ts.

`test_agent_session_compaction.py` mirrors the rest of that file, but it is
built on `create_agent_session` (it predates the suite harness upstream moved
the file onto). This case needs the suite harness — an extension supplying the
compaction result, and event subscribers that prompt — so it lives here rather
than mixing two harness styles into one file.
"""

import pytest
import tonio.colored as tonio

from pidrei.core.compaction import CompactionResult
from pidrei.core.extensions import ToolDefinition
from pidrei_agent.types import AgentToolResult
from pidrei_ai.providers.faux import faux_assistant_message, faux_tool_call
from pidrei_ai.types import TextContent, UserMessage

from .harness import create_harness, get_user_texts


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


def _compaction_factory(pi) -> None:
    async def on_before_compact(event, _ctx):
        preparation = event["preparation"]
        return {
            "compaction": CompactionResult(
                summary="manual compacted",
                first_kept_entry_id=preparation.first_kept_entry_id,
                tokens_before=preparation.tokens_before,
                details={},
            )
        }

    pi.on("session_before_compact", on_before_compact)


@pytest.mark.tonio
async def test_allows_a_queued_prompt_to_start_when_manual_compaction_ends(harnesses):
    harness = await create_harness(
        settings={"compaction": {"keepRecentTokens": 1}}, extension_factories=[_compaction_factory]
    )
    harnesses.append(harness)

    await harness.session_manager.append_message(
        UserMessage(content=[TextContent(text="message to compact")], timestamp=1_000)
    )
    await harness.session_manager.append_message(
        faux_assistant_message("assistant response to compact", timestamp=1_500)
    )
    harness.session.agent.state.messages = harness.session_manager.build_session_context().messages
    harness.set_responses([faux_assistant_message("queued response")])

    queued: list = []

    def on_event(event) -> None:
        if getattr(event, "type", None) == "compaction_end" and event.reason == "manual" and event.result is not None:
            assert harness.session.is_compacting is False
            queued.append(harness.session.prompt("queued after compaction"))

    harness.session.subscribe(on_event)

    await harness.session.compact()
    assert queued, "compaction_end did not start the queued prompt"
    await queued[0]

    assert "queued after compaction" in get_user_texts(harness)
    assert harness.session.get_last_assistant_text() == "queued response"


# -- 0.84.4 compact-before-post-tool-requests cases (same upstream suite) -----


def _large_result_tool(name: str = "large_result", *, terminate: bool | None = None) -> ToolDefinition:
    async def execute(*_args):
        return AgentToolResult(
            content=[TextContent(text=f"large-tool-result:{'x' * 6800}")],
            details={},
            terminate=terminate,
        )

    return ToolDefinition(
        name=name,
        label="Large result",
        description="Returns enough content to cross the compaction threshold",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )


_THRESHOLD_SETTINGS = {"compaction": {"enabled": True, "reserveTokens": 400, "keepRecentTokens": 1750}}
_THRESHOLD_MODELS = [{"id": "faux-1", "context_window": 2600, "max_tokens": 100}]


@pytest.mark.tonio
async def test_compacts_after_a_tool_result_before_the_next_assistant_request_in_the_same_run(harnesses):
    order: list[str] = []

    def factory(pi) -> None:
        async def on_before_compact(event, _ctx):
            order.append("compaction")
            preparation = event["preparation"]
            return {
                "compaction": CompactionResult(
                    summary="compacted history",
                    first_kept_entry_id=preparation.first_kept_entry_id,
                    tokens_before=preparation.tokens_before,
                    details={},
                )
            }

        pi.on("session_before_compact", on_before_compact)

    harness = await create_harness(
        models=_THRESHOLD_MODELS,
        settings=_THRESHOLD_SETTINGS,
        tools=[_large_result_tool()],
        extension_factories=[factory],
    )
    harnesses.append(harness)
    resumed_request: list[str] = []

    async def resumed(context, *_rest):
        order.append("provider")
        resumed_request.append(repr(context.messages))
        return faux_assistant_message("finished after compaction")

    harness.set_responses(
        [
            faux_assistant_message(f"old-history:{'a' * 800}"),
            faux_assistant_message(f"recent-history:{'b' * 800}"),
            faux_assistant_message(faux_tool_call("large_result", {}), stop_reason="toolUse"),
            resumed,
        ]
    )

    await harness.session.prompt("seed old history")
    await harness.session.prompt("seed recent history")
    agent_starts_before = len(harness.events_of_type("agent_start"))
    await harness.session.prompt("run the large tool")

    assert order == ["compaction", "provider"]
    assert len(harness.events_of_type("agent_start")) == agent_starts_before + 1
    last_compaction_start = harness.events_of_type("compaction_start")[-1]
    assert last_compaction_start.reason == "threshold"
    assert "compacted history" in resumed_request[0]
    assert "large-tool-result" in resumed_request[0]
    assert harness.session.get_last_assistant_text() == "finished after compaction"


@pytest.mark.tonio
async def test_includes_steering_queued_during_compaction_in_the_resumed_assistant_request(harnesses):
    compaction_started = tonio.Event()
    compaction_released = tonio.Event()

    def factory(pi) -> None:
        async def on_before_compact(event, _ctx):
            compaction_started.set()
            await compaction_released.wait(None)
            preparation = event["preparation"]
            return {
                "compaction": CompactionResult(
                    summary="compacted history",
                    first_kept_entry_id=preparation.first_kept_entry_id,
                    tokens_before=preparation.tokens_before,
                    details={},
                )
            }

        pi.on("session_before_compact", on_before_compact)

    harness = await create_harness(
        models=_THRESHOLD_MODELS,
        settings=_THRESHOLD_SETTINGS,
        tools=[_large_result_tool()],
        extension_factories=[factory],
    )
    harnesses.append(harness)
    resumed_request: list[str] = []

    async def resumed(context, *_rest):
        resumed_request.append(repr(context.messages))
        return faux_assistant_message("finished after compaction")

    harness.set_responses(
        [
            faux_assistant_message(f"old-history:{'a' * 800}"),
            faux_assistant_message(f"recent-history:{'b' * 800}"),
            faux_assistant_message(faux_tool_call("large_result", {}), stop_reason="toolUse"),
            resumed,
            faux_assistant_message("finished after delayed steering"),
        ]
    )

    await harness.session.prompt("seed old history")
    await harness.session.prompt("seed recent history")
    prompt_handle = tonio.spawn(harness.session.prompt("run the large tool"))
    await compaction_started.wait(None)
    await harness.session.steer("change direction")
    compaction_released.set()
    await prompt_handle

    assert "change direction" in resumed_request[0]
    assert harness.faux.state.call_count == 4


@pytest.mark.tonio
async def test_does_not_compact_after_a_terminating_tool_result(harnesses):
    def factory(pi) -> None:
        async def on_before_compact(event, _ctx):
            preparation = event["preparation"]
            return {
                "compaction": CompactionResult(
                    summary="unexpected compaction",
                    first_kept_entry_id=preparation.first_kept_entry_id,
                    tokens_before=preparation.tokens_before,
                    details={},
                )
            }

        pi.on("session_before_compact", on_before_compact)

    harness = await create_harness(
        models=_THRESHOLD_MODELS,
        settings=_THRESHOLD_SETTINGS,
        tools=[_large_result_tool("terminate_with_large_result", terminate=True)],
        extension_factories=[factory],
    )
    harnesses.append(harness)
    harness.set_responses(
        [
            faux_assistant_message(f"old-history:{'a' * 800}"),
            faux_assistant_message(f"recent-history:{'b' * 800}"),
            faux_assistant_message(faux_tool_call("terminate_with_large_result", {}), stop_reason="toolUse"),
        ]
    )

    await harness.session.prompt("seed old history")
    await harness.session.prompt("seed recent history")
    await harness.session.prompt("run the terminating tool")

    assert harness.events_of_type("compaction_start") == []
    assert [e for e in harness.session_manager.get_entries() if e.get("type") == "compaction"] == []
    assert harness.get_pending_response_count() == 0
