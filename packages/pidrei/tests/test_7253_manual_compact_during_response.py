"""Mirror of pi's suite/regressions/7253-manual-compact-during-response.test.ts.

JS deferreds become `tonio.Event`s, and pi's `Promise.all([prompt, compact])`
becomes a `tonio.spawn` of the prompt and a driver that requests the manual
compaction once the second response is in flight.
"""

import pytest
import tonio.colored as tonio

from pidrei.core.compaction import CompactionResult
from pidrei.core.extensions import ToolDefinition
from pidrei_agent.types import AgentToolResult
from pidrei_ai.providers.faux import faux_assistant_message, faux_tool_call
from pidrei_ai.types import TextContent

from .harness import create_harness


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


def _create_noop_tool() -> ToolDefinition:
    async def execute(*_args):
        return AgentToolResult(content=[TextContent(text="done")], details={})

    return ToolDefinition(
        name="noop",
        label="No-op",
        description="Return immediately",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )


def _compaction_factory(pi) -> None:
    async def on_before_compact(event, _ctx):
        preparation = event["preparation"]
        return {
            "compaction": CompactionResult(
                summary=f"{event['reason']} summary",
                first_kept_entry_id=preparation.first_kept_entry_id,
                tokens_before=preparation.tokens_before,
                details={},
            )
        }

    pi.on("session_before_compact", on_before_compact)


@pytest.mark.tonio
async def test_runs_only_the_requested_manual_compaction_when_the_previous_turn_crossed_the_threshold(harnesses):
    second_response_started = tonio.Event()
    second_response_released = tonio.Event()

    harness = await create_harness(
        models=[{"id": "faux-1", "context_window": 1000, "max_tokens": 100}],
        settings={"compaction": {"enabled": True, "reserveTokens": 999, "keepRecentTokens": 2}},
        tools=[_create_noop_tool()],
        extension_factories=[_compaction_factory],
    )
    harnesses.append(harness)

    async def second(*_args):
        second_response_started.set()
        await second_response_released.wait(None)
        return faux_assistant_message("second response")

    harness.set_responses(
        [
            faux_assistant_message(faux_tool_call("noop", {}), stop_reason="toolUse"),
            second,
        ]
    )

    compaction_results: list = []

    async def drive() -> None:
        await second_response_started.wait(None)
        compact = tonio.spawn(harness.session.compact())
        # pi's `compact()` runs synchronously into `abort()`; a spawned
        # coroutine here only starts on a real suspension, so give it one
        # before releasing the response the abort is waiting for.
        await tonio.time.sleep(0)
        second_response_released.set()
        compaction_results.append(await compact)

    await tonio.spawn(harness.session.prompt("Run the tool, then continue responding."), drive())

    assert compaction_results[0].summary == "manual summary"
    assert [event.reason for event in harness.events_of_type("compaction_start")] == ["manual"]
    assert [event.reason for event in harness.events_of_type("compaction_end")] == ["manual"]
    assert len([e for e in harness.session_manager.get_entries() if e.get("type") == "compaction"]) == 1
