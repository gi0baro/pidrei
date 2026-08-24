"""Mirror of pi coding-agent test/suite/regressions/7925-toolcall-start-metadata.test.ts."""

import pytest

from pidrei.modes.json_event import to_json_event
from pidrei_ai.providers.faux import faux_assistant_message, faux_tool_call

from .harness import create_harness


@pytest.mark.tonio
async def test_includes_the_tool_call_id_and_name_without_cumulative_snapshots():
    harness = await create_harness()
    try:
        harness.set_responses(
            [
                faux_assistant_message(
                    faux_tool_call("write", {"path": "output.txt", "content": "x" * 100}, id="call_7925"),
                    stop_reason="toolUse",
                ),
                faux_assistant_message("done"),
            ]
        )

        await harness.session.prompt("write a file")

        update = next(
            (
                event
                for event in harness.events_of_type("message_update")
                if event.assistant_message_event.type == "toolcall_start"
            ),
            None,
        )
        assert update is not None and getattr(update.message, "role", None) == "assistant", (
            "Expected toolcall_start assistant update"
        )

        assert to_json_event(update) == {
            "type": "message_update",
            "usage": update.message.usage,
            "assistantMessageEvent": {
                "type": "toolcall_start",
                "contentIndex": 0,
                "id": "call_7925",
                "toolName": "write",
            },
        }
    finally:
        harness.cleanup()
