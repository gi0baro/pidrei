"""Mirror of pi coding-agent test/suite/regressions/7911-json-stream-usage.test.ts."""

import pytest

from pidrei.modes.json_event import to_json_event
from pidrei_ai.providers.faux import faux_assistant_message

from .harness import create_harness


@pytest.mark.tonio
async def test_includes_cumulative_usage_without_cumulative_message_snapshots():
    harness = await create_harness()
    try:
        harness.set_responses([faux_assistant_message("hello")])

        await harness.session.prompt("respond")

        # #7290's delta-only wire projection dropped this fixed-size metadata
        # with the snapshots.
        update = next(
            (
                event
                for event in harness.events_of_type("message_update")
                if getattr(event.message, "role", None) == "assistant" and event.message.usage.total_tokens > 0
            ),
            None,
        )
        assert update is not None, "Expected an assistant update with populated usage"

        wire_update = to_json_event(update)
        assert wire_update["usage"] == update.message.usage
        assert "message" not in wire_update
        assert "partial" not in wire_update["assistantMessageEvent"]
    finally:
        harness.cleanup()
