"""Mirror of the `session_compact_failed` case pi added to suite/agent-session-compaction.test.ts.

`test_agent_session_compaction.py` mirrors the older, non-suite compaction file
and is built on `create_agent_session`; this case needs the suite harness (an
extension subscribing to the failure event), so it lives here — the same split
`test_agent_session_compaction_queue.py` makes.
"""

import dataclasses

import pytest

from pidrei_ai.providers.faux import faux_assistant_message
from pidrei_ai.types import TextContent, Usage, UsageCost, UserMessage
from pidrei_ai.utils.clock import now_ms

from .harness import create_harness


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


async def _seed_compactable_session(harness) -> None:
    harness.settings_manager.apply_overrides({"compaction": {"keepRecentTokens": 1}})
    now = now_ms()
    await harness.session_manager.append_message(
        UserMessage(content=[TextContent(text="message to compact")], timestamp=now - 1000)
    )
    model = harness.get_model()
    assistant = dataclasses.replace(
        faux_assistant_message("assistant response to compact", timestamp=now - 500),
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(input=100, total_tokens=100, cost=UsageCost()),
    )
    await harness.session_manager.append_message(assistant)
    harness.session.agent.state.messages = harness.session_manager.build_session_context().messages


@pytest.mark.tonio
async def test_notifies_extensions_when_auto_compaction_fails(harnesses):
    failed_events: list = []

    def factory(pi) -> None:
        async def on_compact_failed(event, _ctx):
            failed_events.append(event)

        pi.on("session_compact_failed", on_compact_failed)

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)
    await _seed_compactable_session(harness)

    async def exploding_stream_fn(*_args, **_kwargs):
        raise Exception("summary generator blew up")

    harness.session.agent.stream_function = exploding_stream_fn

    assert await harness.session._run_auto_compaction("threshold", False) is False

    end_event = harness.events_of_type("compaction_end")[-1]
    assert end_event.reason == "threshold"
    assert end_event.aborted is False
    assert end_event.will_retry is False
    assert end_event.error_message == "Auto-compaction failed: summary generator blew up"
    assert failed_events == [
        {
            "type": "session_compact_failed",
            "reason": "threshold",
            "errorMessage": "Auto-compaction failed: summary generator blew up",
            "aborted": False,
            "willRetry": False,
            "fromExtension": False,
        }
    ]
