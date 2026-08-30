"""Mirror of pi's suite/regressions/7048-compaction-truncated-summary.test.ts."""

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
async def test_does_not_persist_a_length_limited_summary(harnesses):
    harness = await create_harness()
    harnesses.append(harness)
    await _seed_compactable_session(harness)
    harness.set_responses([faux_assistant_message("partial summar", stop_reason="length")])

    with pytest.raises(Exception, match="generation hit the token cap"):
        await harness.session.compact()
    entries = [e for e in harness.session_manager.get_entries() if e.get("type") == "compaction"]
    assert entries == []
