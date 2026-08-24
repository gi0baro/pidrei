"""Mirror of pi coding-agent test/suite/regressions/8328-zero-usage-auto-compaction.test.ts.

pi spies on `_runAutoCompaction` with `vi.spyOn`; here the method is swapped on
the instance, which is the same interception point.
"""

import time

import pytest

from pidrei_ai.types import AssistantMessage, TextContent, Usage, UsageCost, UserMessage

from .harness import Harness, create_harness


async def create_compaction_harness() -> Harness:
    return await create_harness(
        models=[{"id": "faux-1", "context_window": 100, "max_tokens": 20}],
        settings={"compaction": {"enabled": True, "reserveTokens": 10}},
    )


def zero_usage_assistant(harness: Harness) -> AssistantMessage:
    model = harness.get_model()
    return AssistantMessage(
        content=[TextContent(text="response")],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(input=0, output=0, cache_read=0, cache_write=0, total_tokens=0, cost=UsageCost()),
        stop_reason="stop",
        timestamp=int(time.time() * 1000),
    )


def record_auto_compaction(session) -> list[tuple]:
    calls: list[tuple] = []

    async def fake(reason, will_retry):
        calls.append((reason, will_retry))
        return False

    session._run_auto_compaction = fake
    return calls


@pytest.mark.tonio
async def test_uses_the_message_estimate_when_no_assistant_has_reported_usage():
    harness = await create_compaction_harness()
    try:
        assistant = zero_usage_assistant(harness)
        harness.session.agent.state.messages = [
            UserMessage(content=[TextContent(text="x" * 400)], timestamp=int(time.time() * 1000) - 1),
            assistant,
        ]
        calls = record_auto_compaction(harness.session)

        await harness.session._check_compaction(assistant)

        assert calls == [("threshold", False)]
    finally:
        harness.cleanup()


@pytest.mark.tonio
async def test_does_not_compact_when_the_zero_usage_message_estimate_is_below_the_threshold():
    harness = await create_compaction_harness()
    try:
        assistant = zero_usage_assistant(harness)
        harness.session.agent.state.messages = [
            UserMessage(content=[TextContent(text="short")], timestamp=int(time.time() * 1000) - 1),
            assistant,
        ]
        calls = record_auto_compaction(harness.session)

        await harness.session._check_compaction(assistant)

        assert calls == []
    finally:
        harness.cleanup()
