"""v4 branch summarization (mirror of pi agent/test/harness/branch-summarization.test.ts)."""

import pytest

from pidrei_agent.harness.compaction.branch_summarization import collect_entries_for_branch_summary
from pidrei_agent.harness.session.memory import InMemorySessionStorage
from pidrei_agent.harness.session.session import Session
from pidrei_agent.harness.session.types import SessionMetadata
from pidrei_ai.types import TextContent, UserMessage


def message(text: str) -> UserMessage:
    return UserMessage(content=[TextContent(text=text)], timestamp=1)


class _CountingIdGenerator:
    def __init__(self) -> None:
        self._next_id = 0

    def next(self) -> str:
        self._next_id += 1
        return f"entry-{self._next_id}"


@pytest.mark.tonio
async def test_collects_the_abandoned_side_of_a_branch_in_chronological_order():
    session = Session(
        InMemorySessionStorage(SessionMetadata(id="session", created_at=1)), id_generator=_CountingIdGenerator()
    )
    root_id = await session.append_message(message("root"))
    common_id = await session.append_message(message("common"))
    abandoned_ids = [
        await session.append_message(message("abandoned 1")),
        await session.append_message(message("abandoned 2")),
    ]
    await session.create_lane("target", common_id)
    target_id = await session.view("target").append_message(message("target"))

    result = await collect_entries_for_branch_summary(session, abandoned_ids[1], target_id)
    assert result.common_ancestor_id == common_id
    assert [entry.id for entry in result.entries] == abandoned_ids
    assert all(entry.id != root_id for entry in result.entries)


@pytest.mark.tonio
async def test_returns_no_entries_when_there_was_no_previous_leaf():
    session = Session(InMemorySessionStorage(SessionMetadata(id="session", created_at=1)))
    target_id = await session.append_message(message("target"))
    result = await collect_entries_for_branch_summary(session, None, target_id)
    assert result.entries == []
    assert result.common_ancestor_id is None
