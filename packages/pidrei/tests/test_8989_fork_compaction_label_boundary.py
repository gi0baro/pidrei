"""Mirror of pi's suite/regressions/8989-fork-compaction-label-boundary.test.ts."""

import pytest

from pidrei.core.session_manager import SessionManager

from .coding_session_helpers import user_msg


@pytest.mark.tonio
async def test_preserves_compaction_context_when_a_fork_removes_the_boundary_label():
    session = SessionManager.in_memory()
    old_id = await session.append_message(user_msg("old"))
    # find_cut_point() can move a compaction boundary back to this context-invisible label.
    label_id = await session.append_label_change(old_id, "checkpoint")
    kept_id = await session.append_message(user_msg("kept"))
    compaction_id = await session.append_compaction("summary", label_id, 100)
    leaf_id = await session.append_message(user_msg("after"))

    await session.create_branched_session(leaf_id)

    assert session.get_entry(compaction_id)["firstKeptEntryId"] == kept_id
    messages = session.build_session_context().messages
    assert [message.role for message in messages] == ["compactionSummary", "user", "user"]
    assert messages[0].summary == "summary"
    assert [message.content for message in messages[1:]] == [user_msg("kept").content, user_msg("after").content]
