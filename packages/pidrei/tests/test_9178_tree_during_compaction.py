"""Mirror of pi's suite/regressions/9178-tree-during-compaction.test.ts.

JS deferreds become `tonio.Event`s; the pending compaction/navigation promise
becomes a spawned task awaited at the end.
"""

import pytest
import tonio.colored as tonio

from pidrei.core.compaction import CompactionResult

from .coding_session_helpers import assistant_msg, user_msg
from .harness import create_harness, get_message_text


_BUSY = "Wait for the current compaction or tree navigation to finish before navigating the session tree."


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


@pytest.mark.tonio
async def test_rejects_navigation_before_the_active_leaf_can_change(harnesses):
    compaction_started = tonio.Event()
    compaction_released = tonio.Event()

    def factory(pi) -> None:
        async def on_before_compact(event, _ctx):
            compaction_started.set()
            await compaction_released.wait()
            preparation = event["preparation"]
            return {
                "compaction": CompactionResult(
                    summary="summary",
                    first_kept_entry_id=preparation.first_kept_entry_id,
                    tokens_before=preparation.tokens_before,
                    details={},
                )
            }

        pi.on("session_before_compact", on_before_compact)

    harness = await create_harness(settings={"compaction": {"keepRecentTokens": 1}}, extension_factories=[factory])
    harnesses.append(harness)

    await harness.session_manager.append_message(user_msg("first user"))
    navigation_target_id = await harness.session_manager.append_message(assistant_msg("first assistant"))
    await harness.session_manager.append_message(user_msg("second user"))
    original_leaf_id = await harness.session_manager.append_message(assistant_msg("second assistant"))
    harness.session.agent.state.messages = harness.session_manager.build_session_context().messages

    compaction = tonio.spawn(harness.session.compact())
    await compaction_started.wait(5)
    assert compaction_started.is_set()

    assert harness.session.is_compacting is True
    try:
        with pytest.raises(RuntimeError, match=_BUSY):
            await harness.session.navigate_tree(navigation_target_id, {"summarize": False})
        assert harness.session_manager.get_leaf_id() == original_leaf_id
    finally:
        compaction_released.set()
    await compaction

    last_entry = harness.session_manager.get_entries()[-1]
    assert last_entry["type"] == "compaction"
    assert last_entry["parentId"] == original_leaf_id
    assert "second assistant" in [get_message_text(message) for message in harness.session.messages]


@pytest.mark.tonio
async def test_rejects_a_second_navigation_while_the_first_is_waiting(harnesses):
    navigation_started = tonio.Event()
    navigation_released = tonio.Event()

    def factory(pi) -> None:
        async def on_before_tree(_event, _ctx):
            navigation_started.set()
            await navigation_released.wait()

        pi.on("session_before_tree", on_before_tree)

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)

    second_target_id = await harness.session_manager.append_message(user_msg("first user"))
    first_target_id = await harness.session_manager.append_message(assistant_msg("first assistant"))
    await harness.session_manager.append_message(user_msg("second user"))
    original_leaf_id = await harness.session_manager.append_message(assistant_msg("second assistant"))
    harness.session.agent.state.messages = harness.session_manager.build_session_context().messages

    first_navigation = tonio.spawn(harness.session.navigate_tree(first_target_id, {"summarize": False}))
    await navigation_started.wait(5)
    assert navigation_started.is_set()

    try:
        with pytest.raises(RuntimeError, match=_BUSY):
            await harness.session.navigate_tree(second_target_id, {"summarize": False})
        assert harness.session_manager.get_leaf_id() == original_leaf_id
    finally:
        navigation_released.set()

    await first_navigation
    assert harness.session_manager.get_leaf_id() == first_target_id
