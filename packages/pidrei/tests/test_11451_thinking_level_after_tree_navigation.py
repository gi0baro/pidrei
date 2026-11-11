"""Mirror of pi's suite/regressions/11451-thinking-level-after-tree-navigation.test.ts.

pi's `setThinkingLevel` and `appendThinkingLevelChange` are sync calls; both are
awaited here. The last case is pidrei-only.
"""

import pytest

from pidrei_ai.providers.faux import faux_assistant_message

from .harness import create_harness


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


# Regression for #11451: tree navigation can leave the latest thinking_level_change on another branch.
@pytest.mark.tonio
async def test_records_the_live_thinking_level_on_the_new_branch_before_the_next_prompt(harnesses):
    harness = await create_harness(models=[{"id": "faux-reasoning", "reasoning": True}])
    harnesses.append(harness)
    harness.set_responses(
        [faux_assistant_message("one"), faux_assistant_message("two"), faux_assistant_message("three")]
    )
    session, session_manager = harness.session, harness.session_manager

    await session.set_thinking_level("high")
    await session.prompt("one")
    first_answer = session_manager.get_leaf_id()
    await session.set_thinking_level("low")
    await session.prompt("two")
    await session.navigate_tree(first_answer)
    assert session_manager.build_session_context().thinking_level == "high"
    await session.prompt("three")

    assert session.thinking_level == "low"
    # Resume restores the thinking level from this context.
    assert session_manager.build_session_context().thinking_level == "low"


@pytest.mark.tonio
async def test_does_not_record_a_level_that_only_differs_from_the_branch_by_clamping(harnesses):
    harness = await create_harness(models=[{"id": "faux-reasoning", "reasoning": True}])
    harnesses.append(harness)
    harness.set_responses([faux_assistant_message("one")])
    session, session_manager = harness.session, harness.session_manager

    await session.set_thinking_level("high")
    # As after resuming a session that requested xhigh on a model without it: the session runs at high.
    await session_manager.append_thinking_level_change("xhigh")
    await session.prompt("one")

    levels = [
        entry["thinkingLevel"] for entry in session_manager.get_branch() if entry["type"] == "thinking_level_change"
    ]
    assert levels[-1] == "xhigh"


# pidrei-only: pi compares and appends in one synchronous step. Here the recorder's
# append suspends, so a level change landing in it must not leave the older level
# last on the branch. The change is forced into the append by wrapping the session
# manager's method.
@pytest.mark.tonio
async def test_records_a_level_changed_while_the_recorded_level_is_appended(harnesses, monkeypatch):
    harness = await create_harness(models=[{"id": "faux-reasoning", "reasoning": True}])
    harnesses.append(harness)
    harness.set_responses([faux_assistant_message("one"), faux_assistant_message("two")])
    session, session_manager = harness.session, harness.session_manager

    await session.set_thinking_level("high")
    await session.prompt("one")
    first_answer = session_manager.get_leaf_id()
    await session.set_thinking_level("low")
    await session.navigate_tree(first_answer)

    append = session_manager.append_thinking_level_change
    changed: list[str] = []

    async def append_after_a_level_change(level: str) -> str:
        if not changed and level == "low":
            changed.append(level)
            await session.set_thinking_level("medium")
        return await append(level)

    monkeypatch.setattr(session_manager, "append_thinking_level_change", append_after_a_level_change)
    await session.prompt("two")

    assert changed == ["low"]
    assert session.thinking_level == "medium"
    assert session_manager.build_session_context().thinking_level == "medium"
