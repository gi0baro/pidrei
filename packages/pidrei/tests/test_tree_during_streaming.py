"""Mirror of pi's suite/regressions/tree-during-streaming.test.ts."""

import pytest

from pidrei_ai.providers.faux import faux_assistant_message

from .coding_session_helpers import user_msg
from .harness import create_harness


@pytest.mark.tonio
async def test_rejects_navigation_without_changing_the_active_leaf():
    harness = await create_harness()
    try:
        target_id = await harness.session_manager.append_message(user_msg("first"))
        navigation: dict = {}

        # Navigate from inside the response factory, while the run is active.
        async def factory(_context, _options, _state, _model):
            active_leaf_id = harness.session_manager.get_leaf_id()
            try:
                navigation["result"] = await harness.session.navigate_tree(target_id, {"summarize": False})
            except Exception as error:
                navigation["result"] = error
            navigation["leaf_unchanged"] = (
                active_leaf_id != target_id and harness.session_manager.get_leaf_id() == active_leaf_id
            )
            return faux_assistant_message("response")

        harness.set_responses([factory])
        await harness.session.prompt("second")

        assert isinstance(navigation["result"], RuntimeError)
        assert (
            str(navigation["result"]) == "Wait for the current response to finish before navigating the session tree."
        )
        assert navigation["leaf_unchanged"] is True
    finally:
        harness.cleanup()
