"""Mirror of the queued-prompt case pi added to suite/agent-session-compaction.test.ts.

`test_agent_session_compaction.py` mirrors the rest of that file, but it is
built on `create_agent_session` (it predates the suite harness upstream moved
the file onto). This case needs the suite harness — an extension supplying the
compaction result, and event subscribers that prompt — so it lives here rather
than mixing two harness styles into one file.
"""

import pytest

from pidrei.core.compaction import CompactionResult
from pidrei_ai.providers.faux import faux_assistant_message
from pidrei_ai.types import TextContent, UserMessage

from .harness import create_harness, get_user_texts


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


def _compaction_factory(pi) -> None:
    async def on_before_compact(event, _ctx):
        preparation = event["preparation"]
        return {
            "compaction": CompactionResult(
                summary="manual compacted",
                first_kept_entry_id=preparation.first_kept_entry_id,
                tokens_before=preparation.tokens_before,
                details={},
            )
        }

    pi.on("session_before_compact", on_before_compact)


@pytest.mark.tonio
async def test_allows_a_queued_prompt_to_start_when_manual_compaction_ends(harnesses):
    harness = await create_harness(
        settings={"compaction": {"keepRecentTokens": 1}}, extension_factories=[_compaction_factory]
    )
    harnesses.append(harness)

    await harness.session_manager.append_message(
        UserMessage(content=[TextContent(text="message to compact")], timestamp=1_000)
    )
    await harness.session_manager.append_message(
        faux_assistant_message("assistant response to compact", timestamp=1_500)
    )
    harness.session.agent.state.messages = harness.session_manager.build_session_context().messages
    harness.set_responses([faux_assistant_message("queued response")])

    queued: list = []

    def on_event(event) -> None:
        if getattr(event, "type", None) == "compaction_end" and event.reason == "manual" and event.result is not None:
            assert harness.session.is_compacting is False
            queued.append(harness.session.prompt("queued after compaction"))

    harness.session.subscribe(on_event)

    await harness.session.compact()
    assert queued, "compaction_end did not start the queued prompt"
    await queued[0]

    assert "queued after compaction" in get_user_texts(harness)
    assert harness.session.get_last_assistant_text() == "queued response"
