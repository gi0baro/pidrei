"""Mirror of pi's suite/regressions/7150-rpc-prompt-during-compaction.test.ts.

JS deferreds become `tonio.Event`s: one the compaction hook sets when it starts,
one the test sets to release it.
"""

import pytest
import tonio.colored as tonio

from pidrei.core.agent_session import PromptOptions
from pidrei.core.compaction import CompactionResult
from pidrei_ai.providers.faux import faux_assistant_message
from pidrei_ai.types import TextContent, UserMessage

from .harness import create_harness, get_message_text, get_user_texts


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


@pytest.mark.tonio
async def test_rejects_an_rpc_prompt_while_manual_compaction_is_in_progress(harnesses):
    compaction_started = tonio.Event()
    compaction_released = tonio.Event()

    def factory(pi) -> None:
        async def on_before_compact(event, _ctx):
            compaction_started.set()
            await compaction_released.wait(None)
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

    harness = await create_harness(settings={"compaction": {"keepRecentTokens": 1}}, extension_factories=[factory])
    harnesses.append(harness)

    await harness.session_manager.append_message(
        UserMessage(content=[TextContent(text="old user message")], timestamp=1_000)
    )
    await harness.session_manager.append_message(faux_assistant_message("old assistant response", timestamp=1_500))
    harness.session.agent.state.messages = harness.session_manager.build_session_context().messages
    harness.set_responses([faux_assistant_message("probe response")])

    preflight_results: list[bool] = []
    prompt_errors: list[Exception] = []

    async def probe() -> None:
        await compaction_started.wait(None)
        try:
            await harness.session.prompt(
                "PROBE-7150",
                PromptOptions(source="rpc", preflight_result=preflight_results.append),
            )
        except Exception as error:
            prompt_errors.append(error)
        finally:
            compaction_released.set()

    await tonio.spawn(harness.session.compact(), probe())

    persisted_user_texts = [
        get_message_text(entry["message"])
        for entry in harness.session_manager.get_entries()
        if entry.get("type") == "message" and entry["message"].role == "user"
    ]

    assert preflight_results == [False]
    assert len(prompt_errors) == 1
    assert "compaction is in progress" in str(prompt_errors[0])
    assert "PROBE-7150" not in get_user_texts(harness)
    assert "PROBE-7150" not in persisted_user_texts
    assert harness.events_of_type("agent_start") == []
    assert harness.events_of_type("agent_settled") == []
