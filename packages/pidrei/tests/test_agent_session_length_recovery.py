"""Mirror of the length-stop recovery cases pi added to
suite/agent-session-compaction.test.ts (#7540).

Like `test_agent_session_compaction_queue.py`, these need the suite harness,
which `test_agent_session_compaction.py` (built on `create_agent_session`)
does not use.
"""

import pytest

from pidrei.core.compaction import CompactionResult
from pidrei_ai.providers.faux import faux_assistant_message

from .harness import create_harness


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
                summary="overflow compacted",
                first_kept_entry_id=preparation.first_kept_entry_id,
                tokens_before=preparation.tokens_before,
                details={},
            )
        }

    pi.on("session_before_compact", on_before_compact)


@pytest.mark.tonio
async def test_compacts_and_resumes_after_a_length_stop_below_the_desired_output_limit(harnesses):
    harness = await create_harness(
        models=[{"id": "faux-1", "context_window": 1000, "max_tokens": 100}],
        settings={"compaction": {"keepRecentTokens": 1, "reserveTokens": 0}},
        extension_factories=[_compaction_factory],
    )
    harnesses.append(harness)
    harness.set_responses(
        [
            faux_assistant_message("partial response", stop_reason="length"),
            faux_assistant_message("completed response"),
        ]
    )

    await harness.session.prompt("x" * 5000)

    assert harness.faux.state.call_count == 2
    last_end = harness.events_of_type("compaction_end")[-1]
    assert last_end.reason == "overflow"
    assert last_end.aborted is False
    assert last_end.will_retry is True
    assert harness.session.get_last_assistant_text() == "completed response"


@pytest.mark.tonio
async def test_does_not_compact_when_a_length_stop_reaches_the_desired_output_limit(harnesses):
    harness = await create_harness(models=[{"id": "faux-1", "context_window": 1_000_000, "max_tokens": 100}])
    harnesses.append(harness)
    harness.set_responses([faux_assistant_message("x" * 400, stop_reason="length")])

    await harness.session.prompt("hello")

    assert harness.faux.state.call_count == 1
    assert harness.events_of_type("compaction_start") == []


@pytest.mark.tonio
async def test_stops_after_one_compact_and_retry_when_a_second_response_is_also_truncated(harnesses):
    harness = await create_harness(
        models=[{"id": "faux-1", "context_window": 1_000_000, "max_tokens": 100}],
        settings={"compaction": {"keepRecentTokens": 1, "reserveTokens": 0}},
        extension_factories=[_compaction_factory],
    )
    harnesses.append(harness)

    # Timestamps ahead of the compaction entry: `_check_compaction` ignores
    # assistant messages that predate the latest compaction.
    async def truncated(text: str, *_args):
        return faux_assistant_message(text, stop_reason="length", timestamp=2**41)

    harness.set_responses(
        [
            lambda *args: truncated("x" * 64, *args),
            lambda *args: truncated("y" * 64, *args),
        ]
    )

    await harness.session.prompt("x" * 5000)

    assert harness.faux.state.call_count == 2
    overflow_starts = [e for e in harness.events_of_type("compaction_start") if e.reason == "overflow"]
    assert len(overflow_starts) == 1
    assert harness.events_of_type("compaction_end")[-1].error_message == (
        "Context overflow recovery failed after one compact-and-retry attempt. "
        "Try reducing context or switching to a larger-context model."
    )
