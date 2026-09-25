"""Mirror of pi's suite/agent-session-compaction-model-overrides.test.ts.

Regression coverage for #8133. pi's `vi.spyOn(modelRuntime, "getAuth")` is an
instance-attribute swap on the harness's model runtime.
"""

from dataclasses import replace

import pytest

from pidrei.core.compaction import CompactionResult, CompactionSettings
from pidrei_ai.providers.faux import faux_assistant_message
from pidrei_ai.types import TextContent, UserMessage

from .harness import create_harness


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


async def _seed_history(harness, total_tokens: int = 650) -> str:
    model = harness.session.model
    recent_user_id = ""
    now = 1_700_000_000_000
    for label in ("old", "recent"):
        recent_user_id = await harness.session_manager.append_message(
            UserMessage(content=[TextContent(text=label.ljust(400, "x"))], timestamp=now - 2000)
        )
        assistant = faux_assistant_message(label.ljust(400, "y"), timestamp=now - 1000)
        await harness.session_manager.append_message(
            replace(
                assistant,
                api=model.api,
                provider=model.provider,
                model=model.id,
                usage=replace(assistant.usage, input=total_tokens, total_tokens=total_tokens),
            )
        )
    harness.session.agent.state.messages = harness.session_manager.build_session_context().messages
    return recent_user_id


def _compaction_factory(preparations: list, summary: str = "compacted history"):
    async def factory(pi) -> None:
        async def on_before_compact(event, _ctx):
            preparations.append(event)
            preparation = event["preparation"]
            return {
                "compaction": CompactionResult(
                    summary=summary,
                    first_kept_entry_id=preparation.first_kept_entry_id,
                    tokens_before=preparation.tokens_before,
                )
            }

        pi.on("session_before_compact", on_before_compact)

    return factory


@pytest.mark.tonio
@pytest.mark.parametrize("path", ["manual", "pre-prompt", "post-run", "overflow"])
async def test_uses_model_token_settings_for_compaction_and_extension_preparation(harnesses, path):
    preparations: list = []
    harness = await create_harness(
        models=[{"id": "faux-1", "context_window": 4000}],
        tools=[],
        settings={
            "compaction": {
                "enabled": path != "manual",
                "reserveTokens": 10,
                "keepRecentTokens": 20000,
                "modelOverrides": {"faux/faux-1": {"reserveTokens": 2000, "keepRecentTokens": 150}},
            }
        },
        extension_factories=[_compaction_factory(preparations)],
    )
    harnesses.append(harness)
    recent_user_id = await _seed_history(harness, 2500 if path == "pre-prompt" else 650)

    if path == "manual":
        await harness.session.compact()
    else:
        harness.set_responses(
            [
                faux_assistant_message("", stop_reason="error", error_message="prompt is too long"),
                faux_assistant_message("recovered"),
            ]
            if path == "overflow"
            else [faux_assistant_message("z" * 8000 if path == "post-run" else "done")]
        )
        await harness.session.prompt("continue")

    assert len(preparations) == 1
    assert preparations[0]["preparation"].settings == CompactionSettings(
        enabled=path != "manual", reserve_tokens=2000, keep_recent_tokens=150
    )
    assert preparations[0]["reason"] == (
        "manual" if path == "manual" else "overflow" if path == "overflow" else "threshold"
    )
    if path in ("manual", "pre-prompt"):
        assert preparations[0]["preparation"].first_kept_entry_id == recent_user_id
    compaction_ends = harness.events_of_type("compaction_end")
    assert len(compaction_ends) == 1
    assert compaction_ends[0].aborted is False
    assert compaction_ends[0].will_retry is (path == "overflow")
    assert compaction_ends[0].result.summary == "compacted history"
    assert harness.get_pending_response_count() == 0


@pytest.mark.tonio
@pytest.mark.parametrize("path", ["manual", "automatic"])
async def test_passes_resolved_budgets_to_built_in_summarization(harnesses, path):
    harness = await create_harness(
        models=[{"id": "faux-1", "context_window": 4000, "max_tokens": 3000}],
        tools=[],
        settings={
            "compaction": {
                "reserveTokens": 10,
                "modelOverrides": {"faux/faux-1": {"reserveTokens": 2000, "keepRecentTokens": 150}},
            }
        },
    )
    harnesses.append(harness)
    recent_user_id = await _seed_history(harness, 2500)
    budgets: list = []

    async def summarize(_context, options, *_rest):
        budgets.append(options.max_tokens if options is not None else None)
        return faux_assistant_message("built-in summary")

    harness.set_responses([summarize, *([faux_assistant_message("done")] if path == "automatic" else [])])
    if path == "manual":
        await harness.session.compact()
    else:
        await harness.session.prompt("continue")

    assert budgets == [1600]
    compaction = next(entry for entry in harness.session_manager.get_entries() if entry["type"] == "compaction")
    assert compaction["firstKeptEntryId"] == recent_user_id
    assert compaction["summary"] == "built-in summary"
    assert harness.get_pending_response_count() == 0


@pytest.mark.tonio
async def test_uses_the_newly_selected_model_without_changing_ordinary_settings(harnesses):
    harness = await create_harness(
        models=[{"id": "small", "context_window": 4000}, {"id": "big", "context_window": 10000}],
        tools=[],
        settings={
            "compaction": {
                "reserveTokens": 10,
                "modelOverrides": {"faux/big": {"reserveTokens": 8000, "keepRecentTokens": 150}},
            }
        },
        extension_factories=[_compaction_factory([], "big model summary")],
    )
    harnesses.append(harness)
    await _seed_history(harness, 2500)
    harness.set_responses([faux_assistant_message("small response"), faux_assistant_message("big response")])
    await harness.session.prompt("continue on small")
    assert harness.events_of_type("compaction_start") == []
    # Retain usage from the small model: the next check must use the active big model's policy.
    await _seed_history(harness, 2500)
    await harness.session.set_model(harness.get_model("big"))
    await harness.session.prompt("continue on big")

    compaction_ends = harness.events_of_type("compaction_end")
    assert len(compaction_ends) == 1
    assert compaction_ends[0].result.summary == "big model summary"
    assert harness.settings_manager.get_compaction_reserve_tokens() == 10
    await harness.session.set_model(harness.get_model("small"))
    assert harness.settings_manager.get_compaction_reserve_tokens(harness.session.model) == 10


@pytest.mark.tonio
async def test_captures_model_identity_before_awaiting_summarization_auth(harnesses):
    harness = await create_harness(
        models=[{"id": "first"}, {"id": "second"}],
        settings={
            "compaction": {
                "modelOverrides": {
                    "faux/first": {"reserveTokens": 2000, "keepRecentTokens": 150},
                    "faux/second": {"reserveTokens": 4000, "keepRecentTokens": 20000},
                }
            }
        },
    )
    harnesses.append(harness)
    await _seed_history(harness)
    runtime = harness.session.model_runtime
    get_auth = runtime.get_auth

    async def switching_get_auth(model, *args):
        harness.session.agent.state.model = harness.get_model("second")
        return await get_auth(model, *args)

    runtime.get_auth = switching_get_auth
    requests: list = []

    async def summarize(_context, options, _state, model):
        requests.append({"id": model.id, "max_tokens": options.max_tokens if options is not None else None})
        return faux_assistant_message("summary")

    harness.set_responses([summarize])
    await harness.session.compact()

    assert requests == [{"id": "first", "max_tokens": 1600}]
