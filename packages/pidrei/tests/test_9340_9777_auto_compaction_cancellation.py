"""Mirror of pi's suite/regressions/9340-9777-auto-compaction-cancellation.test.ts.

pi's `void harness.session.abort()` inside a listener runs abort's synchronous
prefix before the listener returns; `_request_abort()` is that prefix here.
pi's "reports ... as a failure" cases reject `getAuth`, which fails compaction
only through pi's compat-`streamSimple` required-auth branch; pidrei has no
compat module (its summarization auth always degrades to unauthenticated), so
the same errors are raised from the summary generator instead — the point is
unchanged: `aborted` comes from the cancel token, not from the error text.
"""

import dataclasses

import pytest
import tonio.colored as tonio

from pidrei_ai.providers.faux import faux_assistant_message
from pidrei_ai.types import TextContent, Usage, UsageCost, UserMessage
from pidrei_ai.utils.cancel import AbortError

from .harness import create_harness


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


async def seed_compactable_session(harness) -> None:
    model = harness.get_model()
    await harness.session_manager.append_message(UserMessage(content=[TextContent(text="x" * 500)], timestamp=1))
    assistant = dataclasses.replace(
        faux_assistant_message("y" * 200, timestamp=2),
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(input=100, total_tokens=100, cost=UsageCost()),
    )
    await harness.session_manager.append_message(assistant)
    harness.session.agent.state.messages = harness.session_manager.build_session_context().messages


def run_auto_compaction(harness):
    return harness.session._run_auto_compaction("threshold", False)


async def _cancelling_extension(pi) -> None:
    async def on_before_compact(_event, _ctx):
        return {"cancel": True}

    pi.on("session_before_compact", on_before_compact)


# Regression test for #9340.
@pytest.mark.tonio
async def test_does_not_start_post_run_auto_compaction_after_abort(harnesses):
    harness = await create_harness(
        models=[{"id": "faux-1", "context_window": 200, "max_tokens": 50}],
        settings={
            "compaction": {"enabled": True, "reserveTokens": 50, "keepRecentTokens": 1},
            "retry": {"enabled": False},
        },
        extension_factories=[_cancelling_extension],
    )
    harnesses.append(harness)
    await seed_compactable_session(harness)
    harness.set_responses([faux_assistant_message("", stop_reason="error", error_message="Synthetic network failure")])

    def listener(event) -> None:
        if event.type == "message_end" and getattr(event.message, "role", None) == "assistant":
            harness.session.abort_compaction()
            harness.session._request_abort()

    harness.session.subscribe(listener)

    await harness.session.prompt("z" * 1000)

    assert harness.events_of_type("compaction_start") == []


# Regression test for #9777.
@pytest.mark.tonio
async def test_cancels_summarization_authentication(harnesses, monkeypatch):
    harness = await create_harness(settings={"compaction": {"keepRecentTokens": 1}})
    harnesses.append(harness)
    await seed_compactable_session(harness)
    auth_started = tonio.Event()
    seen: dict = {}

    async def get_auth(_model, overrides=None):
        cancel = overrides.cancel if overrides is not None else None
        seen["cancel"] = cancel
        auth_started.set()
        if cancel is None:
            raise Exception("Missing auth cancel token")
        await cancel.wait()
        raise cancel.reason

    monkeypatch.setattr(harness.session.model_runtime, "get_auth", get_auth)

    compaction = tonio.spawn(run_auto_compaction(harness))
    await auth_started.wait(5)
    assert auth_started.is_set()
    started = len(harness.events_of_type("compaction_start"))
    was_compacting = harness.session.is_compacting
    await harness.session.abort()
    await compaction

    assert (started, was_compacting, seen["cancel"].cancelled) == (1, True, True)
    assert harness.events_of_type("compaction_end")[-1].aborted is True


# Regression test for #9777.
@pytest.mark.tonio
async def test_cancels_synchronously_from_compaction_start(harnesses):
    harness = await create_harness(settings={"compaction": {"keepRecentTokens": 1}})
    harnesses.append(harness)
    await seed_compactable_session(harness)

    def listener(event) -> None:
        if event.type == "compaction_start":
            harness.session.abort_compaction()

    harness.session.subscribe(listener)

    await run_auto_compaction(harness)

    assert harness.faux.state.call_count == 0
    assert harness.events_of_type("compaction_end")[-1].aborted is True


# Regression test for #9777.
@pytest.mark.tonio
@pytest.mark.parametrize(
    "create_error",
    [lambda: Exception("Compaction cancelled"), lambda: AbortError("auth failed")],
    ids=["matching error text", "an unrelated AbortError"],
)
async def test_reports_errors_as_a_failure(harnesses, monkeypatch, create_error):
    harness = await create_harness(settings={"compaction": {"keepRecentTokens": 1}})
    harnesses.append(harness)
    await seed_compactable_session(harness)

    async def failing_compaction(*_args, **_kwargs):
        raise create_error()

    monkeypatch.setattr(harness.session, "_run_default_compaction", failing_compaction)

    await run_auto_compaction(harness)

    event = harness.events_of_type("compaction_end")[-1]
    assert event.aborted is False
    assert str(create_error()) in event.error_message


@pytest.mark.tonio
async def test_reports_extension_cancellation_as_aborted(harnesses):
    harness = await create_harness(
        settings={"compaction": {"keepRecentTokens": 1}}, extension_factories=[_cancelling_extension]
    )
    harnesses.append(harness)
    await seed_compactable_session(harness)

    await run_auto_compaction(harness)

    assert harness.events_of_type("compaction_end")[-1].aborted is True
