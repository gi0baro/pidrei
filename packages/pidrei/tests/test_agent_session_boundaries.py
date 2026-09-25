"""Mirror of pi's suite/agent-session-boundaries.test.ts.

pi checks `JSON.stringify(context.messages)` of each provider request for
present/absent text; the dataclass `repr` carries the same content here.
Spying on `_runAutoCompaction` becomes an instance-attribute replacement.
"""

import dataclasses
import time

import pytest
import tonio.colored as tonio

from pidrei.core.compaction import CompactionResult
from pidrei.core.extensions import ToolDefinition
from pidrei_agent.types import AgentToolResult
from pidrei_ai.providers.faux import faux_assistant_message, faux_tool_call
from pidrei_ai.types import TextContent, UserMessage

from .harness import create_harness, get_message_text


def _now() -> int:
    return int(time.time() * 1000)


def _dump(messages) -> str:
    return repr(list(messages))


def _recording(requests: list[str], text: str, **options):
    async def respond(context, *_rest):
        requests.append(_dump(context.messages))
        return faux_assistant_message(text, **options)

    return respond


def _with_usage(message, **usage):
    return dataclasses.replace(message, usage=dataclasses.replace(message.usage, **usage))


def _latest_user_entry(session_manager) -> dict:
    user = next(
        (
            entry
            for entry in reversed(session_manager.get_branch())
            if entry["type"] == "message" and entry["message"].role == "user"
        ),
        None,
    )
    if user is None:
        raise Exception("missing user entry")
    return user


def _signal_queued(session, event: tonio.Event, *texts: str) -> None:
    """Set `event` once every text sits in the session queues.

    pi's `sendUserMessage` queues within the handler's microtask chain, before
    the loop polls again; pidrei spawns it, so handlers await this instead.
    """

    def listener(update) -> None:
        if update.type == "queue_update" and all(text in (*update.steering, *update.follow_up) for text in texts):
            event.set()

    session.subscribe(listener)


def _noop_tool(name: str = "noop", execute=None) -> ToolDefinition:
    async def default_execute(*_args):
        return AgentToolResult(content=[TextContent(text="done")], details={})

    return ToolDefinition(
        name=name,
        label=name,
        description=name,
        parameters={"type": "object", "properties": {}},
        execute=execute if execute is not None else default_execute,
    )


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


# -- AgentSession actionable boundaries -------------------------------------------------


@pytest.mark.tonio
async def test_commits_a_retain_none_turn_end_compaction_and_explicitly_continues_once(harnesses):
    handled = False
    observed_ids: list[str] = []
    requests: list[str] = []

    async def factory(pi) -> None:
        async def on_turn_end(event, _ctx):
            nonlocal handled
            observed_ids.append(event["messageEntryId"])
            if handled:
                return None
            handled = True
            return {
                "entries": [
                    {
                        "type": "compaction",
                        "summary": "exact handoff",
                        "firstKeptEntryId": None,
                        "details": {"source": "test"},
                    }
                ],
                "continue": True,
            }

        pi.on("turn_end", on_turn_end)

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)
    harness.set_responses(
        [faux_assistant_message("discarded response"), _recording(requests, "continued from handoff")]
    )

    await harness.session.prompt("discarded prompt")

    compaction = next(entry for entry in harness.session_manager.get_entries() if entry["type"] == "compaction")
    assert compaction["summary"] == "exact handoff"
    assert compaction["firstKeptEntryId"] == compaction["id"]
    assert len(requests) == 1
    assert "exact handoff" in requests[0]
    assert "discarded prompt" not in requests[0]
    assert "discarded response" not in requests[0]
    assert len(observed_ids) == 2
    assert len(harness.events_of_type("agent_settled")) == 1


@pytest.mark.tonio
@pytest.mark.parametrize("queue_kind", ["steering", "follow-up", "both"])
async def test_preserves_queue_scheduling_around_a_turn_end_handoff(harnesses, queue_kind):
    handled = False
    requests: list[str] = []
    queued = tonio.Event()
    texts = {
        "steering": ("queued steering",),
        "follow-up": ("queued follow-up",),
        "both": ("queued steering", "queued follow-up"),
    }[queue_kind]

    async def factory(pi) -> None:
        async def on_turn_end(_event, _ctx):
            nonlocal handled
            if handled:
                return None
            handled = True
            if queue_kind in ("steering", "both"):
                pi.send_user_message("queued steering", {"deliverAs": "steer"})
            if queue_kind in ("follow-up", "both"):
                pi.send_user_message("queued follow-up", {"deliverAs": "followUp"})
            await queued.wait(5)
            return {
                "entries": [{"type": "compaction", "summary": "exact handoff", "firstKeptEntryId": None}],
                "continue": True,
            }

        pi.on("turn_end", on_turn_end)

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)
    _signal_queued(harness.session, queued, *texts)
    harness.set_responses(
        [faux_assistant_message("first"), _recording(requests, "second"), _recording(requests, "third")]
    )

    await harness.session.prompt("start")

    assert queued.is_set()
    assert "exact handoff" in requests[0]
    if queue_kind == "steering":
        assert harness.faux.state.call_count == 2
        assert "queued steering" in requests[0]
        assert "queued follow-up" not in requests[0]
    elif queue_kind == "follow-up":
        assert harness.faux.state.call_count == 2
        assert "queued follow-up" in requests[0]
    else:
        assert harness.faux.state.call_count == 3
        assert "queued steering" in requests[0]
        assert "queued follow-up" not in requests[0]
        assert "queued follow-up" in requests[1]


@pytest.mark.tonio
async def test_keeps_a_boundary_replacement_verbatim_through_threshold_compaction(harnesses):
    handled = False
    requests: list[str] = []
    instruction = "EXACT-REPLACEMENT-INSTRUCTION " * 100

    async def factory(pi) -> None:
        async def on_turn_end(event, ctx):
            nonlocal handled
            if handled:
                return None
            handled = True
            user = _latest_user_entry(ctx.session_manager)
            return {
                "entries": [
                    {"type": "context_edit", "targetId": user["id"], "replacement": {"content": instruction}},
                    {"type": "context_edit", "targetId": event["messageEntryId"], "replacement": None},
                    {"type": "custom", "customType": "bookkeeping", "data": {"source": "test"}},
                ],
                "continue": True,
            }

        pi.on("turn_end", on_turn_end)

    harness = await create_harness(
        models=[{"id": "faux-1", "context_window": 2_000, "max_tokens": 100}],
        settings={"compaction": {"enabled": True, "keepRecentTokens": 1, "reserveTokens": 0}},
        extension_factories=[factory, _compaction_summary_factory("older history summary")],
    )
    harnesses.append(harness)
    await harness.session_manager.append_message(UserMessage(content="older input", timestamp=_now() - 2))
    await harness.session_manager.append_message(faux_assistant_message("older answer", timestamp=_now() - 1))
    harness.session.refresh_context()
    harness.set_responses(
        [faux_assistant_message("answered original input"), _recording(requests, "answered replacement")]
    )

    await harness.session.prompt("original input")

    assert len(harness.events_of_type("compaction_start")) > 0
    assert len(requests) == 1
    assert "EXACT-REPLACEMENT-INSTRUCTION" in requests[0]


def _compaction_summary_factory(summary: str):
    async def factory(pi) -> None:
        async def on_before_compact(event, _ctx):
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
async def test_keeps_boundary_input_verbatim_through_threshold_compaction_when_metadata_follows_it(harnesses):
    handled = False
    requests: list[str] = []
    instruction = "EXACT-UNSENT-INSTRUCTION " * 100

    async def factory(pi) -> None:
        async def on_turn_end(_event, _ctx):
            nonlocal handled
            if handled:
                return None
            handled = True
            return {
                "entries": [
                    {"type": "custom_message", "customType": "next-work", "content": instruction, "display": False},
                    {"type": "custom", "customType": "bookkeeping", "data": {"source": "test"}},
                ],
                "continue": True,
            }

        pi.on("turn_end", on_turn_end)

    harness = await create_harness(
        models=[{"id": "faux-1", "context_window": 2_000, "max_tokens": 100}],
        settings={"compaction": {"enabled": True, "keepRecentTokens": 1, "reserveTokens": 0}},
        extension_factories=[factory, _compaction_summary_factory("older history summary")],
    )
    harnesses.append(harness)
    harness.set_responses([faux_assistant_message("first"), _recording(requests, "second")])

    await harness.session.prompt("old input " * 500)

    assert len(harness.events_of_type("compaction_start")) > 0
    assert len(requests) == 1
    assert "EXACT-UNSENT-INSTRUCTION" in requests[0]


@pytest.mark.tonio
async def test_refreshes_canonical_context_before_publishing_boundary_entry_notifications(harnesses):
    handled = False
    snapshots: list[str] = []

    async def factory(pi) -> None:
        async def on_turn_end(_event, _ctx):
            nonlocal handled
            if handled:
                return None
            handled = True
            return {
                "entries": [
                    {"type": "custom", "customType": "metadata", "data": True},
                    {
                        "type": "custom_message",
                        "customType": "visible-context",
                        "content": "committed context",
                        "display": True,
                    },
                ]
            }

        pi.on("turn_end", on_turn_end)

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)

    def listener(event) -> None:
        if event.type == "entry_appended":
            snapshots.append(_dump(harness.session.messages))

    harness.session.subscribe(listener)
    harness.set_responses([faux_assistant_message("done")])

    await harness.session.prompt("start")

    assert len(snapshots) == 2
    assert all("committed context" in snapshot for snapshot in snapshots)


@pytest.mark.tonio
async def test_continues_from_an_agent_before_settle_custom_message_before_final_settlement(harnesses):
    requested = False
    requests: list[str] = []

    async def factory(pi) -> None:
        async def on_before_settle(_event, _ctx):
            nonlocal requested
            if requested:
                return None
            requested = True
            return {
                "entries": [
                    {
                        "type": "custom_message",
                        "customType": "test-continuation",
                        "content": "continue now",
                        "display": False,
                    }
                ],
                "continue": True,
            }

        pi.on("agent_before_settle", on_before_settle)

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)
    harness.set_responses([faux_assistant_message("first"), _recording(requests, "second")])

    await harness.session.prompt("start")

    assert "continue now" in requests[0]
    assert any(
        entry["type"] == "custom_message" and entry["customType"] == "test-continuation" and entry["display"] is False
        for entry in harness.session_manager.get_entries()
    )
    assert len(harness.events_of_type("agent_start")) == 2
    assert len(harness.events_of_type("agent_settled")) == 1


@pytest.mark.tonio
async def test_persists_custom_context_queued_by_agent_end_before_pre_settlement_continuation(harnesses):
    first_run = True
    continued = False
    requests: list[str] = []
    observations: list[tuple[str, str]] = []

    async def factory(pi) -> None:
        async def on_agent_end(_event, _ctx):
            nonlocal first_run
            if not first_run:
                return
            first_run = False
            pi.send_message(
                {"customType": "agent-end-context", "content": "queued after agent end", "display": False},
                {"triggerTurn": False},
            )

        async def on_before_settle(event, _ctx):
            nonlocal continued
            if continued:
                return None
            continued = True
            observations.append((_dump(event["context"].pending_messages), _dump(event["context"].context_messages)))
            return {"continue": True}

        pi.on("agent_end", on_agent_end)
        pi.on("agent_before_settle", on_before_settle)

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)
    harness.set_responses([faux_assistant_message("first"), _recording(requests, "second")])

    await harness.session.prompt("start")

    [(pending, context)] = observations
    assert "queued after agent end" in pending
    assert "queued after agent end" not in context
    assert "queued after agent end" in requests[0]
    assert any(
        entry["type"] == "custom_message" and entry["customType"] == "agent-end-context"
        for entry in harness.session_manager.get_entries()
    )


@pytest.mark.tonio
async def test_keeps_a_pre_settlement_follow_up_deferred_until_the_explicit_continuation_would_stop(harnesses):
    handled = False
    requests: list[str] = []
    queued = tonio.Event()

    async def factory(pi) -> None:
        async def on_before_settle(_event, _ctx):
            nonlocal handled
            if handled:
                return None
            handled = True
            pi.send_user_message("queued follow-up", {"deliverAs": "followUp"})
            await queued.wait(5)
            return {
                "entries": [
                    {
                        "type": "custom_message",
                        "customType": "boundary",
                        "content": "boundary context",
                        "display": False,
                    }
                ],
                "continue": True,
            }

        pi.on("agent_before_settle", on_before_settle)

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)
    _signal_queued(harness.session, queued, "queued follow-up")
    harness.set_responses(
        [faux_assistant_message("first"), _recording(requests, "second"), _recording(requests, "follow-up response")]
    )

    await harness.session.prompt("start")

    assert queued.is_set()
    assert harness.faux.state.call_count == 3
    assert "boundary context" in requests[0]
    assert "queued follow-up" not in requests[0]
    assert "queued follow-up" in requests[1]


@pytest.mark.tonio
async def test_defers_runs_started_by_agent_settled_handlers_until_every_settled_handler_completes(harnesses):
    triggered = False
    lifecycle: list[str] = []

    async def factory(pi) -> None:
        async def on_start(_event, _ctx):
            lifecycle.append("start")

        async def settled_first(_event, ctx):
            nonlocal triggered
            lifecycle.append(f"settled-first:{str(ctx.is_idle()).lower()}")
            if triggered:
                return
            triggered = True
            pi.send_message(
                {"customType": "settled-trigger", "content": "start later", "display": False},
                {"triggerTurn": True},
            )

        async def settled_second(_event, ctx):
            lifecycle.append(f"settled-second:{str(ctx.is_idle()).lower()}")

        pi.on("agent_start", on_start)
        pi.on("agent_settled", settled_first)
        pi.on("agent_settled", settled_second)

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)
    harness.set_responses([faux_assistant_message("first"), faux_assistant_message("second")])

    await harness.session.prompt("start")

    assert lifecycle == [
        "start",
        "settled-first:true",
        "settled-second:true",
        "start",
        "settled-first:true",
        "settled-second:true",
    ]


@pytest.mark.tonio
async def test_does_not_let_an_invalid_explicit_continuation_suppress_natural_tool_continuation(harnesses):
    async def factory(pi) -> None:
        async def on_turn_end(event, ctx):
            user = _latest_user_entry(ctx.session_manager)
            return {
                "entries": [
                    {"type": "context_edit", "targetId": target_id, "replacement": None}
                    for target_id in [user["id"], event["messageEntryId"], *event["toolResultEntryIds"]]
                ],
                "continue": True,
            }

        pi.on("turn_end", on_turn_end)

    harness = await create_harness(tools=[_noop_tool()], extension_factories=[factory])
    harnesses.append(harness)
    harness.set_responses(
        [
            faux_assistant_message(faux_tool_call("noop", {}), stop_reason="toolUse"),
            faux_assistant_message("must not run"),
        ]
    )

    await harness.session.prompt("start")

    assert harness.faux.state.call_count == 2


@pytest.mark.tonio
async def test_dispatches_actionable_turn_end_for_synthetic_run_failures(harnesses):
    turn_ends = 0
    outcomes: list[str] = []

    async def factory(pi) -> None:
        async def on_turn_end(event, _ctx):
            nonlocal turn_ends
            turn_ends += 1
            outcomes.append(event["outcome"])
            return {"entries": [{"type": "custom", "customType": "failure-boundary", "data": True}]}

        pi.on("turn_end", on_turn_end)

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)

    async def failing_prepare_request(_request, _cancel=None):
        raise Exception("request preparation failed")

    harness.session.agent.prepare_request = failing_prepare_request

    await harness.session.prompt("start")

    assert turn_ends == 1
    assert outcomes == ["error"]
    assert harness.faux.state.call_count == 0
    assert any(
        entry["type"] == "custom" and entry["customType"] == "failure-boundary"
        for entry in harness.session_manager.get_entries()
    )


def _inflate_assistant_usage(**usage):
    async def on_message_end(event, _ctx):
        if event["message"].role != "assistant":
            return None
        return {"message": _with_usage(event["message"], **usage)}

    return on_message_end


@pytest.mark.tonio
async def test_does_not_compact_from_usage_belonging_to_a_boundary_omitted_assistant(harnesses):
    handled = False

    async def factory(pi) -> None:
        async def on_turn_end(event, _ctx):
            nonlocal handled
            if handled:
                return None
            handled = True
            return {"entries": [{"type": "context_edit", "targetId": event["messageEntryId"], "replacement": None}]}

        pi.on("message_end", _inflate_assistant_usage(input=9_800, output=1, total_tokens=9_801))
        pi.on("turn_end", on_turn_end)

    harness = await create_harness(
        models=[{"id": "faux-1", "context_window": 10_000, "max_tokens": 100}],
        settings={"compaction": {"enabled": True, "keepRecentTokens": 1, "reserveTokens": 300}},
        extension_factories=[factory],
    )
    harnesses.append(harness)
    harness.set_responses([faux_assistant_message("short response")])

    await harness.session.prompt("small prompt")

    assert harness.events_of_type("compaction_start") == []
    assert harness.session.get_context_usage().tokens < 2_000


@pytest.mark.tonio
async def test_does_not_trigger_successful_response_overflow_from_usage_captured_before_a_boundary_edit(harnesses):
    handled = False

    async def factory(pi) -> None:
        async def on_turn_end(_event, ctx):
            nonlocal handled
            if handled:
                return None
            handled = True
            user = _latest_user_entry(ctx.session_manager)
            return {"entries": [{"type": "context_edit", "targetId": user["id"], "replacement": None}]}

        pi.on("message_end", _inflate_assistant_usage(input=5_100, output=1, total_tokens=5_101))
        pi.on("turn_end", on_turn_end)

    harness = await create_harness(
        models=[{"id": "faux-1", "context_window": 5_000, "max_tokens": 100}],
        settings={"compaction": {"enabled": True, "keepRecentTokens": 1, "reserveTokens": 0}},
        extension_factories=[factory],
    )
    harnesses.append(harness)
    harness.set_responses([faux_assistant_message("done")])

    await harness.session.prompt("large input that is later omitted")

    assert harness.events_of_type("compaction_start") == []
    assert harness.session.get_context_usage().tokens < 2_000


@pytest.mark.tonio
async def test_does_not_trigger_threshold_compaction_from_post_edit_usage_captured_before_a_later_compaction(
    harnesses,
):
    harness = await create_harness(
        models=[{"id": "faux-1", "context_window": 10_000, "max_tokens": 100}],
        settings={"compaction": {"enabled": True, "keepRecentTokens": 1, "reserveTokens": 0}},
    )
    harnesses.append(harness)
    user_id = await harness.session_manager.append_message(UserMessage(content="small input", timestamp=_now() - 3))
    await harness.session_manager.append_context_edit(user_id, {"content": "edited input"})
    response = _with_usage(
        faux_assistant_message("answer", timestamp=_now() - 2), input=50_000, output=1, total_tokens=50_001
    )
    await harness.session_manager.append_message(response)
    await harness.session_manager.append_compaction("small summary", user_id, 50_001)
    harness.session.refresh_context()
    auto_compaction_calls: list[tuple] = []

    # `_abort_generation`: pidrei-only (the abort baseline `_check_compaction` hands over).
    async def run_auto_compaction(reason, will_retry, _abort_generation=None):
        auto_compaction_calls.append((reason, will_retry))
        return False

    harness.session._run_auto_compaction = run_auto_compaction
    error = faux_assistant_message("", stop_reason="error", error_message="invalid_api_key", timestamp=_now() + 1_000)

    await harness.session._check_compaction(error)

    assert auto_compaction_calls == []


@pytest.mark.tonio
async def test_does_not_treat_retained_pre_compaction_assistant_usage_as_post_compaction_usage(harnesses):
    harness = await create_harness()
    harnesses.append(harness)
    retained = _with_usage(faux_assistant_message("retained"), input=10_000, total_tokens=10_001)
    retained_id = await harness.session_manager.append_message(retained)
    await harness.session_manager.append_compaction("summary", retained_id, 10_001)
    harness.session.refresh_context()

    assert harness.session.get_context_usage().tokens is None


@pytest.mark.tonio
async def test_persists_custom_context_sent_during_pre_settlement_before_continuing(harnesses):
    handled = False
    requests: list[str] = []

    async def factory(pi) -> None:
        async def on_before_settle(_event, _ctx):
            nonlocal handled
            if handled:
                return None
            handled = True
            pi.send_message(
                {"customType": "pending-boundary", "content": "persist before continue", "display": False},
                {"triggerTurn": False},
            )
            return {"continue": True}

        pi.on("agent_before_settle", on_before_settle)

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)
    harness.set_responses([faux_assistant_message("first"), _recording(requests, "second")])

    await harness.session.prompt("start")

    assert harness.faux.state.call_count == 2
    assert "persist before continue" in requests[0]
    assert any(
        entry["type"] == "custom_message" and entry["customType"] == "pending-boundary"
        for entry in harness.session_manager.get_entries()
    )


@pytest.mark.tonio
async def test_does_not_consume_queued_input_when_pre_settlement_drafts_leave_system_only_context(harnesses):
    handled = False
    queued = tonio.Event()

    async def factory(pi) -> None:
        async def on_before_settle(_event, ctx):
            nonlocal handled
            if handled:
                return None
            handled = True
            pi.send_user_message("still queued", {"deliverAs": "followUp"})
            await queued.wait(5)
            targets = [
                entry["id"]
                for entry in ctx.session_manager.get_branch()
                if entry["type"] == "message" and entry["message"].role in ("user", "assistant", "toolResult")
            ]
            return {
                "entries": [
                    {"type": "context_edit", "targetId": target_id, "replacement": None} for target_id in targets
                ],
                "continue": True,
            }

        pi.on("agent_before_settle", on_before_settle)

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)
    _signal_queued(harness.session, queued, "still queued")
    harness.set_responses([faux_assistant_message("first"), faux_assistant_message("must not run")])

    await harness.session.prompt("start")

    assert queued.is_set()
    assert harness.faux.state.call_count == 1
    assert harness.session.pending_message_count == 1


@pytest.mark.tonio
async def test_commits_pre_settlement_drafts_but_suppresses_continuation_when_aborted_during_the_hook(harnesses):
    started = tonio.Event()
    release = tonio.Event()

    async def factory(pi) -> None:
        async def on_before_settle(_event, _ctx):
            started.set()
            await release.wait()
            return {
                "entries": [{"type": "custom", "customType": "committed-after-abort", "data": True}],
                "continue": True,
            }

        pi.on("agent_before_settle", on_before_settle)

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)
    harness.set_responses([faux_assistant_message("first"), faux_assistant_message("must not run")])

    prompt = tonio.spawn(harness.session.prompt("start"))
    await started.wait(5)
    assert started.is_set()
    # pi's `session.abort()` runs its synchronous prefix before `release()`;
    # `_request_abort` is that prefix, and the idle wait is the rest of `abort()`.
    harness.session._request_abort()
    abort = tonio.spawn(harness.session.wait_for_idle())
    release.set()
    await prompt
    await abort

    assert harness.faux.state.call_count == 1
    assert any(
        entry["type"] == "custom" and entry["customType"] == "committed-after-abort" and entry["data"] is True
        for entry in harness.session_manager.get_entries()
    )
    assert len(harness.events_of_type("agent_settled")) == 1


# -- durable length recovery ------------------------------------------------------------


@pytest.mark.tonio
async def test_keeps_truncated_tool_attempts_in_context_for_the_natural_next_turn(harnesses):
    executed = False
    requests: list[str] = []

    async def execute(*_args):
        nonlocal executed
        executed = True
        return AgentToolResult(content=[TextContent(text="executed")], details={})

    tool = ToolDefinition(
        name="unsafe_truncated_tool",
        label="Unsafe truncated tool",
        description="Must not execute from a length response",
        parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
        execute=execute,
    )
    harness = await create_harness(tools=[tool])
    harnesses.append(harness)
    harness.set_responses(
        [
            faux_assistant_message(faux_tool_call("unsafe_truncated_tool", {"value": "partial"}), stop_reason="length"),
            _recording(requests, "completed natural continuation"),
        ]
    )

    await harness.session.prompt("start")

    assert executed is False
    assert harness.faux.state.call_count == 2
    assert "may be truncated" in requests[0]
    assert not any(entry["type"] == "context_edit" for entry in harness.session_manager.get_entries())


def _length_ids(session_manager) -> list[str]:
    return [
        entry["id"]
        for entry in session_manager.get_entries()
        if entry["type"] == "message"
        and entry["message"].role == "assistant"
        and entry["message"].stop_reason == "length"
    ]


def _omitted_ids(session_manager) -> list[str]:
    return [entry["targetId"] for entry in session_manager.get_entries() if entry["type"] == "context_edit"]


def _later(text: str, offset_ms: int, **options):
    async def respond(*_args):
        return faux_assistant_message(text, timestamp=_now() + offset_ms, **options)

    return respond


@pytest.mark.tonio
async def test_resets_length_recovery_after_a_successful_intermediate_assistant_turn(harnesses):
    harness = await create_harness(
        models=[{"id": "faux-1", "context_window": 1000, "max_tokens": 100}],
        settings={"compaction": {"keepRecentTokens": 1, "reserveTokens": 0}},
        tools=[_noop_tool()],
        extension_factories=[_compaction_summary_factory("recovered input")],
    )
    harnesses.append(harness)
    harness.set_responses(
        [
            faux_assistant_message("first partial", stop_reason="length"),
            faux_assistant_message(faux_tool_call("noop", {}), stop_reason="toolUse"),
            _later("second partial", 1_000, stop_reason="length"),
            _later("completed second recovery", 2_000),
        ]
    )

    await harness.session.prompt("x" * 5000)

    length_ids = _length_ids(harness.session_manager)
    assert len(length_ids) == 2
    assert set(length_ids) <= set(_omitted_ids(harness.session_manager))
    assert harness.faux.state.call_count == 3


@pytest.mark.tonio
async def test_gives_a_distinct_queued_follow_up_its_own_length_recovery_budget(harnesses):
    sent = False
    queued = tonio.Event()

    async def factory(pi) -> None:
        async def on_agent_end(event, _ctx):
            nonlocal sent
            if sent or not any(get_message_text(message) == "first recovered" for message in event["messages"]):
                return
            sent = True
            pi.send_user_message("distinct follow-up", {"deliverAs": "followUp"})
            await queued.wait(5)

        pi.on("agent_end", on_agent_end)

    harness = await create_harness(
        models=[{"id": "faux-1", "context_window": 1000, "max_tokens": 100}],
        settings={"compaction": {"keepRecentTokens": 1, "reserveTokens": 0}},
        extension_factories=[_compaction_summary_factory("recovered input"), factory],
    )
    harnesses.append(harness)
    _signal_queued(harness.session, queued, "distinct follow-up")
    harness.set_responses(
        [
            faux_assistant_message("first partial", stop_reason="length"),
            faux_assistant_message("first recovered"),
            _later("follow-up partial", 1_000, stop_reason="length"),
            _later("follow-up recovered", 2_000),
        ]
    )

    await harness.session.prompt("x" * 5000)

    length_ids = _length_ids(harness.session_manager)
    assert len(length_ids) == 2
    assert set(length_ids) <= set(_omitted_ids(harness.session_manager))
    assert harness.faux.state.call_count == 4


@pytest.mark.tonio
async def test_finishes_retry_bookkeeping_when_a_retry_receives_a_nonretryable_error(harnesses):
    harness = await create_harness(settings={"retry": {"enabled": True, "maxRetries": 2, "baseDelayMs": 1}})
    harnesses.append(harness)
    harness.set_responses(
        [
            faux_assistant_message("", stop_reason="error", error_message="overloaded_error"),
            faux_assistant_message("", stop_reason="error", error_message="invalid_api_key"),
        ]
    )

    await harness.session.prompt("start")

    assert harness.faux.state.call_count == 2
    assert any(
        event.success is False and event.attempt == 1 and event.final_error == "invalid_api_key"
        for event in harness.events_of_type("auto_retry_end")
    )


@pytest.mark.tonio
async def test_omits_a_recoverable_projected_replacement_by_its_source_entry_id(harnesses):
    async def cancel_factory(pi) -> None:
        async def on_before_compact(_event, _ctx):
            return {"cancel": True}

        pi.on("session_before_compact", on_before_compact)

    harness = await create_harness(
        models=[{"id": "faux-1", "context_window": 1_000, "max_tokens": 100}],
        settings={"compaction": {"enabled": True, "keepRecentTokens": 1, "reserveTokens": 0}},
        extension_factories=[cancel_factory],
    )
    harnesses.append(harness)
    await harness.session_manager.append_message(UserMessage(content="x" * 5_000, timestamp=_now() - 2))
    partial = faux_assistant_message("original partial", stop_reason="length", timestamp=_now() - 1)
    partial_id = await harness.session_manager.append_message(partial)
    await harness.session_manager.append_context_edit(partial_id, {"content": [TextContent(text="edited partial")]})
    harness.session.refresh_context()
    harness.set_responses([faux_assistant_message("new answer")])

    await harness.session.prompt("next prompt")

    edits = [
        entry
        for entry in harness.session_manager.get_entries()
        if entry["type"] == "context_edit" and entry["targetId"] == partial_id
    ]
    assert edits[-1]["replacement"] is None
    assert not any(
        get_message_text(message) == "edited partial"
        for message in harness.session_manager.build_session_projection().messages
    )


@pytest.mark.tonio
async def test_recovers_an_explicit_overflow_error_after_a_retained_boundary_replacement(harnesses):
    replaced = False
    overflow_id: str | None = None

    async def factory(pi) -> None:
        async def on_turn_end(event, _ctx):
            nonlocal replaced, overflow_id
            if replaced or event["outcome"] != "error":
                return None
            replaced = True
            overflow_id = event["messageEntryId"]
            return {
                "entries": [
                    {
                        "type": "context_edit",
                        "targetId": event["messageEntryId"],
                        "replacement": {"content": [TextContent(text="retained error")]},
                    }
                ]
            }

        pi.on("turn_end", on_turn_end)

    harness = await create_harness(
        models=[{"id": "faux-1", "context_window": 1_000, "max_tokens": 100}],
        settings={"compaction": {"enabled": True, "keepRecentTokens": 1, "reserveTokens": 0}},
        extension_factories=[factory, _compaction_summary_factory("recovered overflow")],
    )
    harnesses.append(harness)
    harness.set_responses(
        [
            faux_assistant_message("retained error", stop_reason="error", error_message="prompt is too long"),
            faux_assistant_message("recovered"),
        ]
    )

    await harness.session.prompt("x" * 5_000)

    assert harness.faux.state.call_count == 2
    assert overflow_id is not None
    edits = [
        entry
        for entry in harness.session_manager.get_entries()
        if entry["type"] == "context_edit" and entry["targetId"] == overflow_id
    ]
    assert edits[-1]["replacement"] is None


@pytest.mark.tonio
async def test_keeps_follow_up_work_behind_an_automatic_error_retry(harnesses):
    sent = False
    queued = tonio.Event()
    requests: list[str] = []
    lifecycle: list[str] = []

    async def factory(pi) -> None:
        async def on_turn_end(event, _ctx):
            nonlocal sent
            if sent or event["outcome"] != "error":
                return
            sent = True
            pi.send_user_message("queued follow-up", {"deliverAs": "followUp"})
            await queued.wait(5)

        pi.on("turn_end", on_turn_end)

    harness = await create_harness(
        settings={"retry": {"enabled": True, "maxRetries": 2, "baseDelayMs": 1}}, extension_factories=[factory]
    )
    harnesses.append(harness)
    _signal_queued(harness.session, queued, "queued follow-up")

    def listener(event) -> None:
        if event.type in ("agent_end", "auto_retry_start"):
            lifecycle.append(event.type)

    harness.session.subscribe(listener)
    harness.set_responses(
        [
            faux_assistant_message("", stop_reason="error", error_message="overloaded_error"),
            _recording(requests, "retry recovered"),
            _recording(requests, "follow-up completed"),
        ]
    )

    await harness.session.prompt("start")

    assert queued.is_set()
    assert harness.faux.state.call_count == 3
    assert "queued follow-up" not in requests[0]
    assert "queued follow-up" in requests[1]
    assert lifecycle[:2] == ["agent_end", "auto_retry_start"]


@pytest.mark.tonio
async def test_marks_the_exhausted_retry_run_as_final(harnesses):
    harness = await create_harness(settings={"retry": {"enabled": True, "maxRetries": 1, "baseDelayMs": 1}})
    harnesses.append(harness)
    harness.set_responses(
        [
            faux_assistant_message("", stop_reason="error", error_message="overloaded_error"),
            faux_assistant_message("", stop_reason="error", error_message="overloaded_error"),
        ]
    )

    await harness.session.prompt("start")

    assert [event.will_retry for event in harness.events_of_type("agent_end")] == [True, False]
    assert any(event.success is False and event.attempt == 1 for event in harness.events_of_type("auto_retry_end"))


@pytest.mark.tonio
async def test_keeps_omissions_and_does_not_retry_when_recovery_compaction_fails(harnesses):
    harness = await create_harness(
        models=[{"id": "faux-1", "context_window": 1000, "max_tokens": 100}],
        settings={
            "compaction": {"keepRecentTokens": 1, "reserveTokens": 0},
            "retry": {"enabled": False, "maxRetries": 0, "baseDelayMs": 1},
        },
    )
    harnesses.append(harness)
    harness.set_responses(
        [
            faux_assistant_message("partial response", stop_reason="length"),
            faux_assistant_message("summary failed", stop_reason="error", error_message="summary failed"),
            faux_assistant_message("must not retry"),
        ]
    )

    await harness.session.prompt("x" * 5000)

    entries = harness.session_manager.get_entries()
    assert any(entry["type"] == "context_edit" for entry in entries)
    assert not any(entry["type"] == "compaction" for entry in entries)
    assert any(
        entry["type"] == "message" and get_message_text(entry["message"]) == "partial response" for entry in entries
    )
    assert not any(
        get_message_text(message) == "partial response"
        for message in harness.session_manager.build_session_projection().messages
    )
    assert harness.faux.state.call_count == 2
