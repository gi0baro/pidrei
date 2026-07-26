"""Mirror of pi agent/test/harness/agent-harness.test.ts."""

import time
from dataclasses import dataclass

import pytest
import tonio.colored as tonio

from pidrei_agent.harness.agent_harness import AgentHarness, AgentHarnessOptions
from pidrei_agent.harness.compaction.compaction import CompactionResult
from pidrei_agent.harness.messages import CustomMessage
from pidrei_agent.harness.prompt_templates import PromptTemplate
from pidrei_agent.harness.session.memory_storage import InMemorySessionStorage
from pidrei_agent.harness.session.session import Session
from pidrei_agent.harness.types import (
    AgentHarnessResources,
    BeforeAgentStartResult,
    BranchSummaryOverride,
    HarnessToolResultPatch,
    SessionBeforeCompactResult,
    SessionBeforeTreeResult,
    Skill,
)
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.providers.faux import FauxModelDefinition, faux_assistant_message, faux_provider, faux_tool_call
from pidrei_ai.registry import create_models
from pidrei_ai.types import AssistantMessage, TextContent, Usage, UsageCost, UserMessage
from pidrei_ai.utils.retry import RetryPolicy
from tests.harness_tool_fixtures import (
    calculate_tool,
    create_calculate_tool_with_usage,
    get_current_time_tool,
    make_calculate_tool,
)


@dataclass(slots=True)
class AppSkill(Skill):
    source: str = "project"


@dataclass(slots=True)
class AppPromptTemplate(PromptTemplate):
    source: str = "project"


# Shared collection; each faux provider gets a unique id so coexisting fakes route correctly.
models = create_models()
_faux_count = 0


def new_faux(**options):
    global _faux_count
    _faux_count += 1
    faux = faux_provider(provider=f"faux-{_faux_count}", **options)
    models.set_provider(faux.provider)
    return faux


def text_from_user_messages(messages) -> list[str]:
    texts: list[str] = []
    for message in messages:
        if getattr(message, "role", None) != "user":
            continue
        content = message.content
        if isinstance(content, str):
            texts.append(content)
            continue
        texts.extend(part.text for part in content if getattr(part, "type", None) == "text")
    return texts


def create_usage(input: int, output: int, cache_read: int = 0, cache_write: int = 0) -> Usage:
    return Usage(
        input=input,
        output=output,
        cache_read=cache_read,
        cache_write=cache_write,
        total_tokens=input + output + cache_read + cache_write,
        cost=UsageCost(),
    )


def create_user_message(text: str) -> UserMessage:
    return UserMessage(content=[TextContent(text=text)], timestamp=int(time.time() * 1000))


def create_assistant_message(text: str) -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text=text)],
        api="faux",
        provider="faux",
        model="faux-1",
        usage=create_usage(100, 50),
        stop_reason="stop",
        timestamp=int(time.time() * 1000),
    )


@pytest.mark.tonio
async def test_constructs_directly_and_exposes_queue_modes():
    session = Session(InMemorySessionStorage())
    initial_model = get_builtin_model("anthropic", "claude-sonnet-4-5")
    assert initial_model is not None
    harness = AgentHarness(
        AgentHarnessOptions(
            models=models,
            session=session,
            model=initial_model,
            thinking_level="high",
            system_prompt="You are helpful.",
            steering_mode="all",
            follow_up_mode="all",
        )
    )
    assert harness.get_model() is initial_model
    assert harness.get_thinking_level() == "high"
    assert harness.get_steering_mode() == "all"
    assert harness.get_follow_up_mode() == "all"
    await harness.set_steering_mode("one-at-a-time")
    await harness.set_follow_up_mode("one-at-a-time")
    assert harness.get_steering_mode() == "one-at-a-time"
    assert harness.get_follow_up_mode() == "one-at-a-time"


@pytest.mark.tonio
async def test_drains_one_queued_steering_message_at_a_time_and_emits_queue_updates():
    registration = new_faux()
    user_counts: list[int] = []

    def responder(text):
        def respond(context, _options, _state, _model):
            user_counts.append(len([m for m in context.messages if getattr(m, "role", None) == "user"]))
            return faux_assistant_message(text)

        return respond

    registration.set_responses([responder("first"), responder("second"), responder("third")])
    harness = AgentHarness(
        AgentHarnessOptions(
            models=models,
            session=Session(InMemorySessionStorage()),
            model=registration.get_model(),
            steering_mode="one-at-a-time",
        )
    )
    steer_queue_lengths: list[int] = []
    queued = False

    async def listener(event, _signal):
        nonlocal queued
        if event.type == "queue_update":
            steer_queue_lengths.append(len(event.steer))
        if event.type == "message_start" and getattr(event.message, "role", None) == "assistant" and not queued:
            queued = True
            await harness.steer("one")
            await harness.steer("two")

    harness.subscribe(listener)

    await harness.prompt("hello")

    assert user_counts == [1, 2, 3]
    assert steer_queue_lengths == [1, 2, 1, 0]


@pytest.mark.tonio
async def test_appends_before_agent_start_messages_and_persists_them():
    registration = new_faux()
    request_text: list[str] = []

    def respond(context, _options, _state, _model):
        request_text.extend(text_from_user_messages(context.messages))
        return faux_assistant_message("ok")

    registration.set_responses([respond])
    session = Session(InMemorySessionStorage())
    harness = AgentHarness(AgentHarnessOptions(models=models, session=session, model=registration.get_model()))
    harness.on(
        "before_agent_start",
        lambda _event: BeforeAgentStartResult(messages=[create_user_message("hook")]),
    )

    await harness.prompt("hello")

    persisted_text = []
    for entry in await session.get_entries():
        if entry.type != "message" or getattr(entry.message, "role", None) != "user":
            continue
        persisted_text.extend(text_from_user_messages([entry.message]))
    assert request_text == ["hello", "hook"]
    assert persisted_text == ["hello", "hook"]


@pytest.mark.tonio
async def test_abort_clears_steer_and_follow_up_queues_but_preserves_next_turn_messages():
    registration = new_faux()
    first_response_released = tonio.Event()
    aborted_signal = None
    second_request_text: list[str] = []

    async def first_respond(_context, options, _state, _model):
        nonlocal aborted_signal
        aborted_signal = options.cancel
        await first_response_released.wait(None)
        return faux_assistant_message("aborted-ish")

    def second_respond(context, _options, _state, _model):
        second_request_text.extend(text_from_user_messages(context.messages))
        return faux_assistant_message("second")

    registration.set_responses([first_respond, second_respond])
    harness = AgentHarness(
        AgentHarnessOptions(models=models, session=Session(InMemorySessionStorage()), model=registration.get_model())
    )
    queue_updates: list[tuple[int, int, int]] = []

    def listener(event, _signal):
        if event.type == "queue_update":
            queue_updates.append((len(event.steer), len(event.follow_up), len(event.next_turn)))

    harness.subscribe(listener)

    first_prompt = tonio.spawn(harness.prompt("first"))
    await tonio.sleep(0.02)
    await harness.steer("steer")
    await harness.follow_up("follow")
    await harness.next_turn("next")
    abort_handle = tonio.spawn(harness.abort())
    await tonio.sleep(0.02)
    assert aborted_signal is not None and aborted_signal.cancelled is True
    first_response_released.set()
    abort_result = await abort_handle
    await first_prompt
    await harness.prompt("second")

    assert len(abort_result.cleared_steer) == 1
    assert len(abort_result.cleared_follow_up) == 1
    assert (0, 0, 1) in queue_updates
    assert second_request_text == ["first", "next", "second"]


@pytest.mark.tonio
async def test_drains_follow_up_messages_one_at_a_time_after_the_agent_would_otherwise_stop():
    registration = new_faux()
    user_counts: list[int] = []

    def responder(text):
        def respond(context, _options, _state, _model):
            user_counts.append(len([m for m in context.messages if getattr(m, "role", None) == "user"]))
            return faux_assistant_message(text)

        return respond

    registration.set_responses([responder("first"), responder("second"), responder("third")])
    harness = AgentHarness(
        AgentHarnessOptions(
            models=models,
            session=Session(InMemorySessionStorage()),
            model=registration.get_model(),
            follow_up_mode="one-at-a-time",
        )
    )
    follow_up_queue_lengths: list[int] = []
    queued = False

    async def listener(event, _signal):
        nonlocal queued
        if event.type == "queue_update":
            follow_up_queue_lengths.append(len(event.follow_up))
        if event.type == "message_start" and getattr(event.message, "role", None) == "assistant" and not queued:
            queued = True
            await harness.follow_up("one")
            await harness.follow_up("two")

    harness.subscribe(listener)

    await harness.prompt("hello")

    assert user_counts == [1, 2, 3]
    assert follow_up_queue_lengths == [1, 2, 1, 0]


@pytest.mark.tonio
async def test_settles_thrown_hook_failures_with_persisted_assistant_error_messages():
    registration = new_faux()
    registration.set_responses([faux_assistant_message("should not be used")])
    session = Session(InMemorySessionStorage())
    harness = AgentHarness(AgentHarnessOptions(models=models, session=session, model=registration.get_model()))
    events: list[str] = []
    harness.subscribe(lambda event, _signal: events.append(event.type))

    def context_hook(_event):
        raise Exception("context exploded")

    harness.on("context", context_hook)

    response = await harness.prompt("hello")
    second = await harness.prompt("after failure")
    assert getattr(second, "role", None) == "assistant"

    entries = await session.get_entries()
    messages = [entry.message for entry in entries if entry.type == "message"]
    assert response.stop_reason == "error"
    assert response.error_message == "context exploded"
    assert getattr(messages[0], "role", None) == "user"
    assert getattr(messages[1], "role", None) == "assistant"
    assert messages[1].stop_reason == "error"
    assert messages[1].error_message == "context exploded"
    assert "agent_end" in events
    assert "settled" in events


@pytest.mark.tonio
async def test_refreshes_model_thinking_level_resources_system_prompt_and_active_tools_at_save_points():
    registration = new_faux(
        models=[
            FauxModelDefinition(id="first", reasoning=True),
            FauxModelDefinition(id="second", reasoning=True),
        ]
    )
    second_model = registration.get_model("second")
    assert second_model is not None
    captured: list[tuple[str, object, str, list[str]]] = []

    def capture(context, options, _state, model):
        captured.append(
            (
                model.id,
                options.reasoning if options is not None else None,
                context.system_prompt or "",
                [tool.name for tool in context.tools or []],
            )
        )

    def first_respond(context, options, state, model):
        capture(context, options, state, model)
        return faux_assistant_message(
            faux_tool_call("calculate", {"expression": "1 + 1"}, id="call-1"), stop_reason="toolUse"
        )

    def second_respond(context, options, state, model):
        capture(context, options, state, model)
        return faux_assistant_message("done")

    registration.set_responses([first_respond, second_respond])
    harness = AgentHarness(
        AgentHarnessOptions(
            models=models,
            session=Session(InMemorySessionStorage()),
            model=registration.get_model(),
            thinking_level="off",
            resources=AgentHarnessResources(
                skills=[Skill(name="prompt", description="prompt", content="first prompt", file_path="/skills/prompt")]
            ),
            system_prompt=lambda ctx: (ctx.resources.skills or [Skill("", "", "missing prompt", "")])[0].content,
            tools=[calculate_tool],
        )
    )

    async def listener(event, _signal):
        if event.type == "tool_execution_start":
            await harness.set_model(second_model)
            await harness.set_thinking_level("high")
            await harness.set_resources(
                AgentHarnessResources(
                    skills=[
                        Skill(name="prompt", description="prompt", content="second prompt", file_path="/skills/prompt")
                    ]
                )
            )
            await harness.set_tools([calculate_tool, get_current_time_tool], [get_current_time_tool.name])

    harness.subscribe(listener)

    await harness.prompt("hello")

    assert captured == [
        ("first", None, "first prompt", ["calculate"]),
        ("second", "high", "second prompt", ["get_current_time"]),
    ]


@pytest.mark.tonio
async def test_orders_pending_listener_session_writes_after_agent_emitted_messages():
    registration = new_faux()
    registration.set_responses([faux_assistant_message("ok")])
    session = Session(InMemorySessionStorage())
    harness = AgentHarness(AgentHarnessOptions(models=models, session=session, model=registration.get_model()))
    wrote_pending_message = False

    async def listener(event, _signal):
        nonlocal wrote_pending_message
        if (
            event.type == "message_end"
            and getattr(event.message, "role", None) == "assistant"
            and not wrote_pending_message
        ):
            wrote_pending_message = True
            await harness.append_message(
                CustomMessage(
                    custom_type="listener",
                    content="listener write",
                    display=True,
                    timestamp=int(time.time() * 1000),
                )
            )

    harness.subscribe(listener)

    await harness.prompt("hello")

    entries = await session.get_entries()
    roles = [getattr(entry.message, "role", None) for entry in entries if entry.type == "message"]
    assert roles == ["user", "assistant", "custom"]


@pytest.mark.tonio
async def test_wait_for_idle_waits_for_external_run_settlement_and_awaited_listeners():
    registration = new_faux()
    registration.set_responses([faux_assistant_message("ok")])
    barrier = tonio.Event()
    harness = AgentHarness(
        AgentHarnessOptions(models=models, session=Session(InMemorySessionStorage()), model=registration.get_model())
    )
    listener_finished = False

    async def listener(event, _signal):
        nonlocal listener_finished
        if event.type == "agent_end":
            await barrier.wait(None)
            listener_finished = True

    harness.subscribe(listener)

    prompt_handle = tonio.spawn(harness.prompt("hello"))
    idle_resolved = False

    async def run_idle():
        nonlocal idle_resolved
        await tonio.sleep(0.001)
        await harness.wait_for_idle()
        idle_resolved = True

    idle_handle = tonio.spawn(run_idle())
    await tonio.sleep(0.01)
    assert idle_resolved is False
    assert listener_finished is False
    barrier.set()
    await prompt_handle
    await idle_handle
    assert idle_resolved is True
    assert listener_finished is True


@pytest.mark.tonio
async def test_runs_tool_call_and_tool_result_hooks_through_the_direct_loop():
    registration = new_faux()
    registration.set_responses(
        [
            faux_assistant_message(
                faux_tool_call("calculate", {"expression": "2 + 2"}, id="call-1"), stop_reason="toolUse"
            )
        ]
    )
    session = Session(InMemorySessionStorage())
    tool_usage = create_usage(1, 2, 3, 4)
    patched_tool_usage = create_usage(5, 6, 7, 8)
    calculate_tool_with_usage = create_calculate_tool_with_usage(tool_usage)
    harness = AgentHarness(
        AgentHarnessOptions(
            models=models, session=session, model=registration.get_model(), tools=[calculate_tool_with_usage]
        )
    )
    seen_tool_calls: list[tuple[str, str, object]] = []
    seen_tool_usage = None

    def on_tool_call(event):
        seen_tool_calls.append((event.tool_call_id, event.tool_name, event.input["expression"]))

    def on_tool_result(event):
        nonlocal seen_tool_usage
        assert event.tool_call_id == "call-1"
        assert event.tool_name == "calculate"
        seen_tool_usage = event.usage
        return HarnessToolResultPatch(
            content=[TextContent(text="patched result")],
            details={"patched": True},
            usage=patched_tool_usage,
            terminate=True,
        )

    harness.on("tool_call", on_tool_call)
    harness.on("tool_result", on_tool_result)

    await harness.prompt("hello")

    tool_result = next(
        (
            entry
            for entry in await session.get_entries()
            if entry.type == "message" and getattr(entry.message, "role", None) == "toolResult"
        ),
        None,
    )
    assert seen_tool_calls == [("call-1", "calculate", "2 + 2")]
    assert seen_tool_usage == tool_usage
    assert tool_result is not None
    assert tool_result.message.content == [TextContent(text="patched result")]
    assert tool_result.message.details == {"patched": True}
    assert tool_result.message.usage == patched_tool_usage


@pytest.mark.tonio
async def test_passes_a_static_application_context_to_harness_tools():
    registration = new_faux()
    registration.set_responses(
        [faux_assistant_message(faux_tool_call("context", {"expression": "2 + 2"}, id="call-1"), stop_reason="toolUse")]
    )
    tool_context = {"marker": object()}
    received_context = None

    async def execute(tool_call_id, params, cancel, on_update, context):
        nonlocal received_context
        received_context = context
        result = await calculate_tool.execute(tool_call_id, params, cancel, on_update, context)
        result.terminate = True
        return result

    context_tool = make_calculate_tool().clone(name="context", execute=execute)
    harness = AgentHarness(
        AgentHarnessOptions(
            models=models,
            session=Session(InMemorySessionStorage()),
            model=registration.get_model(),
            tools=[context_tool],
            tool_context=tool_context,
        )
    )

    await harness.prompt("hello")

    assert received_context is tool_context


@pytest.mark.tonio
async def test_resolves_async_tool_context_providers_for_each_turn_snapshot():
    registration = new_faux()
    registration.set_responses(
        [
            faux_assistant_message(
                faux_tool_call("context", {"expression": "1 + 1"}, id="call-1"), stop_reason="toolUse"
            ),
            faux_assistant_message(
                faux_tool_call("context", {"expression": "2 + 2"}, id="call-2"), stop_reason="toolUse"
            ),
            faux_assistant_message("done"),
        ]
    )
    generations: list[int] = []

    async def execute(tool_call_id, params, cancel, on_update, context):
        generations.append(context["generation"])
        return await calculate_tool.execute(tool_call_id, params, cancel, on_update, context)

    context_tool = make_calculate_tool().clone(name="context", execute=execute)
    generation = 0

    async def tool_context():
        nonlocal generation
        generation += 1
        return {"generation": generation}

    harness = AgentHarness(
        AgentHarnessOptions(
            models=models,
            session=Session(InMemorySessionStorage()),
            model=registration.get_model(),
            tools=[context_tool],
            tool_context=tool_context,
        )
    )

    await harness.prompt("hello")

    assert generations == [1, 2]


@pytest.mark.tonio
async def test_persists_generated_compaction_usage():
    registration = new_faux()
    registration.set_responses([faux_assistant_message("## Goal\nTest summary")])
    session = Session(InMemorySessionStorage())
    await session.append_message(create_user_message("one"))
    await session.append_message(create_assistant_message("two"))
    harness = AgentHarness(AgentHarnessOptions(models=models, session=session, model=registration.get_model()))

    result = await harness.compact()
    compaction = next((entry for entry in await session.get_entries() if entry.type == "compaction"), None)

    assert result.usage is not None and result.usage.total_tokens > 0
    assert compaction is not None
    assert compaction.usage == result.usage


@pytest.mark.tonio
async def test_persists_hook_provided_compaction_usage():
    registration = new_faux()
    usage = create_usage(5, 6, 7, 8)
    session = Session(InMemorySessionStorage())
    await session.append_message(create_user_message("one"))
    await session.append_message(create_assistant_message("two"))
    harness = AgentHarness(AgentHarnessOptions(models=models, session=session, model=registration.get_model()))
    harness.on(
        "session_before_compact",
        lambda event: SessionBeforeCompactResult(
            compaction=CompactionResult(
                summary="hook summary",
                first_kept_entry_id=event.preparation.first_kept_entry_id,
                tokens_before=event.preparation.tokens_before,
                usage=usage,
            )
        ),
    )

    result = await harness.compact()
    compaction = next((entry for entry in await session.get_entries() if entry.type == "compaction"), None)

    assert result.usage == usage
    assert compaction is not None
    assert compaction.usage == usage


@pytest.mark.tonio
async def test_retries_transient_compaction_errors_and_emits_retry_events():
    registration = new_faux()
    calls = 0

    def failing(_context, _options, _state, _model):
        nonlocal calls
        calls += 1
        return faux_assistant_message("", stop_reason="error", error_message="terminated")

    def recovering(_context, _options, _state, _model):
        nonlocal calls
        calls += 1
        return faux_assistant_message("## Goal\nRecovered summary")

    registration.set_responses([failing, recovering])
    session = Session(InMemorySessionStorage())
    await session.append_message(create_user_message("one"))
    await session.append_message(create_assistant_message("two"))
    harness = AgentHarness(
        AgentHarnessOptions(
            models=models,
            session=session,
            model=registration.get_model(),
            retry=RetryPolicy(enabled=True, max_retries=1, base_delay_ms=0),
        )
    )
    retry_events: list[str] = []

    def listener(event, _signal):
        if event.type in ("retry_scheduled", "retry_attempt_start", "retry_finished"):
            retry_events.append(f"{event.type}:{event.operation}")

    harness.subscribe(listener)

    result = await harness.compact()

    assert "Recovered summary" in result.summary
    assert calls == 2
    assert retry_events == [
        "retry_scheduled:compaction",
        "retry_attempt_start:compaction",
        "retry_finished:compaction",
    ]


@pytest.mark.tonio
async def test_does_not_retry_non_retryable_compaction_errors():
    registration = new_faux()
    calls = 0

    def failing(_context, _options, _state, _model):
        nonlocal calls
        calls += 1
        return faux_assistant_message("", stop_reason="error", error_message="insufficient_quota")

    registration.set_responses([failing])
    session = Session(InMemorySessionStorage())
    await session.append_message(create_user_message("one"))
    await session.append_message(create_assistant_message("two"))
    harness = AgentHarness(
        AgentHarnessOptions(
            models=models,
            session=session,
            model=registration.get_model(),
            retry=RetryPolicy(enabled=True, max_retries=1, base_delay_ms=0),
        )
    )
    retry_events: list[str] = []

    def listener(event, _signal):
        if event.type in ("retry_scheduled", "retry_attempt_start", "retry_finished"):
            retry_events.append(event.type)

    harness.subscribe(listener)

    with pytest.raises(Exception, match="insufficient_quota"):
        await harness.compact()

    assert calls == 1
    assert retry_events == []


@pytest.mark.tonio
async def test_exhausts_transient_compaction_retries_after_max_retries_failures():
    registration = new_faux()
    calls = 0

    def failing(_context, _options, _state, _model):
        nonlocal calls
        calls += 1
        return faux_assistant_message("", stop_reason="error", error_message="terminated")

    registration.set_responses([failing, failing, failing, failing])
    session = Session(InMemorySessionStorage())
    await session.append_message(create_user_message("one"))
    await session.append_message(create_assistant_message("two"))
    harness = AgentHarness(
        AgentHarnessOptions(
            models=models,
            session=session,
            model=registration.get_model(),
            retry=RetryPolicy(enabled=True, max_retries=3, base_delay_ms=0),
        )
    )
    retry_events: list[str] = []

    def listener(event, _signal):
        if event.type in ("retry_scheduled", "retry_attempt_start", "retry_finished"):
            retry_events.append(f"{event.type}:{event.operation}")

    harness.subscribe(listener)

    with pytest.raises(Exception, match="terminated"):
        await harness.compact()

    assert calls == 4
    assert retry_events == [
        "retry_scheduled:compaction",
        "retry_attempt_start:compaction",
        "retry_scheduled:compaction",
        "retry_attempt_start:compaction",
        "retry_scheduled:compaction",
        "retry_attempt_start:compaction",
        "retry_finished:compaction",
    ]


@pytest.mark.tonio
async def test_retries_transient_branch_summary_errors_and_emits_retry_events():
    registration = new_faux()
    calls = 0

    def failing(_context, _options, _state, _model):
        nonlocal calls
        calls += 1
        return faux_assistant_message("", stop_reason="error", error_message="terminated")

    def recovering(_context, _options, _state, _model):
        nonlocal calls
        calls += 1
        return faux_assistant_message("## Goal\nRecovered branch summary")

    registration.set_responses([failing, recovering])
    session = Session(InMemorySessionStorage())
    target_id = await session.append_message(create_user_message("first branch"))
    await session.append_message(create_assistant_message("first reply"))
    await session.append_message(create_user_message("abandoned work"))
    await session.append_message(create_assistant_message("abandoned reply"))
    harness = AgentHarness(
        AgentHarnessOptions(
            models=models,
            session=session,
            model=registration.get_model(),
            retry=RetryPolicy(enabled=True, max_retries=1, base_delay_ms=0),
        )
    )
    retry_events: list[str] = []

    def listener(event, _signal):
        if event.type in ("retry_scheduled", "retry_attempt_start", "retry_finished"):
            retry_events.append(f"{event.type}:{event.operation}")

    harness.subscribe(listener)

    result = await harness.navigate_tree(target_id, summarize=True)

    assert result.summary_entry is not None
    assert "Recovered branch summary" in result.summary_entry.summary
    assert calls == 2
    assert retry_events == [
        "retry_scheduled:branch_summary",
        "retry_attempt_start:branch_summary",
        "retry_finished:branch_summary",
    ]


@pytest.mark.tonio
async def test_persists_generated_branch_summary_usage():
    registration = new_faux()
    registration.set_responses([faux_assistant_message("## Goal\nBranch summary")])
    session = Session(InMemorySessionStorage())
    target_id = await session.append_message(create_user_message("first branch"))
    await session.append_message(create_assistant_message("first reply"))
    await session.append_message(create_user_message("abandoned work"))
    await session.append_message(create_assistant_message("abandoned reply"))
    harness = AgentHarness(AgentHarnessOptions(models=models, session=session, model=registration.get_model()))

    result = await harness.navigate_tree(target_id, summarize=True)

    assert result.summary_entry is not None
    assert result.summary_entry.usage is not None
    assert result.summary_entry.usage.total_tokens > 0


@pytest.mark.tonio
async def test_persists_hook_provided_branch_summary_usage():
    registration = new_faux()
    usage = create_usage(13, 14, 15, 16)
    session = Session(InMemorySessionStorage())
    target_id = await session.append_message(create_user_message("first branch"))
    await session.append_message(create_assistant_message("first reply"))
    await session.append_message(create_user_message("abandoned work"))
    await session.append_message(create_assistant_message("abandoned reply"))
    harness = AgentHarness(AgentHarnessOptions(models=models, session=session, model=registration.get_model()))
    harness.on(
        "session_before_tree",
        lambda _event: SessionBeforeTreeResult(
            summary=BranchSummaryOverride(summary="hook branch summary", usage=usage)
        ),
    )

    result = await harness.navigate_tree(target_id, summarize=True)

    assert result.summary_entry is not None
    assert result.summary_entry.usage == usage


@pytest.mark.tonio
async def test_preserves_app_tool_types_for_getters_and_update_events():
    session = Session(InMemorySessionStorage())
    model = get_builtin_model("anthropic", "claude-sonnet-4-5")
    assert model is not None
    inspect_tool = make_calculate_tool().clone(name="inspect", source="builtin")
    search_tool = make_calculate_tool().clone(name="search", source="extension")
    harness = AgentHarness(
        AgentHarnessOptions(
            models=models,
            session=session,
            model=model,
            tools=[inspect_tool, search_tool],
            active_tool_names=["inspect"],
        )
    )
    updates: list[dict] = []

    def listener(event, _signal):
        if event.type == "tools_update":
            updates.append(
                {
                    "tool_names": event.tool_names,
                    "previous_tool_names": event.previous_tool_names,
                    "active_tool_names": event.active_tool_names,
                    "previous_active_tool_names": event.previous_active_tool_names,
                    "source": event.source,
                }
            )
            assert [tool.name for tool in harness.get_active_tools()] == event.active_tool_names

    harness.subscribe(listener)

    tools = harness.get_tools()
    active_tools = harness.get_active_tools()
    tools.pop()
    active_tools.pop()
    assert [tool.name for tool in harness.get_tools()] == ["inspect", "search"]
    assert [tool.source for tool in harness.get_active_tools()] == ["builtin"]

    await harness.set_active_tools(["search"])
    await harness.set_tools([search_tool], ["search"])
    with pytest.raises(Exception) as excinfo:
        await harness.set_active_tools(["missing"])
    assert excinfo.value.code == "invalid_argument"
    with pytest.raises(Exception) as excinfo:
        await harness.set_active_tools(["search", "search"])
    assert excinfo.value.code == "invalid_argument"
    with pytest.raises(Exception) as excinfo:
        await harness.set_tools([inspect_tool])
    assert excinfo.value.code == "invalid_argument"
    with pytest.raises(Exception) as excinfo:
        await harness.set_tools([inspect_tool, inspect_tool], ["inspect"])
    assert excinfo.value.code == "invalid_argument"

    assert updates == [
        {
            "tool_names": ["inspect", "search"],
            "previous_tool_names": ["inspect", "search"],
            "active_tool_names": ["search"],
            "previous_active_tool_names": ["inspect"],
            "source": "set",
        },
        {
            "tool_names": ["search"],
            "previous_tool_names": ["inspect", "search"],
            "active_tool_names": ["search"],
            "previous_active_tool_names": ["search"],
            "source": "set",
        },
    ]
    assert [tool.source for tool in harness.get_tools()] == ["extension"]
    assert [tool.name for tool in harness.get_active_tools()] == ["search"]
    assert (await session.build_context()).active_tool_names == ["search"]


@pytest.mark.tonio
async def test_validates_constructor_tool_names():
    session = Session(InMemorySessionStorage())
    model = get_builtin_model("anthropic", "claude-sonnet-4-5")
    assert model is not None
    with pytest.raises(Exception, match="Unknown tool"):
        AgentHarness(
            AgentHarnessOptions(
                session=session, models=models, model=model, tools=[calculate_tool], active_tool_names=["missing"]
            )
        )
    with pytest.raises(Exception, match="Duplicate tool"):
        AgentHarness(
            AgentHarnessOptions(
                models=models,
                session=session,
                model=model,
                tools=[calculate_tool, calculate_tool],
                active_tool_names=[calculate_tool.name],
            )
        )
    with pytest.raises(Exception, match="Duplicate active tool"):
        AgentHarness(
            AgentHarnessOptions(
                models=models,
                session=session,
                model=model,
                tools=[calculate_tool],
                active_tool_names=[calculate_tool.name, calculate_tool.name],
            )
        )


@pytest.mark.tonio
async def test_preserves_app_resource_types_for_getters_and_update_events():
    session = Session(InMemorySessionStorage())
    model = get_builtin_model("anthropic", "claude-sonnet-4-5")
    assert model is not None
    harness = AgentHarness(AgentHarnessOptions(session=session, models=models, model=model))
    skill = AppSkill(
        name="inspect",
        description="Inspect things",
        content="Use inspection tools.",
        file_path="/skills/inspect/SKILL.md",
        source="project",
    )
    prompt_template = AppPromptTemplate(name="review", content="Review $1", source="user")
    resources = AgentHarnessResources(skills=[skill], prompt_templates=[prompt_template])
    updates: list[tuple[object, object]] = []

    def listener(event, _signal):
        if event.type == "resources_update":
            updates.append(
                (
                    event.resources.skills[0].source if event.resources.skills else None,
                    event.previous_resources.skills[0].source if event.previous_resources.skills else None,
                )
            )

    harness.subscribe(listener)

    await harness.set_resources(resources)
    await harness.set_resources(resources)
    resolved = harness.get_resources()

    assert updates == [("project", None), ("project", "project")]
    assert resolved.skills is not None and resolved.skills[0].source == "project"
    assert resolved.prompt_templates is not None and resolved.prompt_templates[0].source == "user"
    assert resolved.skills is not resources.skills
    assert resolved.prompt_templates is not resources.prompt_templates
