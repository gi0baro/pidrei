"""AgentHarness v2 scaffold (mirror of pi agent/test/harness/agent-harness-scaffold.test.ts)."""

from types import SimpleNamespace

import pytest

from pidrei_agent.harness.agent_harness import (
    AgentHarness,
    AgentHarnessOptions,
    HarnessClosed,
    HarnessNotImplemented,
)
from pidrei_agent.harness.compaction.compaction import CompactionSettings
from pidrei_agent.harness.prompt_templates import PromptTemplate
from pidrei_agent.harness.session.memory import InMemorySessionStorage
from pidrei_agent.harness.session.session import Session
from pidrei_agent.harness.session.types import OperationStartedRecord, RunIntent, SessionMetadata
from pidrei_agent.harness.types import AgentHarnessResources, AgentHarnessStreamOptions, Skill
from pidrei_ai.registry import create_models
from pidrei_ai.types import TextContent, Usage, UserMessage
from pidrei_ai.utils.retry import RetryPolicy


MODELS = create_models()


def create_session(id: str = "session") -> Session:
    return Session(InMemorySessionStorage(SessionMetadata(id=id, created_at=1)))


def _options(session: Session) -> AgentHarnessOptions:
    return AgentHarnessOptions(session=session, models=MODELS, model=MODELS.get_model("google", "gemini-2.5-flash"))


async def create_harness(session: Session | None = None) -> AgentHarness:
    harness, _ = await AgentHarness.create(_options(session if session is not None else create_session()))
    return harness


def operation_started(id: str) -> OperationStartedRecord:
    return OperationStartedRecord(
        id=id, lane="main", source_leaf_id=None, intent=RunIntent(original_prompt=[], initial_messages=[])
    )


USER_MESSAGE = UserMessage(content=[TextContent(text="hello")], timestamp=1)
USAGE = Usage(input=1, output=2, total_tokens=3)


@pytest.mark.tonio
async def test_opens_only_record_free_sessions_before_restore_is_implemented():
    session = create_session()
    harness, suspended = await AgentHarness.create(_options(session))

    assert suspended == []
    assert harness.name == "main"
    assert harness.session is session
    assert await harness.get_leaf_id() is None
    assert await harness.session.get_leaf_id() is None

    assert await harness.close() is None

    recorded = create_session("recorded")
    await recorded.append_record(operation_started("run"))
    with pytest.raises(HarnessNotImplemented) as excinfo:
        await AgentHarness.create(_options(recorded))
    assert excinfo.value.operation == "create.restore"


@pytest.mark.tonio
async def test_keeps_scaffold_safe_configuration_as_defensive_copies():
    harness = await create_harness()
    model = MODELS.get_model("anthropic", "claude-sonnet-4-5")
    await harness.set_model(model)
    assert await harness.get_model() is model

    await harness.set_thinking_level("high")
    assert await harness.get_thinking_level() == "high"

    active_tools = ["one"]
    await harness.set_active_tools(active_tools)
    active_tools.append("mutated")
    assert await harness.get_active_tools() == ["one"]
    read_active_tools = await harness.get_active_tools()
    read_active_tools.append("mutated")
    assert await harness.get_active_tools() == ["one"]

    tool = SimpleNamespace(name="tool", label="Tool")
    tools = [tool]
    await harness.set_tools(tools)
    tools.append(SimpleNamespace(name="mutated", label="Mutated"))
    assert [item.name for item in await harness.get_tools()] == ["tool"]
    read_tools = await harness.get_tools()
    read_tools.append(SimpleNamespace(name="mutated", label="Mutated"))
    assert [item.name for item in await harness.get_tools()] == ["tool"]

    resources = AgentHarnessResources(
        skills=[Skill(name="skill", description="desc", content="body", file_path="/tmp/SKILL.md")],
        prompt_templates=[PromptTemplate(name="template", content="body")],
    )
    await harness.set_resources(resources)
    resources.skills.append(Skill(name="mutated", description="desc", content="body", file_path="/tmp/OTHER.md"))
    assert [skill.name for skill in (await harness.get_resources()).skills] == ["skill"]
    read_resources = await harness.get_resources()
    read_resources.skills.append(Skill(name="mutated", description="desc", content="body", file_path="/tmp/OTHER.md"))
    assert [skill.name for skill in (await harness.get_resources()).skills] == ["skill"]

    stream_options = AgentHarnessStreamOptions(max_retries=10)
    await harness.set_stream_options(stream_options)
    stream_options.max_retries = 20
    assert (await harness.get_stream_options()).max_retries == 10
    read_stream_options = await harness.get_stream_options()
    read_stream_options.max_retries = 30
    assert (await harness.get_stream_options()).max_retries == 10

    retry_policy = RetryPolicy(enabled=True, max_retries=2, base_delay_ms=10)
    await harness.set_retry_policy(retry_policy)
    retry_policy.max_retries = 99
    assert await harness.get_retry_policy() == RetryPolicy(enabled=True, max_retries=2, base_delay_ms=10)

    compaction_settings = CompactionSettings(enabled=False, reserve_tokens=1, keep_recent_tokens=2)
    await harness.set_compaction_settings(compaction_settings)
    compaction_settings.reserve_tokens = 99
    assert await harness.get_compaction_settings() == CompactionSettings(
        enabled=False, reserve_tokens=1, keep_recent_tokens=2
    )

    await harness.set_steering_mode("all")
    assert await harness.get_steering_mode() == "all"
    await harness.set_follow_up_mode("all")
    assert await harness.get_follow_up_mode() == "all"


@pytest.mark.tonio
async def test_rejects_every_unfinished_public_operation_explicitly():
    harness = await create_harness()
    callback_called = False

    def callback() -> None:
        nonlocal callback_called
        callback_called = True

    unfinished = [
        ("prompt", lambda: harness.prompt("hello")),
        ("skill", lambda: harness.skill("skill")),
        ("prompt_from_template", lambda: harness.prompt_from_template("template")),
        ("compact", lambda: harness.compact()),
        ("navigate_tree", lambda: harness.navigate_tree(None)),
        ("resume", lambda: harness.resume()),
        ("abort", lambda: harness.abort()),
        ("steer", lambda: harness.steer(USER_MESSAGE)),
        ("follow_up", lambda: harness.follow_up(USER_MESSAGE)),
        ("next_run", lambda: harness.next_run(USER_MESSAGE)),
        ("cancel_queued", lambda: harness.cancel_queued("queued")),
        ("record_usage", lambda: harness.record_usage(USAGE)),
        ("wait_for_idle", lambda: harness.wait_for_idle()),
        ("run_when_idle", lambda: harness.run_when_idle(callback)),
        ("peek_action", lambda: harness.peek_action()),
        ("execute_action", lambda: harness.execute_action()),
        ("run_to_completion", lambda: harness.run_to_completion()),
        ("watch", lambda: harness.watch()),
        ("lane", lambda: harness.lane("main")),
        ("create_lane", lambda: harness.create_lane("thread", None)),
        ("lanes", lambda: harness.lanes()),
        ("watch_session", lambda: harness.watch_session()),
    ]

    for operation, invoke in unfinished:
        with pytest.raises(HarnessNotImplemented) as excinfo:
            await invoke()
        assert excinfo.value.operation == operation, operation
    assert callback_called is False
    with pytest.raises(HarnessNotImplemented):
        harness.hooks.on("before_run", lambda event: None)
    with pytest.raises(HarnessNotImplemented):
        harness.events.on("event", lambda event: None)


@pytest.mark.tonio
async def test_reports_harness_closed_for_unfinished_operations_after_close():
    harness = await create_harness()
    await harness.close()

    with pytest.raises(HarnessClosed):
        await harness.prompt("hello")
    with pytest.raises(HarnessClosed):
        await harness.wait_for_idle()
    with pytest.raises(HarnessClosed):
        harness.hooks.on("before_run", lambda event: None)
    with pytest.raises(HarnessClosed):
        harness.events.on("event", lambda event: None)
