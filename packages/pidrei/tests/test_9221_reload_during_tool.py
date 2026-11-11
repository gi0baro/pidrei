"""Mirror of pi's suite/regressions/9221-reload-during-tool.test.ts.

JS deferreds become `tonio.Event`s, the pending prompt/compaction/navigation a
spawned coroutine awaited at the end, and `vi.fn` spies plain recording lists.
"""

import pytest
import tonio.colored as tonio

from pidrei.core.agent_session import ExtensionBindings, PromptOptions
from pidrei.core.compaction import CompactionResult
from pidrei.core.extensions import ToolDefinition
from pidrei_agent.types import AgentToolResult
from pidrei_ai.providers.faux import faux_assistant_message, faux_tool_call
from pidrei_ai.types import TextContent

from .coding_session_helpers import assistant_msg, user_msg
from .harness import create_harness, get_message_text


_RUNNING = "Wait for the current response to finish before reloading."
_COMPACTING = "Wait for compaction to finish before reloading."


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


async def _cancelled(*_args):
    return {"cancelled": True}


# Regression for https://github.com/earendil-works/pi/issues/9221.
@pytest.mark.tonio
@pytest.mark.parametrize("source", ["command", "direct"])
async def test_rejects_reload_without_invalidating_an_active_tools_context(harnesses, source):
    started = tonio.Event()
    released = tonio.Event()
    shutdown: list = []
    errors: list[str] = []
    provider_results: list = []

    async def factory(pi) -> None:
        async def on_shutdown(event, _ctx):
            shutdown.append(event)

        async def reload_runtime(_args, ctx):
            await ctx.reload()

        async def execute(_tool_call_id, _params, _cancel=None, _on_update=None, ctx=None):
            started.set()
            await released.wait()
            # #9221: the tool itself can still use ctx after an async operation.
            ctx.get_system_prompt()
            return AgentToolResult(content=[TextContent(text="tool succeeded")], details={})

        pi.on("session_shutdown", on_shutdown)
        pi.register_command("reload-runtime", description="Reload extensions", handler=reload_runtime)
        pi.register_tool(
            ToolDefinition(
                name="wait_then_succeed",
                label="Wait then succeed",
                description="Use the tool context after waiting",
                parameters={"type": "object", "properties": {}},
                execute=execute,
            )
        )

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)
    session = harness.session
    await session.bind_extensions(
        ExtensionBindings(
            mode="rpc",
            on_error=lambda error: errors.append(error.error),
            command_context_actions={
                "wait_for_idle": session.wait_for_idle,
                "new_session": _cancelled,
                "fork": _cancelled,
                "navigate_tree": session.navigate_tree,
                "switch_session": _cancelled,
                # Match RPC's ctx.reload binding to the core reload entry point.
                "reload": session.reload,
            },
        )
    )
    runner = session.extension_runner

    async def respond(context, *_rest):
        provider_results.extend(message for message in context.messages if message.role == "toolResult")
        return faux_assistant_message("done")

    harness.set_responses(
        [faux_assistant_message(faux_tool_call("wait_then_succeed", {}), stop_reason="toolUse"), respond]
    )

    prompt = tonio.spawn(session.prompt("run the tool"))
    await started.wait(5)
    assert started.is_set(), "the tool never started"
    try:
        if source == "command":
            await session.prompt("/reload-runtime", PromptOptions(source="rpc"))
        else:
            with pytest.raises(Exception, match=_RUNNING):
                await session.reload()
    finally:
        released.set()
    await prompt

    persisted_results = [
        entry["message"]
        for entry in harness.session_manager.get_entries()
        if entry["type"] == "message" and entry["message"].role == "toolResult"
    ]
    assert len(persisted_results) == 1
    assert persisted_results[0].is_error is False
    assert get_message_text(persisted_results[0]) == "tool succeeded"
    assert provider_results == persisted_results
    assert errors == ([_RUNNING] if source == "command" else [])
    assert shutdown == []
    assert session.extension_runner is runner
    assert "wait_then_succeed" in runner.get_active_tools()


@pytest.mark.tonio
@pytest.mark.parametrize("operation", ["compaction", "tree navigation"])
async def test_rejects_reload_during(harnesses, operation):
    started = tonio.Event()
    released = tonio.Event()
    shutdown: list = []

    async def factory(pi) -> None:
        async def on_shutdown(event, _ctx):
            shutdown.append(event)

        async def on_before_compact(event, _ctx):
            started.set()
            await released.wait()
            preparation = event["preparation"]
            return {
                "compaction": CompactionResult(
                    summary="summary",
                    first_kept_entry_id=preparation.first_kept_entry_id,
                    tokens_before=preparation.tokens_before,
                    details={},
                )
            }

        async def on_before_tree(_event, _ctx):
            started.set()
            await released.wait()

        pi.on("session_shutdown", on_shutdown)
        pi.on("session_before_compact", on_before_compact)
        pi.on("session_before_tree", on_before_tree)

    harness = await create_harness(settings={"compaction": {"keepRecentTokens": 1}}, extension_factories=[factory])
    harnesses.append(harness)
    session = harness.session
    await harness.session_manager.append_message(user_msg("first user"))
    target_id = await harness.session_manager.append_message(assistant_msg("first assistant"))
    await harness.session_manager.append_message(user_msg("second user"))
    await harness.session_manager.append_message(assistant_msg("second assistant"))
    session.agent.state.messages = harness.session_manager.build_session_context().messages
    runner = session.extension_runner

    pending = tonio.spawn(session.compact() if operation == "compaction" else session.navigate_tree(target_id))
    await started.wait(5)
    assert started.is_set(), f"the {operation} never started"
    try:
        assert session.is_streaming is False
        with pytest.raises(Exception, match=_COMPACTING):
            await session.reload()
        assert shutdown == []
        assert session.extension_runner is runner
    finally:
        released.set()
    await pending
    assert session.is_idle is True


@pytest.mark.tonio
async def test_still_reloads_and_replaces_the_runner_when_idle(harnesses, monkeypatch):
    shutdown: list = []
    start: list = []

    async def factory(pi) -> None:
        async def on_shutdown(event, _ctx):
            shutdown.append(event)

        async def on_start(event, ctx):
            start.append((event, ctx))

        pi.on("session_shutdown", on_shutdown)
        pi.on("session_start", on_start)

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)
    session = harness.session
    await session.bind_extensions(ExtensionBindings(on_error=lambda _error: None))
    start.clear()
    runner = session.extension_runner
    reloads: list = []
    resource_reload = session.resource_loader.reload

    def counting_reload(**kwargs):
        reloads.append(kwargs)
        return resource_reload(**kwargs)

    monkeypatch.setattr(session.resource_loader, "reload", counting_reload)

    await session.reload()

    assert len(shutdown) == 1
    assert len(reloads) == 1
    assert any(event.get("reason") == "reload" and ctx is not None for event, ctx in start)
    assert session.extension_runner is not runner
    with pytest.raises(Exception, match="stale"):
        runner.get_active_tools()
