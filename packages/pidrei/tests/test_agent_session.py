"""Mirrors pi coding-agent test/agent-session-concurrent.test.ts,
agent-session-retry.test.ts, agent-session-stats.test.ts, and
agent-session-auto-compaction-queue.test.ts.

Substitutions (documented): pi's "extension-origin steering" concurrent test
needs the extension loader (`pi.sendUserMessage`) — Phase 5. The concurrent
suite's mock-runner tests patch the private `_extension_runner` attribute the
same way pi does.
"""

from typing import ClassVar

import pytest
import tonio.colored as tonio

from pidrei.core.agent_session import AgentSession, AgentSessionConfig
from pidrei.core.auth_storage import AuthStorage
from pidrei.core.session_manager import SessionManager
from pidrei.core.settings_manager import SettingsManager
from pidrei.core.usage_totals import get_usage_cost_breakdown
from pidrei_agent.agent import Agent, AgentInitialState
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import (
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    StartEvent,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UsageCost,
    UserMessage,
)
from pidrei_ai.utils.event_stream import AssistantMessageEventStream

from .agent_session_helpers import (
    abortable_stream_fn,
    create_agent_session,
    create_assistant_message,
    create_test_resource_loader,
)
from .coding_session_helpers import now_ms


class _MockRunner:
    """Minimal runner stub matching what AgentSession touches (pi's inline mock)."""

    def __init__(self, *, has_handlers=None, emit_tool_call=None, emit_message_end=None):
        self._has_handlers = has_handlers if has_handlers is not None else (lambda _event_type: False)
        self._emit_tool_call = emit_tool_call
        self._emit_message_end = emit_message_end

    def has_handlers(self, event_type):
        return self._has_handlers(event_type)

    async def emit(self, _event):
        return None

    async def emit_message_end(self, event):
        if self._emit_message_end is not None:
            return await self._emit_message_end(event)
        return None

    async def emit_tool_call(self, event):
        if self._emit_tool_call is not None:
            return await self._emit_tool_call(event)
        return None

    async def emit_tool_result(self, _event):
        return None

    async def emit_input(self, _text, _images, _source, _streaming_behavior=None):
        from pidrei.core.extensions.runner import InputEventResult

        return InputEventResult(action="continue")

    async def emit_before_agent_start(self, _prompt, _images, _system_prompt, _options):
        return None

    def invalidate(self, _message=None):
        pass

    def get_command(self, _name):
        return None


class TestConcurrentPromptGuard:
    @pytest.mark.tonio
    async def test_throws_when_prompt_called_while_streaming(self, tmp_dir):
        session = await create_agent_session(tmp_dir, stream_fn=abortable_stream_fn)

        first_prompt = tonio.spawn(session.prompt("First message"))
        await tonio.time.sleep(0.01)

        assert session.is_streaming is True

        with pytest.raises(Exception, match="Agent is already processing"):
            await session.prompt("Second message")

        await session.abort()
        try:
            await first_prompt
        except Exception:
            pass
        session.dispose()

    @pytest.mark.tonio
    async def test_allows_steer_while_streaming(self, tmp_dir):
        session = await create_agent_session(tmp_dir, stream_fn=abortable_stream_fn)

        first_prompt = tonio.spawn(session.prompt("First message"))
        await tonio.time.sleep(0.01)

        await session.steer("Steering message")
        assert session.pending_message_count == 1

        await session.abort()
        try:
            await first_prompt
        except Exception:
            pass
        session.dispose()

    @pytest.mark.tonio
    async def test_allows_follow_up_while_streaming(self, tmp_dir):
        session = await create_agent_session(tmp_dir, stream_fn=abortable_stream_fn)

        first_prompt = tonio.spawn(session.prompt("First message"))
        await tonio.time.sleep(0.01)

        await session.follow_up("Follow-up message")
        assert session.pending_message_count == 1

        await session.abort()
        try:
            await first_prompt
        except Exception:
            pass
        session.dispose()

    @pytest.mark.tonio
    async def test_allows_prompt_after_previous_completes(self, tmp_dir):
        def stream_fn(_model, _context, _options=None):
            stream = AssistantMessageEventStream()
            stream.push(StartEvent(partial=create_assistant_message("")))
            stream.push(DoneEvent(reason="stop", message=create_assistant_message("Done")))
            return stream

        session = await create_agent_session(tmp_dir, stream_fn=stream_fn)

        await session.prompt("First message")
        assert session.is_streaming is False
        await session.prompt("Second message")
        session.dispose()

    @pytest.mark.tonio
    async def test_waits_for_queued_agent_events_before_emitting_tool_call(self, tmp_dir):
        from pidrei_agent.types import AgentTool, AgentToolResult

        class DummyTool(AgentTool):
            name = "dummy"
            label = "dummy"
            description = "Dummy tool"
            parameters: ClassVar = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}

            async def execute(self, _tool_call_id, params, cancel=None, on_update=None, ctx=None):
                return AgentToolResult(content=[TextContent(text=f"result:{params.get('q', '')}")], details={})

        tool = DummyTool()

        def stream_fn(_model, context, _options=None):
            stream = AssistantMessageEventStream()
            tool_result_count = sum(1 for m in context.messages if getattr(m, "role", None) == "toolResult")
            if tool_result_count > 0:
                message = create_assistant_message("done", usage=_usage(2))
                stream.push(StartEvent(partial=create_assistant_message("")))
                stream.push(DoneEvent(reason="stop", message=message))
                return stream

            message = create_assistant_message(
                "",
                content=[
                    ToolCall(id="toolu_1", name="dummy", arguments={"q": "x"}),
                    ToolCall(id="toolu_2", name="dummy", arguments={"q": "y"}),
                ],
                stop_reason="toolUse",
                usage=_usage(2),
            )
            stream.push(StartEvent(partial=create_assistant_message("")))
            stream.push(DoneEvent(reason="toolUse", message=message))
            return stream

        session = await create_agent_session(
            tmp_dir, stream_fn=stream_fn, tools=[tool], base_tools_override={"dummy": tool}
        )
        session_manager = session.session_manager

        snapshots: list[list[str]] = []

        async def emit_tool_call(_event):
            snapshots.append(
                [entry["message"].role for entry in session_manager.get_entries() if entry["type"] == "message"]
            )

        session._extension_runner = _MockRunner(
            has_handlers=lambda event_type: event_type == "tool_call", emit_tool_call=emit_tool_call
        )

        await session.prompt("hi")
        await session.agent.wait_for_idle()

        assert snapshots == [["user", "assistant"], ["user", "assistant"]]
        session.dispose()

    @pytest.mark.tonio
    async def test_persists_message_end_events_in_order_with_slow_extension_handlers(self, tmp_dir):
        from pidrei_agent.types import AgentTool, AgentToolResult

        class DummyTool(AgentTool):
            name = "dummy"
            label = "dummy"
            description = "Dummy tool"
            parameters: ClassVar = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}

            async def execute(self, _tool_call_id, params, cancel=None, on_update=None, ctx=None):
                return AgentToolResult(content=[TextContent(text=f"result:{params.get('q', '')}")], details={})

        tool = DummyTool()

        def stream_fn(_model, context, _options=None):
            stream = AssistantMessageEventStream()
            has_tool_result = any(getattr(m, "role", None) == "toolResult" for m in context.messages)
            if has_tool_result:
                message = create_assistant_message("done", usage=_usage(2))
                stream.push(StartEvent(partial=create_assistant_message("")))
                stream.push(DoneEvent(reason="stop", message=message))
                return stream

            message = create_assistant_message(
                "",
                content=[
                    TextContent(text="calling tool"),
                    ToolCall(id="toolu_1", name="dummy", arguments={"q": "x"}),
                ],
                stop_reason="toolUse",
                usage=_usage(2),
            )
            stream.push(StartEvent(partial=create_assistant_message("")))
            stream.push(DoneEvent(reason="toolUse", message=message))
            return stream

        session = await create_agent_session(
            tmp_dir, stream_fn=stream_fn, tools=[tool], base_tools_override={"dummy": tool}
        )
        session_manager = session.session_manager

        async def emit_message_end(event):
            if getattr(event.get("message"), "role", None) == "assistant":
                await tonio.time.sleep(0.04)

        session._extension_runner = _MockRunner(emit_message_end=emit_message_end)

        await session.prompt("hi")
        await session.agent.wait_for_idle()
        await tonio.time.sleep(0.1)

        message_entries = [entry for entry in session_manager.get_entries() if entry["type"] == "message"]
        assert [entry["message"].role for entry in message_entries] == [
            "user",
            "assistant",
            "toolResult",
            "assistant",
        ]
        session.dispose()


def _usage(total: int) -> Usage:
    return Usage(input=1, output=1, cache_read=0, cache_write=0, total_tokens=total, cost=UsageCost())


class TestRetry:
    async def _create_session(self, tmp_dir, *, fail_count=1, max_retries=3, delay_assistant_message_end_ms=0):
        call_count = {"value": 0}

        def stream_fn(_model, _context, _options=None):
            call_count["value"] += 1
            stream = AssistantMessageEventStream()
            if call_count["value"] <= fail_count:
                msg = create_assistant_message("", stop_reason="error", error_message="overloaded_error")
                stream.push(StartEvent(partial=msg))
                stream.push(ErrorEvent(reason="error", error=msg))
            else:
                msg = create_assistant_message("Success")
                stream.push(StartEvent(partial=msg))
                stream.push(DoneEvent(reason="stop", message=msg))
            return stream

        session = await create_agent_session(
            tmp_dir,
            stream_fn=stream_fn,
            settings_overrides={"retry": {"enabled": True, "maxRetries": max_retries, "baseDelayMs": 1}},
        )

        if delay_assistant_message_end_ms > 0:
            original = session._emit_extension_event

            async def delayed(event):
                if event.type == "message_end" and getattr(event.message, "role", None) == "assistant":
                    await tonio.time.sleep(delay_assistant_message_end_ms / 1000)
                await original(event)

            session._emit_extension_event = delayed

        return session, call_count

    @pytest.mark.tonio
    async def test_retries_after_transient_error_and_succeeds(self, tmp_dir):
        session, call_count = await self._create_session(tmp_dir, fail_count=1)
        events: list[str] = []

        def listener(event):
            if event.type == "auto_retry_start":
                events.append(f"start:{event.attempt}")
            if event.type == "auto_retry_end":
                events.append(f"end:success={event.success}")

        session.subscribe(listener)

        await session.prompt("Test")

        assert call_count["value"] == 2
        assert events == ["start:1", "end:success=True"]
        assert session.is_retrying is False
        session.dispose()

    @pytest.mark.tonio
    async def test_exhausts_max_retries_and_emits_failure(self, tmp_dir):
        session, call_count = await self._create_session(tmp_dir, fail_count=99, max_retries=2)
        events: list[str] = []

        def listener(event):
            if event.type == "auto_retry_start":
                events.append(f"start:{event.attempt}")
            if event.type == "auto_retry_end":
                events.append(f"end:success={event.success}")

        session.subscribe(listener)

        await session.prompt("Test")

        assert call_count["value"] == 3
        assert "start:1" in events
        assert "start:2" in events
        assert "end:success=False" in events
        assert session.is_retrying is False
        session.dispose()

    @pytest.mark.tonio
    async def test_prompt_waits_for_retry_even_when_assistant_message_end_delayed(self, tmp_dir):
        session, call_count = await self._create_session(tmp_dir, fail_count=1, delay_assistant_message_end_ms=40)

        await session.prompt("Test")

        assert call_count["value"] == 2
        assert session.is_retrying is False
        session.dispose()

    @pytest.mark.tonio
    async def test_retries_provider_network_error_failures(self, tmp_dir):
        call_count = {"value": 0}

        def stream_fn(_model, _context, _options=None):
            call_count["value"] += 1
            stream = AssistantMessageEventStream()
            if call_count["value"] == 1:
                msg = create_assistant_message(
                    "", stop_reason="error", error_message="Provider finish_reason: network_error"
                )
                stream.push(StartEvent(partial=msg))
                stream.push(ErrorEvent(reason="error", error=msg))
            else:
                msg = create_assistant_message("Recovered after retry")
                stream.push(StartEvent(partial=msg))
                stream.push(DoneEvent(reason="stop", message=msg))
            return stream

        session = await create_agent_session(
            tmp_dir,
            stream_fn=stream_fn,
            settings_overrides={"retry": {"enabled": True, "maxRetries": 3, "baseDelayMs": 1}},
        )

        events: list[str] = []

        def listener(event):
            if event.type == "auto_retry_start":
                events.append(f"start:{event.attempt}")
            if event.type == "auto_retry_end":
                events.append(f"end:success={event.success}")

        session.subscribe(listener)

        await session.prompt("Test")

        assert call_count["value"] == 2
        assert events == ["start:1", "end:success=True"]
        session.dispose()

    @pytest.mark.tonio
    async def test_prompt_waits_for_full_agent_loop_when_retry_produces_tool_calls(self, tmp_dir):
        # Regression: when auto-retry fires and the retry response includes tool
        # use, session.prompt() must wait for the entire tool loop to finish.
        from pidrei_agent.types import AgentTool, AgentToolResult

        call_count = {"value": 0}
        tool_executed = {"value": False}

        class EchoTool(AgentTool):
            name = "echo"
            label = "Echo"
            description = "Echo text back"
            parameters: ClassVar = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

            async def execute(self, _tool_call_id, _params, cancel=None, on_update=None, ctx=None):
                tool_executed["value"] = True
                return AgentToolResult(content=[TextContent(text="echoed")], details=None)

        echo_tool = EchoTool()

        def stream_fn(_model, _context, _options=None):
            call_count["value"] += 1
            stream = AssistantMessageEventStream()
            if call_count["value"] == 1:
                msg = create_assistant_message("", stop_reason="error", error_message="overloaded_error")
                stream.push(StartEvent(partial=msg))
                stream.push(ErrorEvent(reason="error", error=msg))
            elif call_count["value"] == 2:
                msg = create_assistant_message(
                    "",
                    stop_reason="toolUse",
                    content=[
                        TextContent(text="Looking that up now."),
                        ToolCall(id="call_1", name="echo", arguments={"text": "hello"}),
                    ],
                )
                stream.push(StartEvent(partial=msg))
                stream.push(DoneEvent(reason="toolUse", message=msg))
            else:
                msg = create_assistant_message("Final answer.")
                stream.push(StartEvent(partial=msg))
                stream.push(DoneEvent(reason="stop", message=msg))
            return stream

        session = await create_agent_session(
            tmp_dir,
            stream_fn=stream_fn,
            tools=[echo_tool],
            base_tools_override={"echo": echo_tool},
            settings_overrides={"retry": {"enabled": True, "maxRetries": 3, "baseDelayMs": 1}},
        )

        await session.prompt("Test")

        # All three LLM calls must have completed
        assert call_count["value"] == 3
        # Tool must have been executed
        assert tool_executed["value"] is True
        # Agent must not be streaming after prompt returns
        assert session.is_streaming is False
        # A follow-up prompt must work (no "Agent is already processing" error)
        await session.prompt("Follow-up")
        assert call_count["value"] == 4
        session.dispose()


_STATS_MODEL = get_builtin_model("anthropic", "claude-sonnet-4-5")


def _stats_usage(total_tokens: int) -> Usage:
    return Usage(input=total_tokens, output=0, cache_read=0, cache_write=0, total_tokens=total_tokens, cost=UsageCost())


def _stats_assistant(text: str, total_tokens: int, timestamp: int) -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text=text)],
        api=_STATS_MODEL.api,
        provider=_STATS_MODEL.provider,
        model=_STATS_MODEL.id,
        usage=_stats_usage(total_tokens),
        stop_reason="stop",
        timestamp=timestamp,
    )


def _stats_tool_result(usage: Usage) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id="tool-call-1",
        tool_name="test_tool",
        content=[TextContent(text="tool result")],
        usage=usage,
        is_error=False,
        timestamp=1,
    )


def _rich_usage() -> Usage:
    return Usage(
        input=10,
        output=20,
        cache_read=30,
        cache_write=40,
        total_tokens=100,
        cost=UsageCost(input=0.1, output=0.2, cache_read=0.3, cache_write=0.4, total=1),
    )


async def _create_stats_session():
    settings_manager = SettingsManager.in_memory()
    session_manager = SessionManager.in_memory()
    auth_storage = AuthStorage.in_memory()
    from pidrei_ai.auth.types import ApiKeyCredential

    async def set_key(_credential):
        return ApiKeyCredential(key="test-key")

    await auth_storage.modify("anthropic", set_key)
    from pidrei.core.model_runtime import ModelRuntime

    model_runtime = await ModelRuntime.create(credentials=auth_storage, models_path=None, allow_model_network=False)

    def stream_fn(*_args, **_kwargs):
        raise Exception("unused")

    session = AgentSession(
        AgentSessionConfig(
            agent=Agent(
                stream_fn=stream_fn,
                get_api_key=lambda _provider: "test-key",
                initial_state=AgentInitialState(
                    model=_STATS_MODEL,
                    system_prompt="You are a helpful assistant.",
                    tools=[],
                    thinking_level="high",
                ),
            ),
            session_manager=session_manager,
            settings_manager=settings_manager,
            cwd=".",
            model_runtime=model_runtime,
            resource_loader=create_test_resource_loader(),
        )
    )
    return session, session_manager


def _sync_agent_messages(session: AgentSession, session_manager: SessionManager) -> None:
    session.agent.state.messages = session_manager.build_session_context().messages


class TestGetSessionStats:
    @pytest.mark.tonio
    async def test_exposes_current_context_usage_alongside_token_totals(self):
        session, session_manager = await _create_stats_session()
        try:
            await session_manager.append_message(UserMessage(content="hello", timestamp=1))
            await session_manager.append_message(_stats_assistant("hi", 200, 2))
            _sync_agent_messages(session, session_manager)

            stats = session.get_session_stats()
            assert stats.context_usage == session.get_context_usage()
            assert stats.context_usage.tokens == 200
            assert stats.context_usage.context_window == _STATS_MODEL.context_window
            assert stats.context_usage.percent == (200 / _STATS_MODEL.context_window) * 100
        finally:
            session.dispose()

    @pytest.mark.tonio
    async def test_reports_unknown_context_usage_immediately_after_compaction(self):
        session, session_manager = await _create_stats_session()
        try:
            await session_manager.append_message(UserMessage(content="first", timestamp=1))
            await session_manager.append_message(_stats_assistant("response1", 180_000, 2))
            kept_user_id = await session_manager.append_message(UserMessage(content="second", timestamp=3))
            await session_manager.append_message(_stats_assistant("response2", 195_000, 4))
            await session_manager.append_compaction("summary", kept_user_id, 195_000)
            await session_manager.append_message(UserMessage(content="third", timestamp=5))
            _sync_agent_messages(session, session_manager)

            stats = session.get_session_stats()
            # Totals cover ALL entries, including history compacted away.
            assert stats.tokens.input == 375_000
            assert stats.context_usage is not None
            assert stats.context_usage.tokens is None
            assert stats.context_usage.percent is None
        finally:
            session.dispose()

    @pytest.mark.tonio
    async def test_uses_post_compaction_usage_instead_of_stale_kept_usage(self):
        session, session_manager = await _create_stats_session()
        try:
            await session_manager.append_message(UserMessage(content="first", timestamp=1))
            await session_manager.append_message(_stats_assistant("response1", 180_000, 2))
            kept_user_id = await session_manager.append_message(UserMessage(content="second", timestamp=3))
            await session_manager.append_message(_stats_assistant("response2", 195_000, 4))
            await session_manager.append_compaction("summary", kept_user_id, 195_000)
            await session_manager.append_message(UserMessage(content="third", timestamp=5))
            await session_manager.append_message(_stats_assistant("response3", 25_000, 6))
            _sync_agent_messages(session, session_manager)

            stats = session.get_session_stats()
            assert stats.tokens.input == 400_000
            assert stats.context_usage is not None
            assert stats.context_usage.tokens == 25_000
            assert stats.context_usage.percent == (25_000 / _STATS_MODEL.context_window) * 100
        finally:
            session.dispose()

    @pytest.mark.tonio
    async def test_includes_branch_summary_usage_in_session_totals(self):
        session, session_manager = await _create_stats_session()
        try:
            await session_manager.branch_with_summary(None, "summary", None, False, _rich_usage())
            _sync_agent_messages(session, session_manager)

            stats = session.get_session_stats()
            assert (stats.tokens.input, stats.tokens.output, stats.tokens.cache_read, stats.tokens.cache_write) == (
                10,
                20,
                30,
                40,
            )
            assert stats.tokens.total == 100
            assert stats.cost == 1
        finally:
            session.dispose()

    @pytest.mark.tonio
    async def test_includes_compaction_usage_in_session_totals(self):
        session, session_manager = await _create_stats_session()
        try:
            first_kept_entry_id = await session_manager.append_message(UserMessage(content="hello", timestamp=1))
            await session_manager.append_compaction("summary", first_kept_entry_id, 100, None, False, _rich_usage())
            _sync_agent_messages(session, session_manager)

            stats = session.get_session_stats()
            assert stats.tokens.total == 100
            assert stats.cost == 1
        finally:
            session.dispose()

    @pytest.mark.tonio
    async def test_includes_tool_result_usage_in_session_totals(self):
        session, session_manager = await _create_stats_session()
        try:
            await session_manager.append_message(_stats_tool_result(_rich_usage()))
            _sync_agent_messages(session, session_manager)

            stats = session.get_session_stats()
            assert stats.tokens.total == 100
            assert stats.cost == 1
        finally:
            session.dispose()

    @pytest.mark.tonio
    async def test_groups_tool_and_summary_usage_separately_from_model_usage(self):
        from dataclasses import replace

        session_manager = SessionManager.in_memory()
        root_id = await session_manager.append_message(UserMessage(content="hello", timestamp=1))
        assistant = _stats_assistant("response", 100, 2)
        assistant.usage = replace(_stats_usage(100), cost=UsageCost(total=0.5))
        await session_manager.append_message(assistant)
        await session_manager.append_message(_stats_tool_result(replace(_stats_usage(100), cost=UsageCost(total=1))))
        await session_manager.append_compaction(
            "summary", root_id, 100, None, False, replace(_stats_usage(100), cost=UsageCost(total=2))
        )
        await session_manager.branch_with_summary(
            None, "branch summary", None, False, replace(_stats_usage(100), cost=UsageCost(total=3))
        )

        breakdown = get_usage_cost_breakdown(session_manager.get_entries())
        assert [(entry.key, entry.cost, entry.tokens) for entry in breakdown] == [
            ("Tools/summaries", 6, 300),
            (f"{_STATS_MODEL.provider}/{_STATS_MODEL.id}", 0.5, 100),
        ]

    @pytest.mark.tonio
    async def test_ignores_zero_usage_messages_when_checking_post_compaction_usage(self):
        session, session_manager = await _create_stats_session()
        try:
            await session_manager.append_message(UserMessage(content="first", timestamp=1))
            await session_manager.append_message(_stats_assistant("response1", 180_000, 2))
            kept_user_id = await session_manager.append_message(UserMessage(content="second", timestamp=3))
            await session_manager.append_message(_stats_assistant("response2", 195_000, 4))
            await session_manager.append_compaction("summary", kept_user_id, 195_000)
            await session_manager.append_message(UserMessage(content="third", timestamp=5))
            await session_manager.append_message(_stats_assistant("response3", 25_000, 6))
            await session_manager.append_message(UserMessage(content="continue", timestamp=7))
            await session_manager.append_message(_stats_assistant("partial", 0, 8))
            _sync_agent_messages(session, session_manager)

            stats = session.get_session_stats()
            assert stats.context_usage is not None
            assert stats.context_usage.tokens is not None
            assert stats.context_usage.tokens > 25_000
        finally:
            session.dispose()


class TestAutoCompactionQueue:
    async def _create_session(self, tmp_dir):
        def stream_fn(*_args, **_kwargs):
            raise Exception("unused")

        session = await create_agent_session(tmp_dir, stream_fn=stream_fn)
        return session, session.session_manager, session.settings_manager

    @pytest.mark.tonio
    async def test_resumes_after_threshold_compaction_with_only_agent_level_queued_messages(self, tmp_dir):
        session, session_manager, settings_manager = await self._create_session(tmp_dir)
        settings_manager.apply_overrides({"compaction": {"keepRecentTokens": 1}})
        model = session.model
        now = now_ms()
        await session_manager.append_message(
            UserMessage(content=[TextContent(text="message to compact")], timestamp=now - 1000)
        )
        await session_manager.append_message(
            AssistantMessage(
                content=[TextContent(text="assistant response to compact")],
                api=model.api,
                provider=model.provider,
                model=model.id,
                usage=Usage(input=100, output=0, cache_read=0, cache_write=0, total_tokens=100, cost=UsageCost()),
                stop_reason="stop",
                timestamp=now - 500,
            )
        )
        session.agent.state.messages = session_manager.build_session_context().messages

        def summary_stream_fn(summary_model, _context, _options=None):
            stream = AssistantMessageEventStream()
            from pidrei_ai.providers.faux import faux_assistant_message

            message = faux_assistant_message("compacted")
            message.api = summary_model.api
            message.provider = summary_model.provider
            message.model = summary_model.id
            message.usage = Usage(input=10, output=0, cache_read=0, cache_write=0, total_tokens=10, cost=UsageCost())
            stream.push(DoneEvent(reason="stop", message=message))
            return stream

        session.agent.stream_function = summary_stream_fn

        from pidrei.core.messages import CustomMessage

        session.agent.follow_up(
            CustomMessage(
                custom_type="test",
                content=[TextContent(text="Queued custom")],
                display=False,
                timestamp=now_ms(),
            )
        )

        assert session.pending_message_count == 0
        assert session.agent.has_queued_messages() is True

        continue_calls = {"value": 0}

        async def continue_spy():
            continue_calls["value"] += 1

        session.agent.continue_ = continue_spy

        assert await session._run_auto_compaction("threshold", False) is True
        assert continue_calls["value"] == 0
        session.dispose()

    @pytest.mark.tonio
    async def test_does_not_compact_repeatedly_after_overflow_recovery_attempted(self, tmp_dir):
        session, _session_manager, _settings = await self._create_session(tmp_dir)
        model = session.model
        overflow_message = AssistantMessage(
            content=[TextContent(text="")],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=Usage(),
            stop_reason="error",
            error_message="prompt is too long",
            timestamp=now_ms(),
        )

        run_calls = []

        async def run_auto_compaction_spy(reason, will_retry):
            run_calls.append((reason, will_retry))

        session._run_auto_compaction = run_auto_compaction_spy

        events = []

        def listener(event):
            if event.type == "compaction_end":
                events.append((event.type, event.reason, event.error_message))

        session.subscribe(listener)

        await session._check_compaction(overflow_message)
        from dataclasses import replace

        await session._check_compaction(replace(overflow_message, timestamp=now_ms() + 1))

        assert len(run_calls) == 1
        assert (
            "compaction_end",
            "overflow",
            (
                "Context overflow recovery failed after one compact-and-retry attempt. "
                "Try reducing context or switching to a larger-context model."
            ),
        ) in events
        session.dispose()

    @pytest.mark.tonio
    async def test_ignores_stale_pre_compaction_usage_on_pre_prompt_checks(self, tmp_dir):
        session, session_manager, _settings = await self._create_session(tmp_dir)
        model = session.model
        stale_assistant_timestamp = now_ms() - 10_000
        stale_assistant = AssistantMessage(
            content=[TextContent(text="large response before compaction")],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=Usage(input=600_000, output=10_000, total_tokens=610_000, cost=UsageCost()),
            stop_reason="stop",
            timestamp=stale_assistant_timestamp,
        )

        await session_manager.append_message(
            UserMessage(content=[TextContent(text="before compaction")], timestamp=stale_assistant_timestamp - 1000)
        )
        await session_manager.append_message(stale_assistant)

        first_kept_entry_id = session_manager.get_entries()[0]["id"]
        await session_manager.append_compaction(
            "summary", first_kept_entry_id, stale_assistant.usage.total_tokens, None, False
        )

        await session_manager.append_message(
            UserMessage(content=[TextContent(text="session recovery payload")], timestamp=now_ms())
        )

        run_calls = []

        async def run_auto_compaction_spy(reason, will_retry):
            run_calls.append((reason, will_retry))

        session._run_auto_compaction = run_auto_compaction_spy

        await session._check_compaction(stale_assistant, False)

        assert run_calls == []
        session.dispose()

    @pytest.mark.tonio
    async def test_triggers_threshold_compaction_for_error_messages_using_last_successful_usage(self, tmp_dir):
        session, _session_manager, settings_manager = await self._create_session(tmp_dir)
        model = session.model

        # A successful assistant message with token usage just over the threshold.
        compaction_settings = settings_manager.get_compaction_settings()
        threshold_tokens = (model.context_window or 200_000) - compaction_settings["reserve_tokens"] + 1
        successful_assistant = AssistantMessage(
            content=[TextContent(text="large successful response")],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=Usage(
                input=threshold_tokens - 10_000, output=10_000, total_tokens=threshold_tokens, cost=UsageCost()
            ),
            stop_reason="stop",
            timestamp=now_ms(),
        )

        # An error message (e.g. 529 overloaded) with no useful usage data
        error_assistant = AssistantMessage(
            content=[TextContent(text="")],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=Usage(),
            stop_reason="error",
            error_message="529 overloaded",
            timestamp=now_ms() + 1000,
        )

        session.agent.state.messages = [
            UserMessage(content=[TextContent(text="hello")], timestamp=now_ms() - 1000),
            successful_assistant,
            UserMessage(content=[TextContent(text="another prompt")], timestamp=now_ms() + 500),
            error_assistant,
        ]

        run_calls = []

        async def run_auto_compaction_spy(reason, will_retry):
            run_calls.append((reason, will_retry))

        session._run_auto_compaction = run_auto_compaction_spy

        await session._check_compaction(error_assistant)

        assert run_calls == [("threshold", False)]
        session.dispose()

    @pytest.mark.tonio
    async def test_does_not_trigger_threshold_compaction_when_no_prior_usage_exists(self, tmp_dir):
        session, _session_manager, _settings = await self._create_session(tmp_dir)
        model = session.model

        error_assistant = AssistantMessage(
            content=[TextContent(text="")],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=Usage(),
            stop_reason="error",
            error_message="529 overloaded",
            timestamp=now_ms(),
        )

        session.agent.state.messages = [
            UserMessage(content=[TextContent(text="hello")], timestamp=now_ms() - 1000),
            error_assistant,
        ]

        run_calls = []

        async def run_auto_compaction_spy(reason, will_retry):
            run_calls.append((reason, will_retry))

        session._run_auto_compaction = run_auto_compaction_spy

        await session._check_compaction(error_assistant)

        assert run_calls == []
        session.dispose()

    @pytest.mark.tonio
    async def test_does_not_trigger_threshold_compaction_with_only_kept_pre_compaction_usage(self, tmp_dir):
        session, session_manager, _settings = await self._create_session(tmp_dir)
        model = session.model
        pre_compaction_timestamp = now_ms() - 10_000

        kept_assistant = AssistantMessage(
            content=[TextContent(text="kept response from before compaction")],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=Usage(input=180_000, output=10_000, total_tokens=190_000, cost=UsageCost()),
            stop_reason="stop",
            timestamp=pre_compaction_timestamp,
        )

        await session_manager.append_message(
            UserMessage(content=[TextContent(text="before compaction")], timestamp=pre_compaction_timestamp - 1000)
        )
        await session_manager.append_message(kept_assistant)
        first_kept_entry_id = session_manager.get_entries()[0]["id"]
        await session_manager.append_compaction(
            "summary", first_kept_entry_id, kept_assistant.usage.total_tokens, None, False
        )

        error_assistant = AssistantMessage(
            content=[TextContent(text="")],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=Usage(),
            stop_reason="error",
            error_message="529 overloaded",
            timestamp=now_ms(),
        )

        session.agent.state.messages = [
            UserMessage(content=[TextContent(text="kept user msg")], timestamp=pre_compaction_timestamp - 1000),
            kept_assistant,
            UserMessage(content=[TextContent(text="new prompt")], timestamp=now_ms() - 500),
            error_assistant,
        ]

        run_calls = []

        async def run_auto_compaction_spy(reason, will_retry):
            run_calls.append((reason, will_retry))

        session._run_auto_compaction = run_auto_compaction_spy

        await session._check_compaction(error_assistant)

        # Should NOT compact: the only usage data is from a kept pre-compaction message
        assert run_calls == []
        session.dispose()
