"""Mirrors pi coding-agent test/agent-session-runtime-events.test.ts,
agent-session-branching.test.ts, sdk-session-manager.test.ts, and
session-info-modified-timestamp.test.ts.

Substitutions (documented): pi loads inline extensions through the extension
loader (Phase 5 here); these mirrors hand-build Extension records with the
same handler shapes. pi's faux-provider/real-LLM streams become a patched
`model_runtime.stream_simple` returning canned assistant messages.
"""

import json
import os
from datetime import UTC, datetime

import pytest
import tonio.colored as tonio

from pidrei.core.agent_session_runtime import create_agent_session_runtime
from pidrei.core.agent_session_services import (
    AgentSessionServices,
    CreateAgentSessionFromServicesOptions,
    create_agent_session_from_services,
)
from pidrei.core.auth_storage import AuthStorage
from pidrei.core.extensions import Extension, ExtensionRuntime, LoadExtensionsResult
from pidrei.core.model_runtime import ModelRuntime
from pidrei.core.sdk import CreateAgentSessionOptions, create_agent_session
from pidrei.core.session_manager import SessionManager
from pidrei_ai.auth.types import ApiKeyCredential
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import DoneEvent, StartEvent, Usage, UsageCost
from pidrei_ai.utils.event_stream import AssistantMessageEventStream

from .agent_session_helpers import create_assistant_message, create_test_resource_loader
from .coding_session_helpers import now_ms


def _canned_stream_simple(_model, _context, _options=None):
    stream = AssistantMessageEventStream()
    message = create_assistant_message(
        "canned",
        provider="anthropic",
        model="claude-sonnet-4-5",
        usage=Usage(input=5, output=5, total_tokens=10, cost=UsageCost()),
    )
    stream.push(StartEvent(partial=create_assistant_message("")))
    stream.push(DoneEvent(reason="stop", message=message))
    return stream


async def _create_model_runtime(temp_dir: str) -> ModelRuntime:
    auth_storage = AuthStorage.in_memory()

    async def set_key(_credential):
        return ApiKeyCredential(key="test-key")

    await auth_storage.modify("anthropic", set_key)
    model_runtime = await ModelRuntime.create(
        credentials=auth_storage,
        models_path=os.path.join(temp_dir, "models.json"),
        allow_model_network=False,
    )
    model_runtime.stream_simple = _canned_stream_simple
    return model_runtime


async def _create_runtime_host(temp_dir: str, extensions: list[Extension]):
    """pi's createRuntimeHost with inline extensions and a faux provider."""
    temp_dir = str(temp_dir)
    model = get_builtin_model("anthropic", "claude-sonnet-4-5")
    model_runtime = await _create_model_runtime(temp_dir)
    extensions_result = LoadExtensionsResult(extensions=extensions, runtime=ExtensionRuntime())

    async def create_runtime(*, cwd, agent_dir, session_manager, session_start_event=None, **_kwargs):
        services = AgentSessionServices(
            cwd=cwd,
            agent_dir=agent_dir,
            model_runtime=model_runtime,
            settings_manager=await __import__(
                "pidrei.core.settings_manager", fromlist=["SettingsManager"]
            ).SettingsManager.create(cwd, agent_dir),
            resource_loader=create_test_resource_loader(extensions_result),
        )
        result = await create_agent_session_from_services(
            CreateAgentSessionFromServicesOptions(
                services=services,
                session_manager=session_manager,
                session_start_event=session_start_event,
                model=model,
            )
        )
        from pidrei.core.agent_session_runtime import CreateAgentSessionRuntimeResult

        return CreateAgentSessionRuntimeResult(
            session=result.session,
            services=services,
            extensions_result=result.extensions_result,
            model_fallback_message=result.model_fallback_message,
            diagnostics=services.diagnostics,
        )

    runtime_host = await create_agent_session_runtime(
        create_runtime,
        cwd=temp_dir,
        agent_dir=temp_dir,
        session_manager=await SessionManager.create(temp_dir, temp_dir),
    )
    await runtime_host.session.bind_extensions(
        __import__("pidrei.core.agent_session", fromlist=["ExtensionBindings"]).ExtensionBindings()
    )
    return runtime_host


def _extension(handlers: dict) -> Extension:
    return Extension(path="<inline:1>", handlers={key: [value] for key, value in handlers.items()})


def _recorder(events: list):
    async def record(event, _ctx):
        events.append(event)

    return record


class TestRuntimeSessionLifecycleEvents:
    @pytest.mark.tonio
    async def test_emits_before_switch_and_session_start_for_new_and_resume(self, tmp_dir):
        events = []
        runtime_host = await _create_runtime_host(
            tmp_dir,
            [
                _extension(
                    {
                        "session_before_switch": _recorder(events),
                        "session_shutdown": _recorder(events),
                        "session_start": _recorder(events),
                    }
                )
            ],
        )

        assert events == [{"type": "session_start", "reason": "startup"}]
        events.clear()

        await runtime_host.session.prompt("hello")
        original_session_file = runtime_host.session.session_file
        assert original_session_file

        new_session_result = await runtime_host.new_session()
        assert new_session_result["cancelled"] is False
        from pidrei.core.agent_session import ExtensionBindings

        await runtime_host.session.bind_extensions(ExtensionBindings())
        second_session_file = runtime_host.session.session_file
        assert events == [
            {"type": "session_before_switch", "reason": "new", "targetSessionFile": None},
            {"type": "session_shutdown", "reason": "new", "targetSessionFile": second_session_file},
            {"type": "session_start", "reason": "new", "previousSessionFile": original_session_file},
        ]

        events.clear()
        assert second_session_file

        switch_result = await runtime_host.switch_session(original_session_file)
        assert switch_result["cancelled"] is False
        await runtime_host.session.bind_extensions(ExtensionBindings())
        assert events == [
            {"type": "session_before_switch", "reason": "resume", "targetSessionFile": original_session_file},
            {"type": "session_shutdown", "reason": "resume", "targetSessionFile": original_session_file},
            {"type": "session_start", "reason": "resume", "previousSessionFile": second_session_file},
        ]
        await runtime_host.dispose()

    @pytest.mark.tonio
    async def test_honors_session_before_switch_cancellation(self, tmp_dir):
        events = []

        async def before_switch(event, _ctx):
            events.append(event)
            return {"cancel": True}

        runtime_host = await _create_runtime_host(
            tmp_dir,
            [
                _extension(
                    {
                        "session_before_switch": before_switch,
                        "session_start": _recorder(events),
                    }
                )
            ],
        )

        assert events == [{"type": "session_start", "reason": "startup"}]
        events.clear()

        await runtime_host.session.prompt("hello")
        original_session_file = runtime_host.session.session_file

        result = await runtime_host.new_session()
        assert result["cancelled"] is True
        assert runtime_host.session.session_file == original_session_file
        assert events == [{"type": "session_before_switch", "reason": "new", "targetSessionFile": None}]
        await runtime_host.dispose()

    @pytest.mark.tonio
    async def test_runs_before_session_invalidate_after_shutdown_and_before_rebind(self, tmp_dir):
        phases = []

        async def on_shutdown(_event, _ctx):
            phases.append("session_shutdown")

        runtime_host = await _create_runtime_host(
            tmp_dir,
            [_extension({"session_shutdown": on_shutdown})],
        )
        old_session = runtime_host.session

        def before_invalidate():
            phases.append("beforeSessionInvalidate")
            assert old_session.extension_runner.create_context().cwd == old_session.session_manager.get_cwd()

        async def rebind(_session):
            phases.append("rebindSession")

        runtime_host.set_before_session_invalidate(before_invalidate)
        runtime_host.set_rebind_session(rebind)

        await runtime_host.new_session()

        assert phases == ["session_shutdown", "beforeSessionInvalidate", "rebindSession"]
        with pytest.raises(Exception, match="This extension ctx is stale after session replacement or reload"):
            _ = old_session.extension_runner.create_context().cwd
        runtime_host.set_before_session_invalidate(None)
        runtime_host.set_rebind_session(None)
        await runtime_host.dispose()

    @pytest.mark.tonio
    async def test_emits_session_before_fork_and_honors_cancellation(self, tmp_dir):
        events = []
        cancel_next_fork = {"value": False}

        async def before_fork(event, _ctx):
            events.append(event)
            if cancel_next_fork["value"]:
                cancel_next_fork["value"] = False
                return {"cancel": True}
            return None

        runtime_host = await _create_runtime_host(
            tmp_dir,
            [
                _extension(
                    {
                        "session_before_fork": before_fork,
                        "session_shutdown": _recorder(events),
                        "session_start": _recorder(events),
                    }
                )
            ],
        )

        assert events == [{"type": "session_start", "reason": "startup"}]
        events.clear()

        await runtime_host.session.prompt("hello")
        user_message = runtime_host.session.get_user_messages_for_forking()[0]
        previous_session_file = runtime_host.session.session_file

        success_result = await runtime_host.fork(user_message["entryId"])
        assert success_result["cancelled"] is False
        assert success_result["selectedText"] == "hello"
        from pidrei.core.agent_session import ExtensionBindings

        await runtime_host.session.bind_extensions(ExtensionBindings())
        assert events == [
            {"type": "session_before_fork", "entryId": user_message["entryId"], "position": "before"},
            {"type": "session_shutdown", "reason": "fork", "targetSessionFile": runtime_host.session.session_file},
            {"type": "session_start", "reason": "fork", "previousSessionFile": previous_session_file},
        ]

        events.clear()
        cancel_next_fork["value"] = True
        cancel_result = await runtime_host.fork(user_message["entryId"])
        assert cancel_result == {"cancelled": True}
        assert events == [{"type": "session_before_fork", "entryId": user_message["entryId"], "position": "before"}]

        events.clear()
        cancel_next_fork["value"] = True
        cancel_at_result = await runtime_host.fork("missing-entry", position="at")
        assert cancel_at_result == {"cancelled": True}
        assert events == [{"type": "session_before_fork", "entryId": "missing-entry", "position": "at"}]
        await runtime_host.dispose()


class TestForkingSuite:
    @pytest.mark.tonio
    async def test_allows_forking_from_single_message(self, tmp_dir):
        runtime_host = await _create_runtime_host(tmp_dir, [])
        session = runtime_host.session

        await session.prompt("Say hello")
        await session.agent.wait_for_idle()

        user_messages = session.get_user_messages_for_forking()
        assert len(user_messages) == 1
        assert user_messages[0]["text"] == "Say hello"

        result = await runtime_host.fork(user_messages[0]["entryId"])
        assert result["cancelled"] is False
        session = runtime_host.session
        assert result["selectedText"] == "Say hello"

        assert len(session.messages) == 0
        assert session.session_file is not None
        assert not os.path.exists(session.session_file)
        await runtime_host.dispose()

    @pytest.mark.tonio
    async def test_supports_in_memory_forking_in_no_session_mode(self, tmp_dir):
        temp_dir = str(tmp_dir)
        model = get_builtin_model("anthropic", "claude-sonnet-4-5")
        model_runtime = await _create_model_runtime(temp_dir)
        extensions_result = LoadExtensionsResult(runtime=ExtensionRuntime())

        async def create_runtime(*, cwd, agent_dir, session_manager, session_start_event=None, **_kwargs):
            from pidrei.core.agent_session_runtime import CreateAgentSessionRuntimeResult
            from pidrei.core.settings_manager import SettingsManager

            services = AgentSessionServices(
                cwd=cwd,
                agent_dir=agent_dir,
                model_runtime=model_runtime,
                settings_manager=await SettingsManager.create(cwd, agent_dir),
                resource_loader=create_test_resource_loader(extensions_result),
            )
            result = await create_agent_session_from_services(
                CreateAgentSessionFromServicesOptions(
                    services=services,
                    session_manager=session_manager,
                    session_start_event=session_start_event,
                    model=model,
                )
            )
            return CreateAgentSessionRuntimeResult(
                session=result.session, services=services, diagnostics=services.diagnostics
            )

        runtime_host = await create_agent_session_runtime(
            create_runtime,
            cwd=temp_dir,
            agent_dir=temp_dir,
            session_manager=SessionManager.in_memory(temp_dir),
        )
        session = runtime_host.session
        session.subscribe(lambda _event: None)

        assert session.session_file is None

        await session.prompt("Say hi")
        await session.agent.wait_for_idle()

        user_messages = session.get_user_messages_for_forking()
        assert len(user_messages) == 1
        assert len(session.messages) > 0

        result = await runtime_host.fork(user_messages[0]["entryId"])
        assert result["cancelled"] is False
        session = runtime_host.session
        assert result["selectedText"] == "Say hi"

        assert len(session.messages) == 0
        assert session.session_file is None
        await runtime_host.dispose()

    @pytest.mark.tonio
    async def test_forks_from_middle_of_conversation(self, tmp_dir):
        runtime_host = await _create_runtime_host(tmp_dir, [])
        session = runtime_host.session

        await session.prompt("Say one")
        await session.agent.wait_for_idle()
        await session.prompt("Say two")
        await session.agent.wait_for_idle()
        await session.prompt("Say three")
        await session.agent.wait_for_idle()

        user_messages = session.get_user_messages_for_forking()
        assert len(user_messages) == 3

        second_message = user_messages[1]
        result = await runtime_host.fork(second_message["entryId"])
        assert result["cancelled"] is False
        session = runtime_host.session
        assert result["selectedText"] == "Say two"

        assert len(session.messages) == 2
        assert session.messages[0].role == "user"
        assert session.messages[1].role == "assistant"
        await runtime_host.dispose()


class TestSdkSessionManagerDefaults:
    @pytest.mark.tonio
    async def test_uses_agent_dir_for_default_persisted_session_path(self, tmp_dir):
        import re

        cwd = os.path.join(str(tmp_dir), "project")
        agent_dir = os.path.join(str(tmp_dir), "agent")
        os.makedirs(cwd)
        os.makedirs(agent_dir)

        model = get_builtin_model("anthropic", "claude-sonnet-4-5")
        result = await create_agent_session(CreateAgentSessionOptions(cwd=cwd, agent_dir=agent_dir, model=model))
        session = result.session

        safe_path = "--" + re.sub(r"[/\\:]", "-", re.sub(r"^[/\\]", "", cwd)) + "--"
        expected_session_dir = os.path.join(agent_dir, "sessions", safe_path)
        session_dir = session.session_manager.get_session_dir()
        session_file = session.session_manager.get_session_file()

        assert session_dir == expected_session_dir
        assert session_file.startswith(expected_session_dir + "/")

        session.dispose()

    @pytest.mark.tonio
    async def test_keeps_explicit_session_manager_override(self, tmp_dir):
        cwd = os.path.join(str(tmp_dir), "project")
        agent_dir = os.path.join(str(tmp_dir), "agent")
        os.makedirs(cwd)
        os.makedirs(agent_dir)

        model = get_builtin_model("anthropic", "claude-sonnet-4-5")
        session_manager = SessionManager.in_memory(cwd)
        result = await create_agent_session(
            CreateAgentSessionOptions(cwd=cwd, agent_dir=agent_dir, model=model, session_manager=session_manager)
        )
        session = result.session

        assert session.session_manager is session_manager
        assert session.session_manager.is_persisted() is False

        session.dispose()

    @pytest.mark.tonio
    async def test_derives_cwd_from_explicit_session_manager_when_cwd_omitted(self, tmp_dir):
        agent_dir = os.path.join(str(tmp_dir), "agent")
        os.makedirs(agent_dir)
        session_cwd = os.path.join(str(tmp_dir), "session-project")
        os.makedirs(session_cwd)

        model = get_builtin_model("anthropic", "claude-sonnet-4-5")
        session_manager = SessionManager.in_memory(session_cwd)
        result = await create_agent_session(
            CreateAgentSessionOptions(agent_dir=agent_dir, model=model, session_manager=session_manager)
        )
        session = result.session

        assert session.session_manager is session_manager
        assert f"Current working directory: {session_cwd}" in session.system_prompt

        bash_tool = next((tool for tool in session.agent.state.tools if tool.name == "bash"), None)
        assert bash_tool is not None
        tool_result = await bash_tool.execute("test", {"command": "pwd"})
        output = "".join(block.text for block in tool_result.content if getattr(block, "type", None) == "text")

        assert os.path.realpath(output.strip()) == os.path.realpath(session_cwd)

        session.dispose()

    @pytest.mark.tonio
    async def test_exposes_current_session_state_to_built_in_bash_tool(self, tmp_dir):
        cwd = os.path.join(str(tmp_dir), "project")
        agent_dir = os.path.join(str(tmp_dir), "agent")
        os.makedirs(cwd)
        os.makedirs(agent_dir)

        model = get_builtin_model("anthropic", "claude-sonnet-4-5")
        result = await create_agent_session(
            CreateAgentSessionOptions(cwd=cwd, agent_dir=agent_dir, model=model, thinking_level="high")
        )
        session = result.session

        assert session.session_file
        assert "Inspect PIDREI_* environment variables for current model and session details." in session.system_prompt

        bash_tool = next((tool for tool in session.agent.state.tools if tool.name == "bash"), None)
        assert bash_tool is not None
        command = (
            'printf \'%s\\n\' "$PIDREI_SESSION_ID" "$PIDREI_SESSION_FILE" '
            '"$PIDREI_PROVIDER" "$PIDREI_MODEL" "$PIDREI_REASONING_LEVEL"'
        )
        tool_result = await bash_tool.execute("test", {"command": command})
        output = "".join(block.text for block in tool_result.content if getattr(block, "type", None) == "text")

        assert output.strip().split("\n") == [
            session.session_id,
            session.session_file,
            model.provider,
            model.id,
            session.thinking_level,
        ]

        session.dispose()


@pytest.mark.tonio
async def test_session_info_modified_uses_last_message_timestamp(tmp_dir):
    """Mirrors pi test/session-info-modified-timestamp.test.ts."""
    from .coding_session_helpers import assistant_msg

    file_path = os.path.join(str(tmp_dir), "session-modified.jsonl")
    header = {
        "type": "session",
        "id": "test-session",
        "version": 3,
        "timestamp": "1970-01-01T00:00:00.000Z",
        "cwd": "/tmp",
    }
    with open(file_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(header) + "\n")

    # SessionManager only persists once it has seen at least one assistant
    # message; add one so subsequent appends are persisted.
    mgr = await SessionManager.open(file_path)
    await mgr.append_message(assistant_msg("hi", api="openai-completions", provider="openai"))

    before_mtime = os.stat(file_path).st_mtime
    await tonio.time.sleep(0.01)

    mgr = await SessionManager.open(file_path)
    msg_time = now_ms()
    await mgr.append_message(assistant_msg("later", api="openai-completions", provider="openai", timestamp=msg_time))

    sessions = await SessionManager.list("/tmp", os.path.dirname(file_path))
    session_info = next((s for s in sessions if s.path == file_path), None)
    assert session_info is not None
    assert int(session_info.modified.timestamp() * 1000) == msg_time
    assert session_info.modified != datetime.fromtimestamp(before_mtime, tz=UTC)
