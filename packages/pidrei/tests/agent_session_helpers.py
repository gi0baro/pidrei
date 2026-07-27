"""Shared helpers for AgentSession test mirrors (pi test/utilities.ts subset).

pi's e2e suites hit real LLMs behind describe.skipIf(!API_KEY); pidrei mirrors
stay hermetic by driving the same code paths through mock stream functions.
"""

import os
from collections.abc import Callable
from typing import Any

import tonio.colored as tonio

from pidrei.core.agent_session import AgentSession, AgentSessionConfig
from pidrei.core.auth_storage import AuthStorage
from pidrei.core.extensions import ExtensionRuntime, LoadExtensionsResult
from pidrei.core.model_registry import ModelRegistry
from pidrei.core.model_runtime import ModelRuntime
from pidrei.core.session_manager import SessionManager
from pidrei.core.settings_manager import SettingsManager
from pidrei_agent.agent import Agent, AgentInitialState
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import AssistantMessage, DoneEvent, ErrorEvent, StartEvent, TextContent, Usage, UsageCost
from pidrei_ai.utils.event_stream import AssistantMessageEventStream

from .coding_session_helpers import now_ms


class StubResourceLoader:
    """pi test/utilities.ts createTestResourceLoader."""

    def __init__(self, extensions_result: LoadExtensionsResult | None = None):
        self._extensions_result = (
            extensions_result if extensions_result is not None else LoadExtensionsResult(runtime=ExtensionRuntime())
        )

    def get_extensions(self):
        return self._extensions_result

    def get_skills(self):
        from pidrei.core.skills import LoadSkillsResult

        return LoadSkillsResult(skills=[], diagnostics=[])

    def get_prompts(self):
        from pidrei.core.resource_loader import LoadPromptsResult

        return LoadPromptsResult(prompts=[], diagnostics=[])

    def get_agents_files(self):
        return []

    def get_system_prompt(self):
        return None

    def get_append_system_prompt(self):
        return []

    def extend_resources(self, *_args, **_kwargs):
        pass

    async def reload(self, **_kwargs):
        pass


def create_test_resource_loader(extensions_result: LoadExtensionsResult | None = None) -> StubResourceLoader:
    return StubResourceLoader(extensions_result)


def create_assistant_message(text: str, **overrides: Any) -> AssistantMessage:
    defaults: dict[str, Any] = {
        "content": [TextContent(text=text)],
        "api": "anthropic-messages",
        "provider": "anthropic",
        "model": "mock",
        "usage": Usage(input=0, output=0, cache_read=0, cache_write=0, total_tokens=0, cost=UsageCost()),
        "stop_reason": "stop",
        "timestamp": now_ms(),
    }
    defaults.update(overrides)
    return AssistantMessage(**defaults)


def push_done(stream: AssistantMessageEventStream, message: AssistantMessage, reason: str = "stop") -> None:
    stream.push(StartEvent(partial=create_assistant_message("")))
    stream.push(DoneEvent(reason=reason, message=message))


def push_error(stream: AssistantMessageEventStream, message: AssistantMessage, reason: str = "error") -> None:
    stream.push(StartEvent(partial=message))
    stream.push(ErrorEvent(reason=reason, error=message))


async def abortable_stream_fn(_model, _context, options=None) -> AssistantMessageEventStream:
    """Mock stream that responds only to abort (pi's MockAssistantStream + checkAbort)."""
    cancel = getattr(options, "cancel", None)
    stream = AssistantMessageEventStream()
    stream.push(StartEvent(partial=create_assistant_message("")))

    async def watch_abort() -> None:
        while cancel is None or not cancel.cancelled:
            await tonio.time.sleep(0.005)
        stream.push(ErrorEvent(reason="aborted", error=create_assistant_message("Aborted")))

    tonio.spawn.without_tracking(watch_abort())
    return stream


async def create_agent_session(
    temp_dir: str,
    *,
    stream_fn: Callable,
    tools: list | None = None,
    base_tools_override: dict | None = None,
    in_memory_session: bool = True,
    settings_overrides: dict | None = None,
    system_prompt: str = "Test",
    resource_loader: Any = None,
    session_manager: SessionManager | None = None,
    model: Any = None,
    provider_auth: str | None = "anthropic",
) -> AgentSession:
    """AgentSession over an in-memory session with a stubbed provider auth."""
    temp_dir = str(temp_dir)
    if model is None:
        model = get_builtin_model("anthropic", "claude-sonnet-4-5")
    agent = Agent(
        stream_fn=stream_fn,
        get_api_key=lambda _provider: "test-key",
        initial_state=AgentInitialState(model=model, system_prompt=system_prompt, tools=tools or []),
    )

    if session_manager is None:
        session_manager = (
            SessionManager.in_memory() if in_memory_session else await SessionManager.create(temp_dir, temp_dir)
        )
    settings_manager = await SettingsManager.create(temp_dir, temp_dir)
    if settings_overrides:
        settings_manager.apply_overrides(settings_overrides)
    auth_storage = await AuthStorage.create(os.path.join(temp_dir, "auth.json"))

    from pidrei_ai.auth.types import ApiKeyCredential

    async def set_key(_credential):
        return ApiKeyCredential(key="test-key")

    if provider_auth is not None:
        await auth_storage.modify(provider_auth, set_key)
    model_runtime = await ModelRuntime.create(
        credentials=auth_storage, models_path=os.path.join(temp_dir, "models.json"), allow_model_network=False
    )

    return AgentSession(
        AgentSessionConfig(
            agent=agent,
            session_manager=session_manager,
            settings_manager=settings_manager,
            cwd=temp_dir,
            model_runtime=model_runtime,
            resource_loader=resource_loader if resource_loader is not None else create_test_resource_loader(),
            base_tools_override=base_tools_override,
        )
    )


def get_model_runtime(model_registry: ModelRegistry) -> ModelRuntime:
    return model_registry._runtime
