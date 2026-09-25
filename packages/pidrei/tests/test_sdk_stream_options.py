"""Partial mirror of pi coding-agent test/sdk-stream-options.test.ts: the
cache-warming scheduling cases (0.87.1). The stream-option forwarding cases
are not mirrored here.
"""

import dataclasses
import os

import pytest

from pidrei.core.auth_storage import AuthStorage
from pidrei.core.cache_warmer import CacheWarmingStatus
from pidrei.core.model_runtime import ModelRuntime
from pidrei.core.sdk import CreateAgentSessionOptions, create_agent_session
from pidrei.core.session_manager import SessionManager
from pidrei.core.settings_manager import SettingsManager
from pidrei_ai.auth.types import ApiKeyCredential
from pidrei_ai.types import Model, ModelCost, Usage, UsageCost, UserMessage
from pidrei_ai.utils.event_stream import AssistantMessageEventStream

from .agent_session_helpers import create_assistant_message, create_test_resource_loader, push_done
from .coding_session_helpers import now_ms


def _create_model() -> Model:
    return Model(
        id="capture-model",
        name="Capture Model",
        api="anthropic-messages",
        provider="capture-provider",
        base_url="https://capture.invalid/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=10, output=50, cache_read=0.25, cache_write=12.5),
        context_window=128000,
        max_tokens=4096,
        prompt_cache={"short": 300},
    )


def _create_done_message(model: Model, prompt_tokens: int = 0):
    return create_assistant_message(
        "ok",
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(cache_read=prompt_tokens, total_tokens=prompt_tokens, cost=UsageCost()),
    )


class _Fixture:
    def __init__(self) -> None:
        self.session = None
        self.model_runtime: ModelRuntime | None = None
        self.provider_calls = 0

    def stream_simple(self, request_model, _context, _options=None):
        self.provider_calls += 1
        stream = AssistantMessageEventStream()
        push_done(stream, _create_done_message(request_model, 100_000))
        return stream

    def dispose(self) -> None:
        self.session.dispose()
        self.model_runtime.unregister_provider("capture-provider")


async def _create_cache_warming_session(tmp_path, populate=None) -> _Fixture:
    cwd = os.path.join(str(tmp_path), "project")
    agent_dir = os.path.join(str(tmp_path), "agent")
    os.makedirs(cwd)
    os.makedirs(agent_dir)
    model = _create_model()
    fixture = _Fixture()

    auth_storage = AuthStorage.in_memory()

    async def set_key(_credential):
        return ApiKeyCredential(key="test-api-key")

    await auth_storage.modify(model.provider, set_key)
    fixture.model_runtime = await ModelRuntime.create(
        credentials=auth_storage,
        models_path=os.path.join(agent_dir, "models.json"),
        allow_model_network=False,
    )
    fixture.model_runtime.register_provider(model.provider, {"api": model.api, "streamSimple": fixture.stream_simple})
    session_manager = SessionManager.in_memory(cwd)
    if populate is not None:
        await populate(session_manager, model)
    result = await create_agent_session(
        CreateAgentSessionOptions(
            cwd=cwd,
            agent_dir=agent_dir,
            model=model,
            model_runtime=fixture.model_runtime,
            settings_manager=SettingsManager.in_memory({"cacheWarming": "idle"}),
            session_manager=session_manager,
            resource_loader=create_test_resource_loader(),
        )
    )
    fixture.session = result.session
    return fixture


class TestCacheWarmingScheduling:
    @pytest.mark.tonio
    async def test_schedules_cache_warming_after_a_completed_session_request(self, tmp_path):
        fixture = await _create_cache_warming_session(tmp_path)
        try:
            await fixture.session.prompt("test")
            status = fixture.session.cache_warming_status
            assert status is not None
            assert status.next_warm_at is not None
            assert status.next_warm_at > now_ms()

            # Equivalent shallow copies remain current, but removing the request prefix does not.
            state = fixture.session.agent.state
            state.messages = list(state.messages)
            state.model = dataclasses.replace(state.model)
            status = fixture.session.cache_warming_status
            assert status is not None
            assert status.next_warm_at is not None
            assert status.next_warm_at > now_ms()
            state.messages = state.messages[1:]
            status = fixture.session.cache_warming_status
            assert status is not None
            assert status.reason == "conversation context changed"
        finally:
            fixture.dispose()

    @pytest.mark.tonio
    async def test_waits_for_the_next_request_instead_of_restoring_cache_warming(self, tmp_path):
        async def populate(manager: SessionManager, model: Model) -> None:
            await manager.append_model_change(model.provider, model.id)
            await manager.append_thinking_level_change("off")
            await manager.append_message(UserMessage(content="test", timestamp=now_ms() - 60_000))
            assistant = dataclasses.replace(_create_done_message(model, 100_000), timestamp=now_ms() - 59_000)
            await manager.append_message(assistant)
            await manager.append_usage("cache_warm", model.provider, model.id, assistant.usage)

        fixture = await _create_cache_warming_session(tmp_path, populate)
        try:
            assert fixture.provider_calls == 0
            assert fixture.session.cache_warming_status == CacheWarmingStatus(
                state="inactive", reason="waiting for first request"
            )
        finally:
            fixture.dispose()
