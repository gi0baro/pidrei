"""Mirror of pi coding-agent test/agent-session-runtime-diagnostics.test.ts.

pi's faux provider becomes a built-in model on an in-memory auth storage, as
in `test_agent_session_runtime.py`; no request is sent.
"""

import os

import pytest

from pidrei.core.agent_session_runtime import CreateAgentSessionRuntimeResult, create_agent_session_runtime
from pidrei.core.agent_session_services import (
    AgentSessionRuntimeDiagnostic,
    AgentSessionServices,
    CreateAgentSessionFromServicesOptions,
    create_agent_session_from_services,
)
from pidrei.core.auth_storage import AuthStorage
from pidrei.core.model_runtime import ModelRuntime
from pidrei.core.session_manager import SessionManager
from pidrei.core.settings_manager import SettingsManager
from pidrei_ai.auth.types import ApiKeyCredential
from pidrei_ai.providers.all import get_builtin_model

from .agent_session_helpers import create_test_resource_loader


async def _create_runtime_for_test(temp_dir: str, next_diagnostics: dict):
    """A runtime whose factory returns whatever `next_diagnostics["current"]` holds when it runs."""
    auth_storage = AuthStorage.in_memory()

    async def set_key(_credential):
        return ApiKeyCredential(key="test-key")

    await auth_storage.modify("anthropic", set_key)
    model_runtime = await ModelRuntime(
        credentials=auth_storage,
        models_path=os.path.join(temp_dir, "models.json"),
        allow_model_network=False,
    )
    model = get_builtin_model("anthropic", "claude-sonnet-4-5")

    async def create_runtime(*, cwd, agent_dir, session_manager, session_start_event=None, **_kwargs):
        services = AgentSessionServices(
            cwd=cwd,
            agent_dir=agent_dir,
            model_runtime=model_runtime,
            settings_manager=await SettingsManager(cwd, agent_dir),
            resource_loader=create_test_resource_loader(),
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
            session=result.session,
            services=services,
            extensions_result=result.extensions_result,
            model_fallback_message=result.model_fallback_message,
            diagnostics=next_diagnostics["current"],
        )

    return await create_agent_session_runtime(
        create_runtime,
        cwd=temp_dir,
        agent_dir=temp_dir,
        session_manager=SessionManager.in_memory(temp_dir),
    )


class TestTakeUnreportedDiagnostics:
    # The runtime factory runs again on every session replacement, but modes only reported the startup
    # diagnostics, so a problem that first appeared on /new or /resume was never shown. Found while fixing #11264.
    @pytest.mark.tonio
    async def test_returns_each_diagnostic_once_at_startup_and_after_replacements(self, tmp_path):
        settings_warning = AgentSessionRuntimeDiagnostic(type="warning", message="settings warning")
        extension_error = AgentSessionRuntimeDiagnostic(type="error", message="extension failed")
        next_diagnostics = {"current": [settings_warning]}
        runtime = await _create_runtime_for_test(str(tmp_path), next_diagnostics)

        assert runtime.take_unreported_diagnostics() == [settings_warning]
        assert runtime.take_unreported_diagnostics() == []

        next_diagnostics["current"] = [settings_warning, extension_error]
        await runtime.new_session()
        assert runtime.take_unreported_diagnostics() == [extension_error]
        assert runtime.take_unreported_diagnostics() == []

        await runtime.new_session()
        assert runtime.take_unreported_diagnostics() == []
        await runtime.dispose()
