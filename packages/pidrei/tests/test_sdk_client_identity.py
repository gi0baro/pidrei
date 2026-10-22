"""Mirror of pi coding-agent test/sdk-client-identity.test.ts.

pi gates attribution on install telemetry (`setEnableInstallTelemetry`,
`PI_TELEMETRY`); pidrei's gate is the `enableProviderAttribution` setting and
`PIDREI_PROVIDER_ATTRIBUTION`. The expected values are pidrei's: a value
silently reverting to pi's attribution is the regression worth catching.
"""

import os

import pytest

from pidrei.config import VERSION
from pidrei.core.auth_storage import AuthStorage
from pidrei.core.client_identity import ATTRIBUTION_ENV
from pidrei.core.model_runtime import ModelRuntime
from pidrei.core.sdk import CreateAgentSessionOptions, create_agent_session
from pidrei.core.session_manager import SessionManager
from pidrei.core.settings_manager import SettingsManager
from pidrei_ai.auth.types import ApiKeyCredential
from pidrei_ai.types import Context, Model, ModelCost, SimpleStreamOptions
from pidrei_ai.utils.event_stream import AssistantMessageEventStream
from pidrei_ai.utils.identity import PIDREI_ATTRIBUTION, get_attribution_headers
from pidrei_ai.utils.transcript import normalize_context

from .agent_session_helpers import create_assistant_message, create_test_resource_loader, push_done


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    monkeypatch.delenv(ATTRIBUTION_ENV, raising=False)
    cwd = os.path.join(str(tmp_path), "project")
    agent_dir = os.path.join(str(tmp_path), "agent")
    os.makedirs(cwd)
    os.makedirs(agent_dir)
    return cwd, agent_dir


def create_model(provider: str, base_url: str) -> Model:
    return Model(
        id=f"{provider}-test-model",
        name=f"{provider} Test Model",
        api="openai-completions",
        provider=provider,
        base_url=base_url,
        reasoning=False,
        input=["text"],
        cost=ModelCost(),
        context_window=128000,
        max_tokens=4096,
    )


async def capture_options(
    dirs,
    model: Model,
    *,
    attribution_enabled: bool = True,
    provider_headers: dict[str, str] | None = None,
    session_id: str | None = None,
) -> SimpleStreamOptions | None:
    cwd, agent_dir = dirs
    settings_manager = SettingsManager.in_memory()
    if not attribution_enabled:
        settings_manager.set_enable_provider_attribution(False)

    auth_storage = AuthStorage.in_memory()

    async def set_key(_credential):
        return ApiKeyCredential(key="test-api-key")

    await auth_storage.modify(model.provider, set_key)
    model_runtime = await ModelRuntime(
        credentials=auth_storage,
        models_path=os.path.join(agent_dir, "models.json"),
        allow_model_network=False,
    )
    captured: list[SimpleStreamOptions | None] = []

    def stream_simple(request_model, _context, provider_options=None):
        captured.append(provider_options)
        stream = AssistantMessageEventStream()
        push_done(stream, create_assistant_message("ok", api=request_model.api, provider=request_model.provider))
        return stream

    model_runtime.register_provider(
        model.provider,
        {
            "api": model.api,
            **({"headers": provider_headers} if provider_headers else {}),
            "streamSimple": stream_simple,
        },
    )
    session_manager = SessionManager.in_memory(cwd)
    if session_id:
        session_manager.new_session({"id": session_id})

    result = await create_agent_session(
        CreateAgentSessionOptions(
            cwd=cwd,
            agent_dir=agent_dir,
            model=model,
            model_runtime=model_runtime,
            settings_manager=settings_manager,
            session_manager=session_manager,
            resource_loader=create_test_resource_loader(),
        )
    )
    session = result.session
    try:
        stream = await session.agent.stream_function(
            model, normalize_context(Context(messages=[])), SimpleStreamOptions(session_id=session.session_id)
        )
        await stream.result()
        return captured[0] if captured else None
    finally:
        session.dispose()
        model_runtime.unregister_provider(model.provider)


@pytest.mark.tonio
async def test_identifies_pidrei_with_a_detailed_user_agent_and_attribution_as_a_cli_agent(dirs):
    model = create_model("openrouter", "https://openrouter.ai/api/v1")
    options = await capture_options(dirs, model)

    assert options.identity == {"user_agent": "detailed", "version": VERSION, "attribution": PIDREI_ATTRIBUTION}
    assert get_attribution_headers(model, options.identity) == {
        "HTTP-Referer": "https://github.com/gi0baro/pidrei",
        "X-OpenRouter-Title": "pidrei",
        "X-OpenRouter-Categories": "cli-agent",
    }


@pytest.mark.tonio
async def test_turns_attribution_off_when_the_setting_is_disabled(dirs):
    model = create_model("nvidia", "https://integrate.api.nvidia.com/v1")
    options = await capture_options(dirs, model, attribution_enabled=False)

    assert options.identity == {"user_agent": "detailed", "version": VERSION, "attribution": None}
    assert get_attribution_headers(model, options.identity) == {}


@pytest.mark.tonio
async def test_turns_attribution_off_with_pidrei_provider_attribution_0(dirs, monkeypatch):
    monkeypatch.setenv(ATTRIBUTION_ENV, "0")
    options = await capture_options(dirs, create_model("openrouter", "https://openrouter.ai/api/v1"))

    assert options.identity["attribution"] is None


@pytest.mark.tonio
async def test_adds_opencode_session_headers_for_custom_opencode_providers(dirs):
    options = await capture_options(
        dirs, create_model("custom-opencode", "https://opencode.ai/zen/v1"), session_id="opencode-session"
    )

    assert options.headers["x-opencode-session"] == "opencode-session"


@pytest.mark.tonio
async def test_lets_configured_opencode_headers_override_the_session_header(dirs):
    options = await capture_options(
        dirs,
        create_model("opencode", "https://opencode.ai/zen/v1"),
        session_id="opencode-session",
        provider_headers={"x-opencode-session": "configured-session"},
    )

    assert options.headers["x-opencode-session"] == "configured-session"
