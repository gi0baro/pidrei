"""Mirror of pi coding-agent test/suite/regressions/11520-sdk-summary-oauth-auth.test.ts.

Issue #11520: summaries in SDK sessions passed the resolved OAuth token back to
the ModelRuntime as an explicit API key, so provider auth took the API-key
override branch.
"""

import time

import pytest

from pidrei.core.auth_storage import AuthStorage
from pidrei.core.model_runtime import ModelRuntime
from pidrei.core.provider_composer import ExtensionOAuthConfig
from pidrei.core.sdk import CreateAgentSessionOptions, create_agent_session
from pidrei.core.session_manager import SessionManager
from pidrei.core.settings_manager import SettingsManager
from pidrei_ai.auth.types import OAuthCredential
from pidrei_ai.types import Model, ModelCost
from pidrei_ai.utils.event_stream import AssistantMessageEventStream

from .agent_session_helpers import create_assistant_message, create_test_resource_loader, push_done
from .coding_session_helpers import assistant_msg, user_msg


@pytest.mark.tonio
async def test_lets_the_runtime_resolve_stored_oauth_for_branch_summaries(tmp_path, monkeypatch):
    cwd = str(tmp_path)
    expires = int(time.time() * 1000) + 3_600_000
    credentials = AuthStorage.in_memory(
        {"oauth-only-provider": OAuthCredential(access="oauth-access", refresh="refresh", expires=expires)}
    )
    model_runtime = await ModelRuntime(credentials=credentials, models_path=None, allow_model_network=False)
    model = Model(
        id="oauth-model",
        name="OAuth Model",
        api="openai-completions",
        provider="oauth-only-provider",
        base_url="https://example.test/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(),
        context_window=128000,
        max_tokens=4096,
    )
    requests: list = []

    def stream_simple(request_model, _context, options=None):
        requests.append(options)
        stream = AssistantMessageEventStream()
        push_done(
            stream,
            create_assistant_message(
                "branch summary text", api=request_model.api, provider=request_model.provider, model=request_model.id
            ),
        )
        return stream

    async def login(_callbacks):
        return {"access": "oauth-access", "refresh": "refresh", "expires": int(time.time() * 1000) + 60_000}

    async def refresh_token(credential, _cancel=None):
        return credential

    model_runtime.register_provider(
        model.provider,
        {
            "name": "OAuth Only",
            "baseUrl": model.base_url,
            "api": model.api,
            # With an API key method as well, an explicit key would bypass the stored OAuth login.
            "apiKey": "UNUSED_ENV_KEY",
            "oauth": ExtensionOAuthConfig(
                name="Subscription",
                login=login,
                refresh_token=refresh_token,
                get_api_key=lambda credential: credential.access,
            ),
            "models": [
                {
                    "id": model.id,
                    "name": model.name,
                    "reasoning": False,
                    "input": ["text"],
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    "contextWindow": 128000,
                    "maxTokens": 4096,
                }
            ],
            "streamSimple": stream_simple,
        },
    )
    # pi wraps `getAuth`, which its `streamSimple` also calls; here both `get_auth(model)` and the
    # stream's request preparation go through `_get_model_auth`.
    auth_overrides: list = []
    get_model_auth = model_runtime._get_model_auth

    async def recording_get_model_auth(provider, request_model, overrides):
        auth_overrides.append(overrides.api_key)
        return await get_model_auth(provider, request_model, overrides)

    monkeypatch.setattr(model_runtime, "_get_model_auth", recording_get_model_auth)

    session_manager = SessionManager.in_memory(cwd)
    result = await create_agent_session(
        CreateAgentSessionOptions(
            cwd=cwd,
            agent_dir=cwd,
            model=model,
            model_runtime=model_runtime,
            settings_manager=SettingsManager.in_memory(),
            session_manager=session_manager,
            resource_loader=create_test_resource_loader(),
        )
    )
    session = result.session
    try:
        target_id = await session_manager.append_message(user_msg("first branch"))
        await session_manager.append_message(assistant_msg("first reply"))
        await session_manager.append_message(user_msg("abandoned branch work"))
        await session_manager.append_message(assistant_msg("abandoned reply"))

        navigated = await session.navigate_tree(target_id, {"summarize": True})

        assert "branch summary text" in navigated.summary_entry["summary"]
        assert len(requests) == 1
        assert requests[0].api_key == "oauth-access"
        assert len(auth_overrides) > 0
        assert all(api_key is None for api_key in auth_overrides)
    finally:
        session.dispose()
