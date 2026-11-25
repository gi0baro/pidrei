"""Mirror of pi coding-agent test/suite/regressions/12164-stored-command-key-auth-status.test.ts."""

import pytest

from pidrei.core.provider_composer import AuthStatus
from pidrei.core.resolve_config_value import clear_config_value_cache
from pidrei_ai.auth.types import ApiKeyCredential
from pidrei_ai.registry import ModelsRefreshOptions

from .harness import create_harness


# Providers without an API key of their own, so only their stored credentials configure them.
PROVIDERS_WITHOUT_KEYS = {
    "providers": {
        provider: {"baseUrl": "https://example.test/v1", "api": "openai-completions", "models": [{"id": "model"}]}
        for provider in ["failing-command", "working-command"]
    }
}


@pytest.mark.tonio
async def test_reports_a_stored_key_whose_command_fails_as_stored_but_not_configured():
    harness = await create_harness(with_configured_auth=False, models_json=PROVIDERS_WITHOUT_KEYS)
    try:
        runtime = harness.session.model_runtime

        async def failing_command(_current):
            return ApiKeyCredential(key="!exit 1")

        async def working_command(_current):
            return ApiKeyCredential(key="!echo key")

        await harness.auth_storage.modify("failing-command", failing_command)
        await harness.auth_storage.modify("working-command", working_command)
        await runtime.refresh(ModelsRefreshOptions(allow_network=False))

        assert runtime.get_provider_auth_status("failing-command") == AuthStatus(configured=False, source="stored")
        assert runtime.has_configured_auth("failing-command") is False
        assert runtime.get_provider_auth_status("working-command") == AuthStatus(configured=True, source="stored")
        assert runtime.has_configured_auth("working-command") is True
    finally:
        # The command results are cached for the process.
        clear_config_value_cache()
        harness.cleanup()
