"""Mirror of pi coding-agent test/runtime-credentials.test.ts."""

import time

import pytest

from pidrei.core.auth_storage import AuthStorage
from pidrei.core.runtime_credentials import RuntimeCredentials
from pidrei_ai.auth.types import ApiKeyCredential, CredentialInfo, OAuthCredential


@pytest.mark.tonio
async def test_runtime_overrides_mask_stored_credentials_without_persisting():
    storage = AuthStorage.in_memory({"anthropic": ApiKeyCredential(key="stored-key")})
    credentials = RuntimeCredentials(storage)

    credentials.set_runtime_api_key("anthropic", "runtime-key")
    assert await credentials.read("anthropic") == ApiKeyCredential(key="runtime-key")
    assert await storage.read("anthropic") == ApiKeyCredential(key="stored-key")

    credentials.remove_runtime_api_key("anthropic")
    assert await credentials.read("anthropic") == ApiKeyCredential(key="stored-key")


@pytest.mark.tonio
async def test_enumeration_merges_overrides_without_exposing_keys():
    storage = AuthStorage.in_memory(
        {"anthropic": OAuthCredential(access="access", refresh="refresh", expires=int(time.time() * 1000) + 60_000)}
    )
    credentials = RuntimeCredentials(storage)
    credentials.set_runtime_api_key("anthropic", "runtime-key")
    credentials.set_runtime_api_key("openai", "other-runtime-key")

    assert await credentials.list() == [
        CredentialInfo(provider_id="anthropic", type="api_key"),
        CredentialInfo(provider_id="openai", type="api_key"),
    ]


@pytest.mark.tonio
async def test_delete_clears_both_the_override_and_persisted_credential():
    storage = AuthStorage.in_memory({"anthropic": ApiKeyCredential(key="stored-key")})
    credentials = RuntimeCredentials(storage)
    credentials.set_runtime_api_key("anthropic", "runtime-key")

    await credentials.delete("anthropic")

    assert await credentials.read("anthropic") is None
    assert await credentials.list() == []
