"""Mirror of pi coding-agent test/runtime-credentials.test.ts."""

import time

import pytest

from pidrei.core.auth_storage import AuthStorage
from pidrei.core.runtime_credentials import RuntimeCredentials
from pidrei_ai.auth.types import ApiKeyCredential, AuthOperationOptions, CredentialInfo, OAuthCredential
from pidrei_ai.utils.cancel import AbortError, CancelToken


@pytest.mark.tonio
async def test_forwards_operation_options_to_the_persistent_store():
    controller = CancelToken()
    received: list = []

    class RecordingStore:
        async def read(self, _provider_id, options=None):
            received.append(options.cancel if options is not None else None)

        async def list(self, options=None):
            received.append(options.cancel if options is not None else None)
            return []

        async def modify(self, _provider_id, _fn, options=None):
            received.append(options.cancel if options is not None else None)

        async def delete(self, _provider_id, options=None):
            received.append(options.cancel if options is not None else None)

    credentials = RuntimeCredentials(RecordingStore())
    options = AuthOperationOptions(cancel=controller)

    async def keep(_current):
        return None

    await credentials.read("anthropic", options)
    await credentials.list(options)
    await credentials.modify("anthropic", keep, options)
    await credentials.delete("anthropic", options)

    assert received == [controller, controller, controller, controller]


@pytest.mark.tonio
async def test_keeps_a_runtime_override_when_persistent_deletion_is_cancelled():
    aborted = AbortError("cancelled")
    storage = AuthStorage.in_memory({"anthropic": ApiKeyCredential(key="stored-key")})
    delete_calls = {"count": 0}

    async def failing_delete(_provider_id, _options=None):
        delete_calls["count"] += 1
        raise aborted

    storage.delete = failing_delete
    credentials = RuntimeCredentials(storage)
    credentials.set_runtime_api_key("anthropic", "runtime-key")

    with pytest.raises(AbortError):
        await credentials.delete("anthropic", AuthOperationOptions(cancel=CancelToken()))
    assert delete_calls["count"] == 1
    assert await credentials.read("anthropic") == ApiKeyCredential(key="runtime-key")


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
