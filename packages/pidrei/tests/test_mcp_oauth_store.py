"""Mirror of pi's mcp-oauth-store.test.ts.

`tokens()` and `remove()` are awaited here: the store reads the credential
file, and pidrei's file backend takes its lock asynchronously.
"""

import json

import pytest

from pidrei.core.auth_storage import InMemoryAuthStorageBackend
from pidrei.extensions.mcp.oauth import McpOAuthCredentialStore


SERVER_URL = "https://mcp.example.com/mcp"


def _state(access_token: str) -> dict:
    return {"serverUrl": SERVER_URL, "tokens": {"access_token": access_token, "token_type": "Bearer"}}


def _stored_keys(backend: InMemoryAuthStorageBackend) -> list[str]:
    return list(json.loads(backend.with_lock(lambda current: (current or "{}", None))))


def _store_legacy(backend: InMemoryAuthStorageBackend) -> None:
    backend.with_lock(lambda _current: (None, json.dumps({SERVER_URL: _state("legacy-token")})))


@pytest.mark.tonio
async def test_keeps_separate_credentials_for_servers_sharing_a_server_url():
    # https://github.com/earendil-works/pi/issues/10252
    store = McpOAuthCredentialStore(InMemoryAuthStorageBackend())
    await store.for_server("work", SERVER_URL).save(_state("work-token"))
    await store.for_server("personal", SERVER_URL).save(_state("personal-token"))

    assert (await store.for_server("work", SERVER_URL).load())["tokens"]["access_token"] == "work-token"
    assert (await store.for_server("personal", SERVER_URL).load())["tokens"]["access_token"] == "personal-token"

    assert await store.remove("work", SERVER_URL) is True
    assert await store.for_server("work", SERVER_URL).load() is None
    assert (await store.tokens("personal", SERVER_URL))["access_token"] == "personal-token"


@pytest.mark.tonio
async def test_moves_credentials_stored_by_server_url_to_the_first_server_that_loads_them():
    backend = InMemoryAuthStorageBackend()
    _store_legacy(backend)
    store = McpOAuthCredentialStore(backend)

    # Reading tokens does not take the legacy state over.
    assert (await store.tokens("work", SERVER_URL))["access_token"] == "legacy-token"
    assert _stored_keys(backend) == [SERVER_URL]

    assert (await store.for_server("my_work", SERVER_URL).load())["tokens"]["access_token"] == "legacy-token"
    # Names differing only in `-` and `_` are the same server.
    assert (await store.tokens("my-work", SERVER_URL))["access_token"] == "legacy-token"
    assert await store.for_server("personal", SERVER_URL).load() is None
    assert _stored_keys(backend) == [f"mcp__my_work|{SERVER_URL}"]


@pytest.mark.tonio
async def test_signs_out_of_credentials_stored_by_server_url():
    backend = InMemoryAuthStorageBackend()
    _store_legacy(backend)
    store = McpOAuthCredentialStore(backend)

    assert await store.remove("work", SERVER_URL) is True
    assert _stored_keys(backend) == []
    assert await store.remove("work", SERVER_URL) is False
