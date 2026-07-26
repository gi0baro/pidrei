"""Mirror of pi coding-agent test/auth-storage.test.ts.

Adaptations:
- pi's proper-lockfile "compromised lock" test targets that library's mtime
  watchdog, which is not ported — skipped.
- The lock-failure test patches this module's async lock acquisition seam
  instead of vitest-spying on lockfile.lock.
"""

import contextlib
import json
import time

import pytest
import tonio.colored as tonio

import pidrei.core.auth_storage as auth_storage_module
from pidrei.core.auth_storage import AuthStorage
from pidrei_ai.auth.types import ApiKeyCredential, CredentialInfo, OAuthCredential


def write_auth_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def read_auth_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@contextlib.contextmanager
def env_var(name, value):
    import os

    original = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if original is None:
            del os.environ[name]
        else:
            os.environ[name] = original


@pytest.mark.tonio
async def test_reads_and_resolves_stored_api_key_credentials(tmp_dir):
    auth_path = tmp_dir / "auth.json"
    with env_var("TEST_AUTH_STORAGE_KEY", "environment-key"):
        write_auth_json(auth_path, {"anthropic": {"type": "api_key", "key": "$TEST_AUTH_STORAGE_KEY"}})
        storage = AuthStorage.create(str(auth_path))
        assert await storage.read("anthropic") == ApiKeyCredential(key="environment-key")


@pytest.mark.tonio
async def test_resolves_command_backed_api_key_credentials(tmp_dir):
    auth_path = tmp_dir / "auth.json"
    write_auth_json(auth_path, {"anthropic": {"type": "api_key", "key": "!printf 'command-key'"}})
    storage = AuthStorage.create(str(auth_path))
    assert await storage.read("anthropic") == ApiKeyCredential(key="command-key")


@pytest.mark.tonio
async def test_returns_oauth_credentials_unchanged():
    credential = OAuthCredential(
        access="access-token", refresh="refresh-token", expires=int(time.time() * 1000) + 60_000
    )
    storage = AuthStorage.in_memory({"anthropic": credential})
    assert await storage.read("anthropic") == credential


@pytest.mark.tonio
async def test_credential_scoped_env_takes_precedence_and_remains_inspectable(tmp_dir):
    auth_path = tmp_dir / "auth.json"
    write_auth_json(
        auth_path,
        {
            "anthropic": {
                "type": "api_key",
                "key": "$SCOPED_KEY",
                "env": {"SCOPED_KEY": "scoped-value", "REGION": "test-region"},
            }
        },
    )
    storage = AuthStorage.create(str(auth_path))
    resolved = await storage.read("anthropic")
    assert resolved.key == "scoped-value"
    assert resolved.env == {"SCOPED_KEY": "scoped-value", "REGION": "test-region"}


@pytest.mark.tonio
async def test_modify_persists_a_credential_while_preserving_unrelated_external_edits(tmp_dir):
    auth_path = tmp_dir / "auth.json"
    write_auth_json(auth_path, {"anthropic": {"type": "api_key", "key": "old"}})
    storage = AuthStorage.create(str(auth_path))
    write_auth_json(
        auth_path,
        {"anthropic": {"type": "api_key", "key": "old"}, "openai": {"type": "api_key", "key": "external"}},
    )

    async def update(_current):
        return ApiKeyCredential(key="new")

    await storage.modify("anthropic", update)

    assert read_auth_json(auth_path) == {
        "anthropic": {"type": "api_key", "key": "new"},
        "openai": {"type": "api_key", "key": "external"},
    }


@pytest.mark.tonio
async def test_modify_with_none_leaves_the_current_credential_unchanged(tmp_dir):
    auth_path = tmp_dir / "auth.json"
    write_auth_json(auth_path, {"anthropic": {"type": "api_key", "key": "stored"}})
    storage = AuthStorage.create(str(auth_path))

    async def keep(_current):
        return None

    assert await storage.modify("anthropic", keep) == ApiKeyCredential(key="stored")
    assert await storage.read("anthropic") == ApiKeyCredential(key="stored")


@pytest.mark.tonio
async def test_serializes_concurrent_modifications(tmp_dir):
    auth_path = tmp_dir / "auth.json"
    write_auth_json(auth_path, {})
    first = AuthStorage.create(str(auth_path))
    second = AuthStorage.create(str(auth_path))

    async def set_anthropic(_current):
        return ApiKeyCredential(key="anthropic-key")

    async def set_openai(_current):
        return ApiKeyCredential(key="openai-key")

    await tonio.spawn(first.modify("anthropic", set_anthropic), second.modify("openai", set_openai))

    assert read_auth_json(auth_path) == {
        "anthropic": {"type": "api_key", "key": "anthropic-key"},
        "openai": {"type": "api_key", "key": "openai-key"},
    }


@pytest.mark.tonio
async def test_delete_removes_one_credential_while_preserving_others(tmp_dir):
    auth_path = tmp_dir / "auth.json"
    write_auth_json(
        auth_path,
        {"anthropic": {"type": "api_key", "key": "anthropic-key"}, "openai": {"type": "api_key", "key": "openai-key"}},
    )
    storage = AuthStorage.create(str(auth_path))
    write_auth_json(
        auth_path,
        {
            "anthropic": {"type": "api_key", "key": "anthropic-key"},
            "openai": {"type": "api_key", "key": "openai-key"},
            "google": {"type": "api_key", "key": "external-key"},
        },
    )
    await storage.delete("anthropic")
    assert await storage.list() == [
        CredentialInfo(provider_id="openai", type="api_key"),
        CredentialInfo(provider_id="google", type="api_key"),
    ]
    assert await storage.read("anthropic") is None
    assert await storage.read("openai") == ApiKeyCredential(key="openai-key")
    assert await storage.read("google") == ApiKeyCredential(key="external-key")


@pytest.mark.tonio
async def test_in_memory_storage_implements_the_same_credential_store_behavior():
    storage = AuthStorage.in_memory({"anthropic": ApiKeyCredential(key="initial")})
    assert await storage.read("anthropic") == ApiKeyCredential(key="initial")

    async def update(_current):
        return ApiKeyCredential(key="updated")

    await storage.modify("anthropic", update)
    assert await storage.read("anthropic") == ApiKeyCredential(key="updated")
    await storage.delete("anthropic")
    assert await storage.list() == []


@pytest.mark.tonio
async def test_does_not_write_after_lock_acquisition_failure_and_recovers_on_retry(tmp_dir):
    auth_path = tmp_dir / "auth.json"
    write_auth_json(auth_path, {"anthropic": {"type": "api_key", "key": "stored"}})
    storage = AuthStorage.create(str(auth_path))

    original_acquire = auth_storage_module._acquire_lock_async
    calls = {"count": 0}

    async def failing_acquire(path):
        calls["count"] += 1
        raise Exception("lock unavailable")

    auth_storage_module._acquire_lock_async = failing_acquire

    async def set_openai(_current):
        return ApiKeyCredential(key="new")

    try:
        with pytest.raises(Exception, match="lock unavailable"):
            await storage.modify("openai", set_openai)
        assert read_auth_json(auth_path) == {"anthropic": {"type": "api_key", "key": "stored"}}
    finally:
        auth_storage_module._acquire_lock_async = original_acquire

    await storage.modify("openai", set_openai)
    assert read_auth_json(auth_path) == {
        "anthropic": {"type": "api_key", "key": "stored"},
        "openai": {"type": "api_key", "key": "new"},
    }


@pytest.mark.tonio
async def test_does_not_overwrite_malformed_auth_files(tmp_dir):
    auth_path = tmp_dir / "auth.json"
    write_auth_json(auth_path, {"anthropic": {"type": "api_key", "key": "stored"}})
    storage = AuthStorage.create(str(auth_path))
    auth_path.write_text("{invalid-json", encoding="utf-8")

    async def set_openai(_current):
        return ApiKeyCredential(key="new")

    with pytest.raises(Exception):
        await storage.modify("openai", set_openai)
    assert auth_path.read_text(encoding="utf-8") == "{invalid-json"
