"""Mirror of pi coding-agent test/auth-storage.test.ts.

Adaptations:
- pi's "compromised lock" tests spy proper-lockfile's mtime watchdog, which is
  not ported; the equivalent pidrei failure seam is async lock acquisition
  itself, so those tests patch this module's `_acquire_lock_async` instead.
"""

import contextlib
import json
import time

import pytest
import tonio.colored as tonio

import pidrei.core.auth_storage as auth_storage_module
from pidrei.core.auth_storage import AuthStorage, FileAuthStorageBackend
from pidrei.utils import lockfile
from pidrei_ai.auth.resolve import ModelsError
from pidrei_ai.auth.types import (
    ApiKeyCredential,
    AuthOperationOptions,
    CredentialInfo,
    ModelAuth,
    OAuthAuth,
    OAuthCredential,
    ProviderAuth,
)
from pidrei_ai.registry import create_models, create_provider
from pidrei_ai.utils.cancel import AbortError, CancelToken


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
        storage = await AuthStorage.create(str(auth_path))
        assert await storage.read("anthropic") == ApiKeyCredential(key="environment-key")


@pytest.mark.tonio
async def test_resolves_command_backed_api_key_credentials(tmp_dir):
    auth_path = tmp_dir / "auth.json"
    write_auth_json(auth_path, {"anthropic": {"type": "api_key", "key": "!printf 'command-key'"}})
    storage = await AuthStorage.create(str(auth_path))
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
    storage = await AuthStorage.create(str(auth_path))
    resolved = await storage.read("anthropic")
    assert resolved.key == "scoped-value"
    assert resolved.env == {"SCOPED_KEY": "scoped-value", "REGION": "test-region"}


@pytest.mark.tonio
async def test_coalesces_file_reloads_across_concurrent_readers_and_storage_instances(tmp_dir):
    auth_path = tmp_dir / "auth.json"
    write_auth_json(auth_path, {"anthropic": {"type": "api_key", "key": "old"}})
    first = await AuthStorage.create(str(auth_path))
    second = await AuthStorage.create(str(auth_path))

    calls = {"count": 0}
    original_acquire = auth_storage_module._acquire_lock_async

    async def counting_acquire(path, cancel=None):
        calls["count"] += 1
        return await original_acquire(path, cancel)

    auth_storage_module._acquire_lock_async = counting_acquire
    try:
        write_auth_json(
            auth_path,
            {"anthropic": {"type": "api_key", "key": "new"}, "openai": {"type": "api_key", "key": "openai-key"}},
        )

        anthropic, openai, credentials = await tonio.spawn(first.read("anthropic"), second.read("openai"), first.list())
        assert anthropic == ApiKeyCredential(key="new")
        assert openai == ApiKeyCredential(key="openai-key")
        assert credentials == [
            CredentialInfo(provider_id="anthropic", type="api_key"),
            CredentialInfo(provider_id="openai", type="api_key"),
        ]
        assert calls["count"] == 1

        assert await second.read("anthropic") == ApiKeyCredential(key="new")
        assert calls["count"] == 1

        write_auth_json(auth_path, {"anthropic": {"type": "api_key", "key": "newest"}})
        assert await first.read("anthropic") == ApiKeyCredential(key="newest")
        assert calls["count"] == 2
    finally:
        auth_storage_module._acquire_lock_async = original_acquire


@pytest.mark.tonio
async def test_modify_persists_a_credential_while_preserving_unrelated_external_edits(tmp_dir):
    auth_path = tmp_dir / "auth.json"
    write_auth_json(auth_path, {"anthropic": {"type": "api_key", "key": "old"}})
    storage = await AuthStorage.create(str(auth_path))
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
    storage = await AuthStorage.create(str(auth_path))

    async def keep(_current):
        return None

    assert await storage.modify("anthropic", keep) == ApiKeyCredential(key="stored")
    assert await storage.read("anthropic") == ApiKeyCredential(key="stored")


@pytest.mark.tonio
async def test_serializes_concurrent_modifications(tmp_dir):
    auth_path = tmp_dir / "auth.json"
    write_auth_json(auth_path, {})
    first = await AuthStorage.create(str(auth_path))
    second = await AuthStorage.create(str(auth_path))

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
    storage = await AuthStorage.create(str(auth_path))
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
    storage = await AuthStorage.create(str(auth_path))

    original_acquire = auth_storage_module._acquire_lock_async
    calls = {"count": 0}

    async def failing_acquire(path, cancel=None):
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
async def test_pre_aborted_file_operations_do_not_create_the_backing_file_or_run_the_mutation(tmp_dir):
    auth_path = tmp_dir / "auth.json"
    backend = FileAuthStorageBackend(str(auth_path))
    controller = CancelToken()
    controller.cancel()
    calls = {"count": 0}

    async def update(_content):
        calls["count"] += 1
        return None, json.dumps({})

    with pytest.raises(AbortError):
        await backend.with_lock_async(update, AuthOperationOptions(cancel=controller))
    assert calls["count"] == 0
    assert not auth_path.exists()


@pytest.mark.tonio
async def test_aborts_while_waiting_for_a_held_file_lock_without_running_the_mutation_later(tmp_dir):
    auth_path = tmp_dir / "auth.json"
    write_auth_json(auth_path, {"anthropic": {"type": "api_key", "key": "stored"}})
    release = lockfile.lock_sync(str(auth_path), stale=30.0)
    backend = FileAuthStorageBackend(str(auth_path))
    controller = CancelToken()
    calls = {"count": 0}

    async def update(_content):
        calls["count"] += 1
        return None, json.dumps({})

    outcome: dict = {}

    async def run_pending() -> None:
        try:
            await backend.with_lock_async(update, AuthOperationOptions(cancel=controller))
            outcome["error"] = None
        except BaseException as error:
            outcome["error"] = error

    async def drive() -> None:
        await tonio.time.sleep(0.01)
        controller.cancel()

    await tonio.spawn(run_pending(), drive())
    assert isinstance(outcome["error"], AbortError)
    assert calls["count"] == 0

    release()
    await tonio.time.sleep(0.15)
    assert calls["count"] == 0
    assert read_auth_json(auth_path) == {"anthropic": {"type": "api_key", "key": "stored"}}


@pytest.mark.tonio
async def test_releases_a_file_lock_acquired_concurrently_with_cancellation_before_mutation(tmp_dir):
    auth_path = tmp_dir / "auth.json"
    write_auth_json(auth_path, {"anthropic": {"type": "api_key", "key": "stored"}})
    backend = FileAuthStorageBackend(str(auth_path))
    controller = CancelToken()
    released = {"count": 0}
    calls = {"count": 0}

    def fake_lock_sync(_path, stale=None):
        controller.cancel()

        def release() -> None:
            released["count"] += 1

        return release

    async def update(_content):
        calls["count"] += 1
        return None, json.dumps({})

    original_lock_sync = auth_storage_module.lockfile.lock_sync
    auth_storage_module.lockfile.lock_sync = fake_lock_sync
    try:
        with pytest.raises(AbortError):
            await backend.with_lock_async(update, AuthOperationOptions(cancel=controller))
        await tonio.time.sleep(0.01)
    finally:
        auth_storage_module.lockfile.lock_sync = original_lock_sync
    assert calls["count"] == 0
    assert released["count"] == 1


@pytest.mark.tonio
async def test_holds_the_file_lock_until_a_cancelled_active_callback_settles_without_committing_it(tmp_dir):
    auth_path = tmp_dir / "auth.json"
    write_auth_json(auth_path, {"anthropic": {"type": "api_key", "key": "stored"}})
    backend = FileAuthStorageBackend(str(auth_path))
    controller = CancelToken()
    started = tonio.Event()
    blocked = tonio.Event()
    competing_calls = {"count": 0}
    outcome: dict = {}

    async def cancelled_update(_content):
        started.set()
        await blocked.wait()
        return None, json.dumps({"openai": {"type": "api_key", "key": "cancelled"}})

    async def competing_update(_content):
        competing_calls["count"] += 1
        return None, json.dumps({"google": {"type": "api_key", "key": "committed"}})

    async def run_pending() -> None:
        try:
            await backend.with_lock_async(cancelled_update, AuthOperationOptions(cancel=controller))
            outcome["error"] = None
        except BaseException as error:
            outcome["error"] = error

    async def run_competing() -> None:
        await started.wait()
        controller.cancel()
        pending_competing = backend.with_lock_async(competing_update)
        await tonio.time.sleep(0.02)
        assert competing_calls["count"] == 0
        blocked.set()
        await pending_competing

    await tonio.spawn(run_pending(), run_competing())
    assert isinstance(outcome["error"], AbortError)
    assert competing_calls["count"] == 1
    assert read_auth_json(auth_path) == {"google": {"type": "api_key", "key": "committed"}}


@pytest.mark.tonio
async def test_cancels_a_signalled_credential_read_waiting_for_a_held_file_lock(tmp_dir):
    """pi also asserts a single lock attempt; pidrei's coalesced shared reload
    (see `_read_latest_data`) completes detached after the release, so only
    the caller-visible semantics are asserted here."""
    auth_path = tmp_dir / "auth.json"
    write_auth_json(auth_path, {"anthropic": {"type": "api_key", "key": "old"}})
    storage = await AuthStorage.create(str(auth_path))
    write_auth_json(auth_path, {"anthropic": {"type": "api_key", "key": "new-value"}})
    release = lockfile.lock_sync(str(auth_path), stale=30.0)
    controller = CancelToken()
    outcome: dict = {}

    async def run_pending() -> None:
        try:
            await storage.read("anthropic", AuthOperationOptions(cancel=controller))
            outcome["error"] = None
        except BaseException as error:
            outcome["error"] = error

    async def drive() -> None:
        await tonio.time.sleep(0.01)
        controller.cancel()

    await tonio.spawn(run_pending(), drive())
    assert isinstance(outcome["error"], AbortError)
    release()
    await tonio.time.sleep(0.15)
    assert await storage.read("anthropic") == ApiKeyCredential(key="new-value")


@pytest.mark.tonio
async def test_serializes_in_memory_mutations_across_providers():
    storage = AuthStorage.in_memory()
    started = tonio.Event()
    blocked = tonio.Event()
    second_calls = {"count": 0}

    async def first_fn(_current):
        started.set()
        await blocked.wait()
        return ApiKeyCredential(key="anthropic-key")

    async def second_fn(_current):
        second_calls["count"] += 1
        return ApiKeyCredential(key="openai-key")

    async def run_first() -> None:
        await storage.modify("anthropic", first_fn)

    async def run_second() -> None:
        await started.wait()
        pending_second = storage.modify("openai", second_fn)
        await tonio.time.sleep(0.01)
        assert second_calls["count"] == 0
        blocked.set()
        await pending_second

    await tonio.spawn(run_first(), run_second())
    assert await storage.read("anthropic") == ApiKeyCredential(key="anthropic-key")
    assert await storage.read("openai") == ApiKeyCredential(key="openai-key")


@pytest.mark.tonio
async def test_cancels_a_queued_in_memory_mutation_without_running_it_later():
    storage = AuthStorage.in_memory()
    started = tonio.Event()
    blocked = tonio.Event()
    second_calls = {"count": 0}
    controller = CancelToken()
    outcome: dict = {}

    async def first_fn(_current):
        started.set()
        await blocked.wait()
        return ApiKeyCredential(key="anthropic-key")

    async def second_fn(_current):
        second_calls["count"] += 1
        return ApiKeyCredential(key="openai-key")

    async def run_first() -> None:
        await storage.modify("anthropic", first_fn)

    async def run_second() -> None:
        await started.wait()
        try:
            controller.cancel()
            await storage.modify("openai", second_fn, AuthOperationOptions(cancel=controller))
            outcome["error"] = None
        except BaseException as error:
            outcome["error"] = error
        blocked.set()

    await tonio.spawn(run_first(), run_second())
    await tonio.time.sleep(0.01)
    assert isinstance(outcome["error"], AbortError)
    assert second_calls["count"] == 0
    assert await storage.read("openai") is None


@pytest.mark.tonio
async def test_preserves_the_stored_credential_after_cancelling_an_active_refresh_mutation():
    previous = OAuthCredential(access="expired", refresh="refresh-token", expires=0)
    storage = AuthStorage.in_memory({"oauth": previous})
    controller = CancelToken()
    started = tonio.Event()
    blocked = tonio.Event()
    competing_calls = {"count": 0}
    outcome: dict = {}

    async def refresh_fn(_current):
        started.set()
        await blocked.wait()
        return OAuthCredential(access="refreshed", refresh="refresh-token", expires=int(time.time() * 1000) + 60_000)

    async def competing_fn(_current):
        competing_calls["count"] += 1
        return ApiKeyCredential(key="other")

    async def run_pending() -> None:
        try:
            await storage.modify("oauth", refresh_fn, AuthOperationOptions(cancel=controller))
            outcome["error"] = None
        except BaseException as error:
            outcome["error"] = error

    async def run_competing() -> None:
        await started.wait()
        controller.cancel()
        pending_competing = storage.modify("other", competing_fn)
        await tonio.time.sleep(0.01)
        assert competing_calls["count"] == 0
        blocked.set()
        await pending_competing

    await tonio.spawn(run_pending(), run_competing())
    assert isinstance(outcome["error"], AbortError)
    assert competing_calls["count"] == 1
    assert await storage.read("oauth") == previous


@pytest.mark.tonio
async def test_retries_a_briefly_contended_file_lock(tmp_dir):
    auth_path = tmp_dir / "auth.json"
    write_auth_json(auth_path, {"anthropic": {"type": "api_key", "key": "stored"}})
    backend = FileAuthStorageBackend(str(auth_path))
    attempts = {"count": 0}
    released = {"count": 0}

    def contended_lock_sync(_path, stale=None):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise lockfile.LockedError("locked")

        def release() -> None:
            released["count"] += 1

        return release

    update_calls = {"count": 0}

    async def update(_content):
        update_calls["count"] += 1
        return None, None

    original_lock_sync = auth_storage_module.lockfile.lock_sync
    auth_storage_module.lockfile.lock_sync = contended_lock_sync
    try:
        await backend.with_lock_async(update)
    finally:
        auth_storage_module.lockfile.lock_sync = original_lock_sync

    assert attempts["count"] == 2
    assert update_calls["count"] == 1
    assert released["count"] == 1


@pytest.mark.tonio
async def test_surfaces_a_compromised_file_storage_lock(tmp_dir):
    auth_path = tmp_dir / "auth.json"
    write_auth_json(auth_path, {"anthropic": {"type": "api_key", "key": "stored"}})
    backend = FileAuthStorageBackend(str(auth_path))
    calls = {"count": 0}

    async def update(_content):
        calls["count"] += 1
        return None, json.dumps({})

    async def compromised_acquire(path, cancel=None):
        raise Exception("lock compromised")

    original_acquire = auth_storage_module._acquire_lock_async
    auth_storage_module._acquire_lock_async = compromised_acquire
    try:
        with pytest.raises(Exception, match="lock compromised"):
            await backend.with_lock_async(update)
    finally:
        auth_storage_module._acquire_lock_async = original_acquire

    assert calls["count"] == 0
    assert read_auth_json(auth_path) == {"anthropic": {"type": "api_key", "key": "stored"}}


class _UnusedApi:
    def stream(self, *_args, **_kwargs):
        raise Exception("not used")

    def stream_simple(self, *_args, **_kwargs):
        raise Exception("not used")


@pytest.mark.tonio
async def test_translates_a_credential_store_refresh_failure_and_allows_a_later_retry():
    provider_id = "oauth-provider"
    base = AuthStorage.in_memory(
        {provider_id: OAuthCredential(access="expired-access", refresh="refresh-token", expires=0)}
    )
    state = {"fail_next_modify": True}

    class FlakyCredentialStore:
        async def read(self, id, options=None):
            return await base.read(id, options)

        async def list(self, options=None):
            return await base.list(options)

        async def modify(self, id, fn, options=None):
            if state["fail_next_modify"]:
                state["fail_next_modify"] = False
                raise Exception("credential store unavailable")
            return await base.modify(id, fn, options)

        async def delete(self, id, options=None):
            await base.delete(id, options)

    async def login(_interaction):
        raise Exception("not used")

    async def refresh(credential, _cancel):
        return OAuthCredential(
            access="refreshed-access",
            refresh=credential.refresh,
            expires=int(time.time() * 1000) + 60_000,
            extra=credential.extra,
        )

    async def to_auth(credential):
        return ModelAuth(api_key=credential.access)

    provider = create_provider(
        id=provider_id,
        name="OAuth Provider",
        auth=ProviderAuth(oauth=OAuthAuth(name="OAuth", login=login, refresh=refresh, to_auth=to_auth)),
        models=[],
        api=_UnusedApi(),
    )
    models = create_models(credentials=FlakyCredentialStore())
    models.set_provider(provider)

    with pytest.raises(ModelsError) as exc_info:
        await models.get_auth(provider_id)
    assert exc_info.value.code == "auth"

    result = await models.get_auth(provider_id)
    assert result is not None
    assert result.auth.api_key == "refreshed-access"


@pytest.mark.tonio
async def test_does_not_overwrite_malformed_auth_files(tmp_dir):
    auth_path = tmp_dir / "auth.json"
    write_auth_json(auth_path, {"anthropic": {"type": "api_key", "key": "stored"}})
    storage = await AuthStorage.create(str(auth_path))
    auth_path.write_text("{invalid-json", encoding="utf-8")

    async def set_openai(_current):
        return ApiKeyCredential(key="new")

    with pytest.raises(Exception):
        await storage.modify("openai", set_openai)
    assert auth_path.read_text(encoding="utf-8") == "{invalid-json"
