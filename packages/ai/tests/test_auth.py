"""Tests for the auth layer port (credential store + resolve_provider_auth)."""

import time
from dataclasses import dataclass, field

import pytest
import tonio.colored as tonio

from pppi_ai.auth.credential_store import InMemoryCredentialStore
from pppi_ai.auth.resolve import AuthResolutionOverrides, ModelsError, resolve_provider_auth
from pppi_ai.auth.types import (
    ApiKeyAuth,
    ApiKeyCredential,
    AuthResult,
    ModelAuth,
    OAuthAuth,
    OAuthCredential,
    ProviderAuth,
)


def now_ms() -> int:
    return int(time.time() * 1000)


class FakeAuthContext:
    def __init__(self, env: dict[str, str] | None = None):
        self._env = env or {}

    async def env(self, name: str) -> str | None:
        return self._env.get(name)

    async def file_exists(self, path: str) -> bool:
        return False


@dataclass
class FakeProvider:
    id: str
    auth: ProviderAuth


def env_api_key_auth(env_name: str) -> ApiKeyAuth:
    """Resolve like pi's env-var providers: stored key wins, then ctx env."""

    async def resolve(ctx, credential):
        key = credential.key if credential is not None and credential.key else await ctx.env(env_name)
        if not key:
            return None
        source = "stored" if credential is not None and credential.key else env_name
        return AuthResult(auth=ModelAuth(api_key=key), env=credential.env if credential else None, source=source)

    return ApiKeyAuth(name="Test API key", resolve=resolve)


@dataclass
class OAuthCalls:
    refreshes: int = 0
    to_auths: list[str] = field(default_factory=list)


def make_oauth(calls: OAuthCalls, *, fail_refresh: bool = False, rotate_to: str = "rotated") -> OAuthAuth:
    async def login(interaction):
        raise NotImplementedError

    async def refresh(credential, cancel):
        calls.refreshes += 1
        if fail_refresh:
            raise RuntimeError("invalid_grant")
        return OAuthCredential(refresh=credential.refresh, access=rotate_to, expires=now_ms() + 3_600_000)

    async def to_auth(credential):
        calls.to_auths.append(credential.access)
        return ModelAuth(api_key=credential.access)

    return OAuthAuth(name="Test OAuth", login=login, refresh=refresh, to_auth=to_auth)


# -- credential store ----------------------------------------------------------


@pytest.mark.tonio
async def test_store_modify_returns_current_when_fn_returns_none():
    store = InMemoryCredentialStore()

    async def keep(_current):
        return None

    assert await store.modify("p", keep) is None

    async def write(_current):
        return ApiKeyCredential(key="k")

    await store.modify("p", write)
    result = await store.modify("p", keep)
    assert result is not None
    assert result.key == "k"


@pytest.mark.tonio
async def test_store_list_and_delete():
    store = InMemoryCredentialStore()

    async def write(_current):
        return ApiKeyCredential(key="k")

    await store.modify("p1", write)
    infos = await store.list()
    assert [(info.provider_id, info.type) for info in infos] == [("p1", "api_key")]

    await store.delete("p1")
    assert await store.read("p1") is None
    assert await store.list() == []


@pytest.mark.tonio
async def test_store_serializes_modify_per_provider():
    store = InMemoryCredentialStore()
    order: list[str] = []

    async def slow(_current):
        order.append("slow:start")
        await tonio.yield_now()
        await tonio.yield_now()
        order.append("slow:end")
        return ApiKeyCredential(key="slow")

    async def fast(_current):
        order.append("fast:start")
        order.append("fast:end")
        return ApiKeyCredential(key="fast")

    async def run_slow():
        return await store.modify("p", slow)

    async def run_fast():
        await tonio.yield_now()
        return await store.modify("p", fast)

    await tonio.spawn(run_slow(), run_fast())
    # Whatever the interleaving of task starts, the two modify bodies must not overlap.
    assert order.index("slow:end") < order.index("fast:start") or order.index("fast:end") < order.index("slow:start")


# -- resolve_provider_auth -----------------------------------------------------


@pytest.mark.tonio
async def test_override_api_key_wins():
    provider = FakeProvider("p", ProviderAuth(api_key=env_api_key_auth("TEST_KEY")))
    store = InMemoryCredentialStore()

    result = await resolve_provider_auth(
        provider, store, FakeAuthContext(), AuthResolutionOverrides(api_key="override-key")
    )
    assert result is not None
    assert result.auth.api_key == "override-key"
    assert result.source == "stored"


@pytest.mark.tonio
async def test_ambient_env_resolution_when_nothing_stored():
    provider = FakeProvider("p", ProviderAuth(api_key=env_api_key_auth("TEST_KEY")))
    store = InMemoryCredentialStore()

    result = await resolve_provider_auth(provider, store, FakeAuthContext({"TEST_KEY": "from-env"}))
    assert result is not None
    assert result.auth.api_key == "from-env"
    assert result.source == "TEST_KEY"

    assert await resolve_provider_auth(provider, store, FakeAuthContext()) is None


@pytest.mark.tonio
async def test_overrides_env_overlays_auth_context():
    provider = FakeProvider("p", ProviderAuth(api_key=env_api_key_auth("TEST_KEY")))
    store = InMemoryCredentialStore()

    result = await resolve_provider_auth(
        provider, store, FakeAuthContext(), AuthResolutionOverrides(env={"TEST_KEY": "scoped"})
    )
    assert result is not None
    assert result.auth.api_key == "scoped"


@pytest.mark.tonio
async def test_stored_credential_owns_provider_no_env_fallback():
    # A stored oauth credential with no oauth handler resolves to None — never env.
    provider = FakeProvider("p", ProviderAuth(api_key=env_api_key_auth("TEST_KEY")))
    store = InMemoryCredentialStore()

    async def write(_current):
        return OAuthCredential(refresh="r", access="a", expires=now_ms() + 3_600_000)

    await store.modify("p", write)
    result = await resolve_provider_auth(provider, store, FakeAuthContext({"TEST_KEY": "from-env"}))
    assert result is None


@pytest.mark.tonio
async def test_valid_oauth_costs_zero_refreshes():
    calls = OAuthCalls()
    provider = FakeProvider("p", ProviderAuth(oauth=make_oauth(calls)))
    store = InMemoryCredentialStore()

    async def write(_current):
        return OAuthCredential(refresh="r", access="valid", expires=now_ms() + 3_600_000)

    await store.modify("p", write)
    result = await resolve_provider_auth(provider, store, FakeAuthContext())
    assert result is not None
    assert result.auth.api_key == "valid"
    assert result.source == "OAuth"
    assert calls.refreshes == 0


@pytest.mark.tonio
async def test_expired_oauth_refreshes_once_under_lock_and_persists():
    calls = OAuthCalls()
    provider = FakeProvider("p", ProviderAuth(oauth=make_oauth(calls)))
    store = InMemoryCredentialStore()

    async def write(_current):
        return OAuthCredential(refresh="r", access="stale", expires=now_ms() - 1000)

    await store.modify("p", write)

    async def resolve_once():
        return await resolve_provider_auth(provider, store, FakeAuthContext())

    first, second = await tonio.spawn(resolve_once(), resolve_once())
    assert first is not None and first.auth.api_key == "rotated"
    assert second is not None and second.auth.api_key == "rotated"
    # Double-checked locking: exactly one refresh despite two concurrent resolves.
    assert calls.refreshes == 1

    stored = await store.read("p")
    assert stored is not None and stored.type == "oauth" and stored.access == "rotated"


@pytest.mark.tonio
async def test_failed_oauth_refresh_raises_models_error_and_preserves_credential():
    calls = OAuthCalls()
    provider = FakeProvider("p", ProviderAuth(oauth=make_oauth(calls, fail_refresh=True)))
    store = InMemoryCredentialStore()

    async def write(_current):
        return OAuthCredential(refresh="r", access="stale", expires=now_ms() - 1000)

    await store.modify("p", write)

    with pytest.raises(ModelsError) as excinfo:
        await resolve_provider_auth(provider, store, FakeAuthContext())
    assert excinfo.value.code == "oauth"

    stored = await store.read("p")
    assert stored is not None and stored.type == "oauth" and stored.access == "stale"


@pytest.mark.tonio
async def test_stored_api_key_with_env_override_merge():
    provider = FakeProvider("p", ProviderAuth(api_key=env_api_key_auth("TEST_KEY")))
    store = InMemoryCredentialStore()

    async def write(_current):
        return ApiKeyCredential(key="stored-key", env={"REGION": "eu", "KEEP": "yes"})

    await store.modify("p", write)
    result = await resolve_provider_auth(
        provider, store, FakeAuthContext(), AuthResolutionOverrides(env={"REGION": "us"})
    )
    assert result is not None
    assert result.auth.api_key == "stored-key"
    assert result.env == {"REGION": "us", "KEEP": "yes"}
