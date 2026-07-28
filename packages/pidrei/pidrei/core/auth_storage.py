"""Mirror of pi coding-agent src/core/auth-storage.ts.

CredentialStore implementation backed by auth.json. Provider auth
orchestration belongs to ModelRuntime and pidrei-ai Models.

pi's proper-lockfile "compromised lock" watchdog (mtime polling on the held
lock) is not ported; the mkdir-based lock plus retry/backoff covers the
observable locking contract.
"""

import json
import os
import random
from collections.abc import Awaitable, Callable
from typing import Any

import tonio.colored as tonio

from pidrei_ai.auth.types import ApiKeyCredential, Credential, CredentialInfo, CredentialStore, OAuthCredential

from ..config import get_agent_dir
from ..utils import lockfile
from ..utils.paths import normalize_path
from .resolve_config_value import resolve_config_value


def parse_credential(raw: dict[str, Any]) -> Credential:
    """Parse one auth.json credential object into the typed credential."""
    if raw.get("type") == "oauth":
        extra = {key: value for key, value in raw.items() if key not in ("type", "refresh", "access", "expires")}
        return OAuthCredential(refresh=raw["refresh"], access=raw["access"], expires=raw["expires"], extra=extra)
    return ApiKeyCredential(key=raw.get("key"), env=raw.get("env"))


def serialize_credential(credential: Credential) -> dict[str, Any]:
    """Serialize a credential back to the auth.json shape (extras flattened,
    None optionals omitted like JSON.stringify drops undefined)."""
    if isinstance(credential, OAuthCredential):
        return {
            "type": "oauth",
            "access": credential.access,
            "refresh": credential.refresh,
            "expires": credential.expires,
            **credential.extra,
        }
    raw: dict[str, Any] = {"type": "api_key"}
    if credential.key is not None:
        raw["key"] = credential.key
    if credential.env is not None:
        raw["env"] = dict(credential.env)
    return raw


async def _acquire_lock_async(path: str) -> Callable[[], None]:
    """Async lock acquisition mirroring pi's proper-lockfile retry profile:
    10 retries, exponential backoff from 100ms capped at 10s, randomized,
    30s staleness."""
    attempt = 0
    while True:
        try:
            return await tonio.spawn_blocking(lockfile.lock_sync, path, stale=30.0)
        except lockfile.LockedError:
            if attempt >= 10:
                raise
            delay = min(0.1 * (2**attempt), 10.0) * (1 + random.random())  # noqa: S311
            attempt += 1
            await tonio.time.sleep(delay)


class FileAuthStorageBackend:
    def __init__(self, auth_path: str | None = None):
        if auth_path is None:
            auth_path = os.path.join(get_agent_dir(), "auth.json")
        self._auth_path = normalize_path(auth_path)

    def _ensure_parent_dir(self) -> None:
        directory = os.path.dirname(self._auth_path)
        if not os.path.exists(directory):
            os.makedirs(directory, mode=0o700, exist_ok=True)

    def _ensure_file_exists(self) -> None:
        if not os.path.exists(self._auth_path):
            self._write("{}")

    def _ensure_ready(self) -> None:
        """Both preparation steps as one unit, so the async path pays one hop."""
        self._ensure_parent_dir()
        self._ensure_file_exists()

    def _read(self) -> str | None:
        if not os.path.exists(self._auth_path):
            return None
        with open(self._auth_path, encoding="utf-8") as f:
            return f.read()

    def _write(self, content: str) -> None:
        with open(self._auth_path, "w", encoding="utf-8") as f:
            f.write(content)
        os.chmod(self._auth_path, 0o600)

    def with_lock(self, fn: Callable[[str | None], tuple[Any, str | None]]) -> Any:
        self._ensure_parent_dir()
        self._ensure_file_exists()

        release = lockfile.acquire_lock_sync_with_retry(self._auth_path)
        try:
            result, next_content = fn(self._read())
            if next_content is not None:
                self._write(next_content)
            return result
        finally:
            release()

    async def with_lock_async(self, fn: Callable[[str | None], Awaitable[tuple[Any, str | None]]]) -> Any:
        """The callback is async, so this cannot be offloaded as one unit the
        way `with_lock` can — each filesystem step goes to the pool on its own,
        including the lock release in the `finally`."""
        await tonio.spawn_blocking(self._ensure_ready)

        release = await _acquire_lock_async(self._auth_path)
        try:
            current = await tonio.spawn_blocking(self._read)
            result, next_content = await fn(current)
            if next_content is not None:
                await tonio.spawn_blocking(self._write, next_content)
            return result
        finally:
            await tonio.spawn_blocking(release)


class InMemoryAuthStorageBackend:
    def __init__(self):
        self._value: str | None = None

    def with_lock(self, fn: Callable[[str | None], tuple[Any, str | None]]) -> Any:
        result, next_content = fn(self._value)
        if next_content is not None:
            self._value = next_content
        return result

    async def with_lock_async(self, fn: Callable[[str | None], Awaitable[tuple[Any, str | None]]]) -> Any:
        result, next_content = await fn(self._value)
        if next_content is not None:
            self._value = next_content
        return result


type AuthStorageBackend = FileAuthStorageBackend | InMemoryAuthStorageBackend


def _serialize_data(data: dict[str, Credential]) -> str:
    return json.dumps({provider: serialize_credential(credential) for provider, credential in data.items()}, indent=2)


class AuthStorage(CredentialStore):
    """Credential storage backed by a JSON file."""

    def __init__(self, storage: AuthStorageBackend):
        self._storage = storage
        self._data: dict[str, Credential] = {}
        # No load here: reading auth.json is I/O and a constructor cannot
        # await. `create()` loads asynchronously; `in_memory()` loads inline
        # because its backend never touches a file.

    @staticmethod
    async def create(auth_path: str | None = None) -> AuthStorage:
        storage = AuthStorage(FileAuthStorageBackend(auth_path))
        await storage.reload_async()
        return storage

    @staticmethod
    def from_storage(storage: AuthStorageBackend) -> AuthStorage:
        return AuthStorage(storage)

    @staticmethod
    def in_memory(data: dict[str, Credential] | None = None) -> AuthStorage:
        storage = InMemoryAuthStorageBackend()
        content = _serialize_data(data or {})
        storage.with_lock(lambda _current: (None, content))
        store = AuthStorage.from_storage(storage)
        store.reload()
        return store

    def _parse_storage_data(self, content: str | None) -> dict[str, Credential]:
        if not content:
            return {}
        return {provider: parse_credential(raw) for provider, raw in json.loads(content).items()}

    def reload(self) -> None:
        """Reload credentials from storage.

        Sync, so only safe for a backend that does no file I/O — i.e. the
        in-memory one. File-backed stores use `reload_async`.
        """
        try:
            content = self._storage.with_lock(lambda current: (current, None))
            self._data = self._parse_storage_data(content)
        except Exception:
            pass  # Preserve the last valid in-memory snapshot.

    async def reload_async(self) -> None:
        """Reload credentials from storage without blocking a runtime worker."""

        async def read(current: str | None) -> tuple[str | None, str | None]:
            return current, None

        try:
            content = await self._storage.with_lock_async(read)
            self._data = self._parse_storage_data(content)
        except Exception:
            pass  # Preserve the last valid in-memory snapshot.

    async def read(self, provider: str) -> Credential | None:
        credential = self._data.get(provider)
        if not isinstance(credential, ApiKeyCredential):
            return credential
        if credential.key is None:
            return credential
        # `resolve_config_value` may run a shell command (`!cmd` syntax), so it
        # goes to the pool rather than blocking a runtime worker.
        resolved = await resolve_config_value(credential.key, credential.env)
        return ApiKeyCredential(key=resolved, env=credential.env)

    async def modify(
        self,
        provider: str,
        fn: Callable[[Credential | None], Awaitable[Credential | None]],
    ) -> Credential | None:
        async def under_lock(content: str | None) -> tuple[Credential | None, str | None]:
            current_data = self._parse_storage_data(content)
            next_credential = await fn(current_data.get(provider))
            if next_credential is None:
                self._data = current_data
                return current_data.get(provider), None

            merged = {**current_data, provider: next_credential}
            self._data = merged
            return next_credential, _serialize_data(merged)

        return await self._storage.with_lock_async(under_lock)

    async def delete(self, provider: str) -> None:
        async def under_lock(content: str | None) -> tuple[None, str]:
            current_data = self._parse_storage_data(content)
            current_data.pop(provider, None)
            self._data = current_data
            return None, _serialize_data(current_data)

        await self._storage.with_lock_async(under_lock)

    async def list(self) -> list[CredentialInfo]:
        """List credential metadata without resolving configured key values."""
        return [
            CredentialInfo(provider_id=provider_id, type=credential.type)
            for provider_id, credential in self._data.items()
        ]


def read_stored_credential(provider_id: str, auth_path: str | None = None) -> Credential | None:
    """One-off synchronous read of a stored credential from an auth.json file,
    without instantiating a store or resolving configured key values."""
    if auth_path is None:
        auth_path = os.path.join(get_agent_dir(), "auth.json")
    try:
        with open(normalize_path(auth_path), encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get(provider_id)
        return parse_credential(raw) if raw is not None else None
    except Exception:
        return None
