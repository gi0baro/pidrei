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
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import tonio.colored as tonio

from pidrei_ai.auth.types import (
    ApiKeyCredential,
    AuthOperationOptions,
    Credential,
    CredentialInfo,
    CredentialStore,
    OAuthCredential,
)
from pidrei_ai.utils import clock
from pidrei_ai.utils.cancel import AbortError, CancelToken

from ..config import get_agent_dir
from ..utils import lockfile
from ..utils.abort import race_with_cancel
from ..utils.paths import get_file_revision, normalize_path
from .resolve_config_value import is_command_config_value, resolve_config_value


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


@dataclass(slots=True)
class _AuthFileReload:
    controller: CancelToken
    done: Any  # tonio.Event marking the reload settled
    box: list  # single-slot result/error holder
    readers: int = 0


@dataclass(slots=True)
class _AuthFileReadState:
    """Read-side cache shared by every AuthStorage bound to the same file.

    `reload` holds the in-flight coalesced reload (pi shares a promise; the
    Event/box pair is the established tonio equivalent), reference-counted so
    the last departing reader aborts a reload nobody is waiting for."""

    data: dict[str, Credential] = field(default_factory=dict)
    revision: str | None = None
    reload: _AuthFileReload | None = None
    guard: threading.Lock = field(default_factory=threading.Lock)


# pi keeps a single shared slot ("optimize the common path without retaining
# an unbounded set of custom paths") because vitest gives every test file a
# fresh module registry. pytest shares one process, so a single slot would be
# claimed by the first test's tmp path and disable revision caching for every
# later store; a per-path map keeps pi's sharing semantics for all paths.
_shared_auth_file_read_states: dict[str, _AuthFileReadState] = {}

# Test seam (the counterpart of pi's utils/paths getFileRevision import).
_get_file_revision = get_file_revision


async def _acquire_lock_async(path: str, cancel: CancelToken | None = None) -> Callable[[], None]:
    """Async lock acquisition mirroring pi's manual retry loop: short
    randomized exponential backoff from 10ms capped at 1s, retried until the
    30s staleness window would have reclaimed the lock anyway, abortable
    between attempts."""
    stale_ms = 30_000
    max_delay_ms = 2_000
    deadline = clock.now_ms() + stale_ms
    retry = 0
    while True:
        if cancel is not None:
            cancel.raise_if_cancelled()
        try:
            release = await tonio.spawn_blocking(lockfile.lock_sync, path, stale=stale_ms / 1000)
        except lockfile.LockedError:
            if cancel is not None:
                cancel.raise_if_cancelled()
            remaining_ms = deadline - clock.now_ms()
            if remaining_ms <= 0:
                raise
            base_delay_ms = min(10 * (2**retry), max_delay_ms / 2)
            retry += 1
            delay_ms = min(round(base_delay_ms * (1 + random.random())), remaining_ms)  # noqa: S311
            try:
                await clock.sleep_ms(delay_ms, cancel)
            except AbortError:
                if cancel is not None:
                    cancel.raise_if_cancelled()
                raise
            continue
        if cancel is not None and cancel.cancelled:
            await tonio.spawn_blocking(release)
            cancel.raise_if_cancelled()
        return release


class FileAuthStorageBackend:
    def __init__(self, auth_path: str | None = None):
        if auth_path is None:
            auth_path = os.path.join(get_agent_dir(), "auth.json")
        self._auth_path = normalize_path(auth_path)
        # In-process queue in front of the cross-process disk lock: pi's
        # proper-lockfile contends on disk even within one process; here the
        # 100ms+ lock backoff would turn a burst of per-provider reads into a
        # backoff storm, so same-process callers serialize on this chain and
        # only genuinely concurrent processes ever hit the retry ladder.
        self._async_chain: Any = None
        self._chain_guard = threading.Lock()

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

    async def with_lock_async(
        self,
        fn: Callable[[str | None], Awaitable[tuple[Any, str | None]]],
        options: AuthOperationOptions | None = None,
    ) -> Any:
        """The callback is async, so this cannot be offloaded as one unit the
        way `with_lock` can — each filesystem step goes to the pool on its own,
        including the lock release in the `finally`."""
        cancel = options.cancel if options is not None else None
        if cancel is not None:
            cancel.raise_if_cancelled()
        done = tonio.Event()
        # The tail swap is pi's synchronous promise chaining; on worker
        # threads it needs a lock or two callers can link behind the same
        # predecessor and run concurrently.
        with self._chain_guard:
            previous = self._async_chain
            self._async_chain = done

        async def _operation() -> Any:
            try:
                if previous is not None:
                    await previous.wait()
                if cancel is not None:
                    cancel.raise_if_cancelled()
                await tonio.spawn_blocking(self._ensure_ready)

                release = await _acquire_lock_async(self._auth_path, cancel)
                try:
                    if cancel is not None:
                        cancel.raise_if_cancelled()
                    current = await tonio.spawn_blocking(self._read)
                    result, next_content = await fn(current)
                    if cancel is not None:
                        cancel.raise_if_cancelled()
                    if next_content is not None:
                        await tonio.spawn_blocking(self._write, next_content)
                    return result
                finally:
                    await tonio.spawn_blocking(release)
            finally:
                done.set()

        return await race_with_cancel(_operation(), cancel)


class ReadOnlyAuthStorage(CredentialStore):
    """One-shot read-only view of auth.json: strict validation, no locks, no
    persistence, refusal to write."""

    def __init__(self, auth_path: str | None = None):
        if auth_path is None:
            auth_path = os.path.join(get_agent_dir(), "auth.json")
        self._auth_path = normalize_path(auth_path)
        self._data: dict[str, Credential] | None = None

    def _load_sync(self) -> dict[str, Credential]:
        if self._data is not None:
            return self._data
        try:
            with open(self._auth_path, encoding="utf-8") as f:
                parsed = json.load(f)
        except FileNotFoundError:
            self._data = {}
            return self._data
        except Exception as error:
            raise Exception(f"Failed to read auth.json: {error}")

        if not isinstance(parsed, dict):
            raise Exception("Invalid auth.json: expected an object")  # noqa: TRY004 - pi throws a plain Error
        data: dict[str, Credential] = {}
        for provider_id, credential in parsed.items():
            if not isinstance(credential, dict):
                raise Exception(f'Invalid auth.json credential for provider "{provider_id}"')  # noqa: TRY004
            if credential.get("type") == "api_key":
                key = credential.get("key")
                env = credential.get("env")
                valid_key = key is None or isinstance(key, str)
                valid_env = env is None or (
                    isinstance(env, dict) and all(isinstance(entry, str) for entry in env.values())
                )
                if valid_key and valid_env:
                    data[provider_id] = parse_credential(credential)
                    continue
            elif (
                credential.get("type") == "oauth"
                and isinstance(credential.get("access"), str)
                and isinstance(credential.get("refresh"), str)
                and isinstance(credential.get("expires"), int | float)
                and not isinstance(credential.get("expires"), bool)
            ):
                data[provider_id] = parse_credential(credential)
                continue
            raise Exception(f'Invalid auth.json credential for provider "{provider_id}"')
        self._data = data
        return data

    async def read(self, provider_id: str, options: AuthOperationOptions | None = None) -> Credential | None:
        if options is not None and options.cancel is not None:
            options.cancel.raise_if_cancelled()
        credential = (await tonio.spawn_blocking(self._load_sync)).get(provider_id)
        if options is not None and options.cancel is not None:
            options.cancel.raise_if_cancelled()
        if credential is None:
            return None
        # `!command` keys stay unresolved: read-only access must not execute
        # configured commands.
        if (
            not isinstance(credential, ApiKeyCredential)
            or not credential.key
            or is_command_config_value(credential.key)
        ):
            return credential
        resolved = await resolve_config_value(credential.key, credential.env)
        return ApiKeyCredential(key=resolved, env=credential.env)

    async def list(self, options: AuthOperationOptions | None = None) -> list[CredentialInfo]:
        if options is not None and options.cancel is not None:
            options.cancel.raise_if_cancelled()
        data = await tonio.spawn_blocking(self._load_sync)
        if options is not None and options.cancel is not None:
            options.cancel.raise_if_cancelled()
        return [
            CredentialInfo(provider_id=provider_id, type=credential.type) for provider_id, credential in data.items()
        ]

    async def modify(
        self,
        provider_id: str,
        fn: Callable[[Credential | None], Awaitable[Credential | None]],
        options: AuthOperationOptions | None = None,
    ) -> Credential | None:
        raise Exception("Read-only credential storage cannot modify auth.json")

    async def delete(self, provider_id: str, options: AuthOperationOptions | None = None) -> None:
        raise Exception("Read-only credential storage cannot modify auth.json")


class InMemoryAuthStorageBackend:
    def __init__(self):
        self._value: str | None = None
        # Settled marker of the last queued async operation (pi chains promises).
        self._async_chain: Any = None
        self._chain_guard = threading.Lock()

    def with_lock(self, fn: Callable[[str | None], tuple[Any, str | None]]) -> Any:
        result, next_content = fn(self._value)
        if next_content is not None:
            self._value = next_content
        return result

    async def with_lock_async(
        self,
        fn: Callable[[str | None], Awaitable[tuple[Any, str | None]]],
        options: AuthOperationOptions | None = None,
    ) -> Any:
        cancel = options.cancel if options is not None else None
        done = tonio.Event()
        with self._chain_guard:
            previous = self._async_chain
            self._async_chain = done

        async def _operation() -> Any:
            if previous is not None:
                await previous.wait()
            try:
                if cancel is not None:
                    cancel.raise_if_cancelled()
                result, next_content = await fn(self._value)
                if cancel is not None:
                    cancel.raise_if_cancelled()
                if next_content is not None:
                    self._value = next_content
                return result
            finally:
                done.set()

        return await race_with_cancel(_operation(), cancel)


type AuthStorageBackend = FileAuthStorageBackend | InMemoryAuthStorageBackend


def _serialize_data(data: dict[str, Credential]) -> str:
    return json.dumps({provider: serialize_credential(credential) for provider, credential in data.items()}, indent=2)


class AuthStorage(CredentialStore):
    """Credential storage backed by a JSON file."""

    def __init__(self, storage: AuthStorageBackend, auth_path: str | None = None):
        self._storage = storage
        self._auth_path = auth_path
        if auth_path is not None:
            self._read_state = _shared_auth_file_read_states.setdefault(auth_path, _AuthFileReadState())
        else:
            self._read_state = _AuthFileReadState()
        # No load here: reading auth.json is I/O and a constructor cannot
        # await. `create()` loads asynchronously (skipping the reload when the
        # shared revision is current); `in_memory()` loads inline because its
        # backend never touches a file.

    @staticmethod
    async def create(auth_path: str | None = None) -> AuthStorage:
        if auth_path is None:
            auth_path = os.path.join(get_agent_dir(), "auth.json")
        normalized = normalize_path(auth_path)
        storage = AuthStorage(FileAuthStorageBackend(normalized), auth_path=normalized)
        revision = await tonio.spawn_blocking(_get_file_revision, normalized)
        if revision is not None and revision == storage._read_state.revision:
            return storage
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

    def _update_read_state(self, data: dict[str, Credential], revision: str | None = None) -> None:
        self._read_state.data = data
        self._read_state.revision = revision

    def reload(self) -> None:
        """Reload credentials from storage.

        Sync, so only safe for a backend that does no file I/O — i.e. the
        in-memory one (which also means no file revision to capture).
        File-backed stores use `reload_async`.
        """
        try:
            content = self._storage.with_lock(lambda current: (current, None))
            self._update_read_state(self._parse_storage_data(content))
        except Exception:
            pass  # Preserve the last valid in-memory snapshot.

    async def _reload_from_storage_async(self, options: AuthOperationOptions | None = None) -> dict[str, Credential]:
        async def under_lock(content: str | None) -> tuple[dict[str, Credential], str | None]:
            current_data = self._parse_storage_data(content)
            revision = (
                await tonio.spawn_blocking(_get_file_revision, self._auth_path) if self._auth_path is not None else None
            )
            self._update_read_state(current_data, revision)
            return current_data, None

        return await self._storage.with_lock_async(under_lock, options)

    async def reload_async(self) -> None:
        """Reload credentials from storage without blocking a runtime worker."""
        try:
            await self._reload_from_storage_async()
        except Exception:
            pass  # Preserve the last valid in-memory snapshot.

    async def _read_latest_data(self, options: AuthOperationOptions | None = None) -> dict[str, Credential]:
        cancel = options.cancel if options is not None else None
        if cancel is not None:
            cancel.raise_if_cancelled()
        state = self._read_state
        if self._auth_path is None:
            # No file revision to cache against: reload every time (cheap for
            # the in-memory backend), swallowing failures for plain reads.
            if cancel is not None:
                return await self._reload_from_storage_async(options)
            try:
                return await self._reload_from_storage_async()
            except Exception:
                return state.data

        revision = await tonio.spawn_blocking(_get_file_revision, self._auth_path)
        if revision is not None and revision == state.revision:
            return state.data

        # Concurrent readers share one reload (a burst of per-provider reads
        # must not multiply locked reloads); each reader races only its own
        # wait, and the last departing reader aborts a reload nobody awaits.
        with state.guard:
            if state.reload is None:
                controller = CancelToken()
                reload = _AuthFileReload(controller=controller, done=tonio.Event(), box=[])
                state.reload = reload

                async def _run_reload(reload: _AuthFileReload = reload) -> None:
                    try:
                        reload.box.append(
                            (
                                "value",
                                await self._reload_from_storage_async(AuthOperationOptions(cancel=reload.controller)),
                            )
                        )
                    except BaseException as error:
                        reload.box.append(("error", error))
                    finally:
                        with state.guard:
                            if state.reload is reload:
                                state.reload = None
                        reload.done.set()

                tonio.spawn.without_tracking(_run_reload())
            reload = state.reload
            reload.readers += 1

        try:

            async def _wait(reload: _AuthFileReload = reload) -> dict[str, Credential]:
                await reload.done.wait()
                kind, payload = reload.box[0]
                if kind == "error":
                    raise payload
                return payload

            if cancel is not None:
                return await race_with_cancel(_wait(), cancel)
            try:
                return await _wait()
            except Exception:
                return state.data
        finally:
            with state.guard:
                reload.readers -= 1
                last_reader = reload.readers == 0 and state.reload is reload
                if last_reader:
                    state.reload = None
            if last_reader:
                reload.controller.cancel()

    async def read(self, provider: str, options: AuthOperationOptions | None = None) -> Credential | None:
        credential = (await self._read_latest_data(options)).get(provider)
        if options is not None and options.cancel is not None:
            options.cancel.raise_if_cancelled()
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
        options: AuthOperationOptions | None = None,
    ) -> Credential | None:
        latest: list[dict[str, Credential]] = [self._read_state.data]
        revision_box: list[str | None] = [None]

        async def under_lock(content: str | None) -> tuple[Credential | None, str | None]:
            current_data = self._parse_storage_data(content)
            next_credential = await fn(current_data.get(provider))
            if next_credential is None:
                latest[0] = current_data
                revision_box[0] = (
                    await tonio.spawn_blocking(_get_file_revision, self._auth_path)
                    if self._auth_path is not None
                    else None
                )
                return current_data.get(provider), None

            merged = {**current_data, provider: next_credential}
            latest[0] = merged
            return next_credential, _serialize_data(merged)

        result = await self._storage.with_lock_async(under_lock, options)
        self._update_read_state(latest[0], revision_box[0])
        return result

    async def delete(self, provider: str, options: AuthOperationOptions | None = None) -> None:
        latest: list[dict[str, Credential]] = [self._read_state.data]

        async def under_lock(content: str | None) -> tuple[None, str]:
            current_data = self._parse_storage_data(content)
            current_data.pop(provider, None)
            latest[0] = current_data
            return None, _serialize_data(current_data)

        await self._storage.with_lock_async(under_lock, options)
        self._update_read_state(latest[0])

    async def list(self, options: AuthOperationOptions | None = None) -> list[CredentialInfo]:
        """List credential metadata without resolving configured key values."""
        entries = await self._read_latest_data(options)
        if options is not None and options.cancel is not None:
            options.cancel.raise_if_cancelled()
        return [
            CredentialInfo(provider_id=provider_id, type=credential.type) for provider_id, credential in entries.items()
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
