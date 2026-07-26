"""Port of pi's in-memory credential store (packages/ai/src/auth/credential-store.ts).

pi serializes writes per provider through a promise chain; here each provider
id gets a tonio lock. Reads stay lock-free, mirroring pi.
"""

import threading
from collections.abc import Awaitable, Callable

from tonio.colored import sync

from pidrei_ai.auth.types import Credential, CredentialInfo, CredentialStore


class InMemoryCredentialStore(CredentialStore):
    """Default in-memory credential store. Apps inject persistent stores."""

    def __init__(self) -> None:
        self._credentials: dict[str, Credential] = {}
        self._locks: dict[str, sync.Lock] = {}
        self._guard = threading.Lock()

    def _lock_for(self, provider_id: str) -> sync.Lock:
        with self._guard:
            lock = self._locks.get(provider_id)
            if lock is None:
                lock = sync.Lock()
                self._locks[provider_id] = lock
            return lock

    async def read(self, provider_id: str) -> Credential | None:
        return self._credentials.get(provider_id)

    async def list(self) -> list[CredentialInfo]:
        return [
            CredentialInfo(provider_id=provider_id, type=credential.type)
            for provider_id, credential in self._credentials.items()
        ]

    async def modify(
        self,
        provider_id: str,
        fn: Callable[[Credential | None], Awaitable[Credential | None]],
    ) -> Credential | None:
        async with self._lock_for(provider_id):
            current = self._credentials.get(provider_id)
            updated = await fn(current)
            if updated is not None:
                self._credentials[provider_id] = updated
            return updated if updated is not None else current

    async def delete(self, provider_id: str) -> None:
        async with self._lock_for(provider_id):
            self._credentials.pop(provider_id, None)
