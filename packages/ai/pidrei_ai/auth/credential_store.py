"""Port of pi's in-memory credential store (packages/ai/src/auth/credential-store.ts).

pi serializes writes per provider through a promise chain; here each provider
id gets a tonio lock. Reads stay lock-free, mirroring pi. pi's cancellation
race abandons the queued task without releasing the chain early — the
detached task here likewise finishes (and releases the lock) on its own.
"""

import threading
from collections.abc import Awaitable, Callable

from tonio.colored import sync

from pidrei_ai.auth.types import AuthOperationOptions, Credential, CredentialInfo, CredentialStore
from pidrei_ai.utils.abort import operation_cancel, race_with_cancel


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

    async def read(self, provider_id: str, options: AuthOperationOptions | None = None) -> Credential | None:
        if options is not None and options.cancel is not None:
            options.cancel.raise_if_cancelled()
        return self._credentials.get(provider_id)

    async def list(self, options: AuthOperationOptions | None = None) -> list[CredentialInfo]:
        if options is not None and options.cancel is not None:
            options.cancel.raise_if_cancelled()
        return [
            CredentialInfo(provider_id=provider_id, type=credential.type)
            for provider_id, credential in self._credentials.items()
        ]

    async def modify(
        self,
        provider_id: str,
        fn: Callable[[Credential | None], Awaitable[Credential | None]],
        options: AuthOperationOptions | None = None,
    ) -> Credential | None:
        cancel = operation_cancel(options.cancel if options is not None else None)

        async def _task() -> Credential | None:
            async with self._lock_for(provider_id):
                cancel.raise_if_cancelled()
                current = self._credentials.get(provider_id)
                updated = await fn(current)
                cancel.raise_if_cancelled()
                if updated is not None:
                    self._credentials[provider_id] = updated
                return updated if updated is not None else current

        return await race_with_cancel(_task(), cancel)

    async def delete(self, provider_id: str, options: AuthOperationOptions | None = None) -> None:
        cancel = operation_cancel(options.cancel if options is not None else None)

        async def _task() -> None:
            async with self._lock_for(provider_id):
                cancel.raise_if_cancelled()
                self._credentials.pop(provider_id, None)

        return await race_with_cancel(_task(), cancel)
