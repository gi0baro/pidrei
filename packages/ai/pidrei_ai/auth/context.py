"""Port of pi's default auth context (packages/ai/src/auth/context.ts)."""

import os
from pathlib import Path

import tonio.colored as tonio


class DefaultAuthContext:
    """Env vars from os.environ; file existence checked off the runtime workers."""

    async def env(self, name: str) -> str | None:
        value = os.environ.get(name)
        return value if isinstance(value, str) and value.strip() else None

    async def file_exists(self, path: str) -> bool:
        try:
            return await tonio.spawn_blocking(Path(path).expanduser().exists)
        except Exception:
            return False


def default_provider_auth_context() -> DefaultAuthContext:
    return DefaultAuthContext()
