"""Session backend conformance contracts (port of pi `session/testing/types.ts`)."""

from collections.abc import Awaitable, Callable
from typing import Protocol

from ..types import SessionRepo


class SessionBackendFixture(Protocol):
    """A fresh backend instance owned by one conformance case."""

    @property
    def repository(self) -> SessionRepo: ...

    async def dispose(self) -> None: ...


# Creates an isolated fixture for one conformance case.
type SessionBackendFixtureFactory = Callable[[], Awaitable[SessionBackendFixture]]


class SessionBackendConformanceCase(Protocol):
    """A runner-independent conformance case that can be registered with any test framework."""

    @property
    def group(self) -> str: ...

    @property
    def name(self) -> str: ...

    def run(self) -> Awaitable[None]: ...
