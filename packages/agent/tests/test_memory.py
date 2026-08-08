"""In-memory backend conformance + Session id-generator behavior (mirror of pi
agent/test/harness/session/memory.test.ts)."""

import pytest

from pidrei_agent.harness.session.memory import InMemorySessionRepo, InMemorySessionStorage
from pidrei_agent.harness.session.session import Session
from pidrei_agent.harness.session.testing.conformance import create_session_backend_conformance
from pidrei_agent.harness.session.types import SessionMetadata


class _InMemoryFixture:
    def __init__(self) -> None:
        self.repository = InMemorySessionRepo()

    async def dispose(self) -> None:
        pass


async def _create_fixture() -> _InMemoryFixture:
    return _InMemoryFixture()


CONFORMANCE = create_session_backend_conformance(_create_fixture)


@pytest.mark.tonio
@pytest.mark.parametrize("case", CONFORMANCE, ids=[f"{case.group}: {case.name}" for case in CONFORMANCE])
async def test_in_memory_session_repo_conformance(case):
    await case.run()


class _CountingIdGenerator:
    def __init__(self) -> None:
        self._next_id = 0

    def next(self) -> str:
        self._next_id += 1
        return f"generated-{self._next_id}"


@pytest.mark.tonio
async def test_uses_one_injectable_id_generator_across_lane_views():
    session = Session(
        InMemorySessionStorage(SessionMetadata(id="session", created_at=1)), id_generator=_CountingIdGenerator()
    )
    main_id = await session.append_custom_entry("note")
    await session.create_lane("thread", main_id)
    thread_id = await session.view("thread").append_custom_entry("note")

    assert main_id == "generated-1"
    assert thread_id == "generated-2"
