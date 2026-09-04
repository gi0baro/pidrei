"""Mirror of pi ai/test/uuid.test.ts, plus the pidrei-own thread-safety case.

pi's fake timers become a patched `_now_ms` seam and `vi.stubGlobal("crypto")`
a patched `_random_bytes` seam; both are plain sync tests, so `monkeypatch`
is fine here.
"""

import math
import re

import pytest
import tonio.colored as tonio

from pidrei_ai.utils import uuid as uuid_module
from pidrei_ai.utils.uuid import uuidv7


UUID_V7_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
TIMESTAMP = 0x0123456789AB


def parse_timestamp(uuid: str) -> int:
    return int(uuid.replace("-", "")[:12], 16)


def test_generates_ordered_uuidv7s_while_preserving_follower_timestamps(monkeypatch):
    clock = [TIMESTAMP]
    monkeypatch.setattr(uuid_module, "_now_ms", lambda: clock[0])
    # vitest isolates the module per file; here earlier tests already moved
    # the generator's clock floor past the pinned TIMESTAMP.
    monkeypatch.setattr(uuid_module, "_last_ordinary_timestamp", -1)

    first = uuidv7()
    second = uuidv7()
    clock[0] = TIMESTAMP - 1
    after_rollback = uuidv7()
    clock[0] = TIMESTAMP + 1
    after_advance = uuidv7()
    ordinary_ids = [first, second, after_rollback, after_advance]
    follower_timestamp = TIMESTAMP - 1_000
    followers = [uuidv7(follower_timestamp), uuidv7(follower_timestamp)]

    for value in [*ordinary_ids, *followers]:
        assert UUID_V7_RE.match(value)
    assert ordinary_ids == sorted(ordinary_ids)
    assert len(set(ordinary_ids)) == len(ordinary_ids)
    assert [parse_timestamp(value) for value in ordinary_ids] == [TIMESTAMP, TIMESTAMP, TIMESTAMP, TIMESTAMP + 1]
    assert [parse_timestamp(value) for value in followers] == [follower_timestamp, follower_timestamp]
    assert len(set(followers)) == len(followers)


def test_uses_fresh_randomness_for_every_uuid_tail(monkeypatch):
    random_byte = [0]

    def fill(count: int) -> bytes:
        random_byte[0] += 1
        return bytes([random_byte[0]]) * count

    monkeypatch.setattr(uuid_module, "_random_bytes", fill)

    assert [uuidv7(TIMESTAMP)[-8:], uuidv7(TIMESTAMP)[-8:]] == ["01010101", "02020202"]


@pytest.mark.parametrize("timestamp", [0, 2**48 - 1])
def test_accepts_timestamp_boundary(timestamp):
    assert parse_timestamp(uuidv7(timestamp)) == timestamp


@pytest.mark.parametrize("timestamp", [-1, 2**48, 1.5, math.nan, math.inf])
def test_rejects_invalid_timestamp(timestamp):
    with pytest.raises(ValueError):
        uuidv7(timestamp)


def test_format_version_and_variant():
    for _ in range(100):
        assert UUID_V7_RE.match(uuidv7())


@pytest.mark.tonio
async def test_uniqueness_across_threads():
    """pidrei-own: the generator state is lock-guarded on the free-threaded build."""

    async def generate(count: int) -> list[str]:
        return [uuidv7() for _ in range(count)]

    batches = await tonio.map(generate, [1000] * 8)

    all_values = [value for batch in batches for value in batch]
    assert len(set(all_values)) == len(all_values)
    # Per-thread batches must each be strictly increasing.
    for batch in batches:
        assert batch == sorted(batch)
