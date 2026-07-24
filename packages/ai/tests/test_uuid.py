import re

import pytest
import tonio.colored as tonio

from pppi_ai.utils.uuid import uuidv7


UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


def test_format_version_and_variant():
    for _ in range(100):
        assert UUID_RE.match(uuidv7())


def test_monotonic_ordering():
    values = [uuidv7() for _ in range(2000)]

    assert values == sorted(values)


@pytest.mark.tonio
async def test_uniqueness_across_threads():
    async def generate(count: int) -> list[str]:
        return [uuidv7() for _ in range(count)]

    batches = await tonio.map(generate, [1000] * 8)

    all_values = [value for batch in batches for value in batch]
    assert len(set(all_values)) == len(all_values)
    # Per-thread batches must each be strictly increasing.
    for batch in batches:
        assert batch == sorted(batch)
