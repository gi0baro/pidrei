"""Mirror of pi's suite/regressions/7290-json-stream-linear.test.ts.

Byte size is measured on the serialized wire form (`to_wire` + `json.dumps`),
which is what print/RPC mode actually writes.
"""

import json

import pytest

from pidrei.core.json_wire import to_wire
from pidrei.modes.json_event import to_json_event
from pidrei_ai.providers.faux import faux_assistant_message

from .harness import create_harness


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


async def _measure_update_bytes(harnesses, text: str) -> int:
    harness = await create_harness()
    harnesses.append(harness)
    harness.set_responses([faux_assistant_message(text)])

    await harness.session.prompt("respond")

    session_updates = harness.events_of_type("message_update")
    for update in session_updates:
        assert getattr(update, "message", None) is not None
        assert hasattr(update.assistant_message_event, "partial")

    updates = [to_json_event(event) for event in session_updates]
    assert len(updates) > 0
    for update in updates:
        # `usage` joined the wire shape in 0.84.2 (c93ea6cc): fixed-size, so
        # the linear-scaling property below still holds.
        assert set(update) == {"type", "usage", "assistantMessageEvent"}
        assert "partial" not in update["assistantMessageEvent"]
    return sum(len(json.dumps(to_wire(update), ensure_ascii=False).encode("utf-8")) for update in updates)


@pytest.mark.tonio
async def test_emits_delta_only_message_updates_whose_size_scales_linearly(harnesses):
    small_bytes = await _measure_update_bytes(harnesses, "x" * 2_000)
    large_bytes = await _measure_update_bytes(harnesses, "x" * 4_000)

    assert large_bytes > small_bytes
    assert large_bytes / small_bytes < 2.2
