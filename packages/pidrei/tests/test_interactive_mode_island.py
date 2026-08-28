"""pidrei-specific: the TUI-island wiring of interactive mode's agent listener
(PROPER_MT_DESIGN.md step 1).

No pi counterpart: pi's single JS thread makes every listener owner-side for
free. Here `_route_event` posts each session event's application to the UI
owner (`TuiBase.post_ui`) and `_settle_ui_after_agent_event` is the
per-agent-event owner barrier that keeps the fused emit contract
("listeners settled" ⇒ UI updated).
"""

from types import SimpleNamespace

import pytest
import tonio.colored as tonio

from pidrei.modes.interactive.interactive_mode import InteractiveMode
from pidrei_tui._owner import OwnerTask
from pidrei_tui.tui import TuiBase


class _Ui:
    """The slice of a TUI the routing touches, with the real `post_ui`."""

    post_ui = TuiBase.post_ui

    def __init__(self, owner: OwnerTask) -> None:
        self.input_owner = owner


def _fake_mode(owner: OwnerTask):
    applied: list = []
    fake = SimpleNamespace(applied=applied, ui=_Ui(owner))
    fake._handle_event = applied.append
    return fake


@pytest.mark.tonio
async def test_routed_events_apply_on_the_owner_in_order_before_the_barrier_settles():
    owner = OwnerTask()
    async with tonio.scope() as scope:
        owner.start(scope)
        fake = _fake_mode(owner)

        events = [f"event-{i}" for i in range(5)]
        for event in events:
            InteractiveMode._route_event(fake, event)
        # Nothing is applied inline on this (dispatcher-shaped) task...
        assert fake.applied == []

        # ...but the awaited barrier means everything routed during the emit
        # has been applied when it returns — the fused contract.
        await InteractiveMode._settle_ui_after_agent_event(fake, None)
        assert fake.applied == events

        owner.close()


@pytest.mark.tonio
async def test_events_routed_before_the_owner_starts_apply_at_start_in_order():
    # `post` always enqueues: work posted before start() runs when the owner
    # starts, still in order.
    owner = OwnerTask()
    fake = _fake_mode(owner)
    InteractiveMode._route_event(fake, "early-one")
    InteractiveMode._route_event(fake, "early-two")
    assert fake.applied == []

    async with tonio.scope() as scope:
        owner.start(scope)
        await InteractiveMode._settle_ui_after_agent_event(fake, None)
        assert fake.applied == ["early-one", "early-two"]
        owner.close()


@pytest.mark.tonio
async def test_barrier_orders_after_mutations_posted_by_helper_wrappers():
    """A wrapped helper (`ui.post_ui`) called while an event applies posts
    behind the event's job; the barrier still settles after it — order is
    the post order, on one task."""
    owner = OwnerTask()
    async with tonio.scope() as scope:
        owner.start(scope)
        order: list = []
        fake = SimpleNamespace(ui=_Ui(owner))

        def handle_event(event) -> None:
            order.append(("event", event))
            if event == "first":
                fake.ui.post_ui(lambda: order.append(("helper", event)))

        fake._handle_event = handle_event

        InteractiveMode._route_event(fake, "first")
        InteractiveMode._route_event(fake, "second")
        await InteractiveMode._settle_ui_after_agent_event(fake, None)
        assert order == [("event", "first"), ("event", "second"), ("helper", "first")]

        owner.close()
