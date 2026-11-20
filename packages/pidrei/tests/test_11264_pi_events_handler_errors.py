"""Mirror of pi's suite/regressions/11264-pi-events-handler-errors.test.ts.

#11264: the event bus printed a raising pi.events handler to stderr, which draws over the interactive
TUI. Such errors now go to the extension error listener like other extension errors.

pi drains with `setImmediate`; here the waits are Events: the error listener sets one, and in the
dispose case the runtime's `report_error` is wrapped to set one once it has decided. Pi's
`console.error` spy is a recorder swapped in for the event bus's `write_stderr`. The harness binds
the extensions once, with the error listener, where pi calls `bindExtensions` itself.
"""

import os

import pytest
import tonio.colored as tonio

import pidrei.core.event_bus as event_bus_module
from pidrei.core.agent_session import ExtensionBindings
from pidrei.core.event_bus import EventBus
from pidrei.core.extensions.loader import create_extension_runtime, load_extension_from_factory
from pidrei.core.extensions.types import LoadExtensionsResult

from .agent_session_helpers import create_test_resource_loader
from .harness import create_harness


_WAIT_S = 5


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


@pytest.fixture
def stderr_writes(monkeypatch):
    writes: list[str] = []
    monkeypatch.setattr(event_bus_module, "write_stderr", writes.append)
    return writes


async def _setup(harnesses):
    event_bus = EventBus()
    late_started = tonio.Event()
    release = tonio.Event()

    async def factory(pi) -> None:
        async def on_boom(_data):
            raise Exception("handler failed")

        async def on_late(_data):
            late_started.set()
            await release.wait(_WAIT_S)
            raise Exception("late failure")

        pi.events.on("test:boom", on_boom)
        pi.events.on("test:late", on_late)

    runtime = create_extension_runtime()
    extension = await load_extension_from_factory(factory, os.getcwd(), event_bus, runtime, "<boom>")
    errors: list = []
    reported = tonio.Event()

    def on_error(error) -> None:
        errors.append(error)
        reported.set()

    harness = await create_harness(
        resource_loader=create_test_resource_loader(LoadExtensionsResult(extensions=[extension], runtime=runtime)),
        extension_bindings=ExtensionBindings(on_error=on_error),
    )
    harnesses.append(harness)
    return event_bus, runtime, harness, errors, reported, late_started, release


@pytest.mark.tonio
async def test_reports_a_raising_handler_to_the_extension_error_listener_instead_of_stderr(harnesses, stderr_writes):
    event_bus, _runtime, _harness, errors, reported, _late_started, _release = await _setup(harnesses)

    event_bus.emit("test:boom", None)
    await reported.wait(_WAIT_S)

    assert [(error.extension_path, error.event, error.error) for error in errors] == [
        ("<boom>", "events:test:boom", "handler failed")
    ]
    assert stderr_writes == []


@pytest.mark.tonio
async def test_drops_an_error_from_a_handler_that_fails_after_its_session_was_disposed(harnesses, stderr_writes):
    event_bus, runtime, harness, errors, _reported, late_started, release = await _setup(harnesses)
    report_error = runtime.report_error
    decided = tonio.Event()

    def recording_report_error(error) -> None:
        try:
            report_error(error)
        finally:
            decided.set()

    runtime.report_error = recording_report_error

    event_bus.emit("test:late", None)
    await late_started.wait(_WAIT_S)
    assert late_started.is_set()
    harness.session.dispose()
    release.set()
    await decided.wait(_WAIT_S)

    assert decided.is_set()
    assert errors == []
    assert stderr_writes == []
