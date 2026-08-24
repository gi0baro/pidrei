"""Mirror of pi's suite/regressions/8423-extension-factory-failure.test.ts.

`EventBus.emit` spawns each handler, so pi's synchronous emit-then-assert
becomes an emit, a short drain, and a host counter proving the emit was
delivered at all (same shape as `test_7193_event_bus_lifecycle`). pi's
second case parks the failing factory on a promise it resolves by hand; here
the two loads run as sibling coroutines under `tonio.spawn`, gated on a
`tonio.Event`.
"""

import os

import pytest
import tonio.colored as tonio

from pidrei.core.event_bus import EventBus
from pidrei.core.extensions.loader import create_extension_runtime, load_extension_from_factory


PROVIDER_CONFIG = {"baseUrl": "https://provider.test/v1", "apiKey": "provider-test-key"}


@pytest.mark.tonio
async def test_discards_runtime_changes_and_disables_the_failed_api():
    runtime = create_extension_runtime()
    event_bus = EventBus()
    captured: list = []
    counts = {"extension": 0, "host": 0}
    flag_during_load: list = []

    async def on_host(_data) -> None:
        counts["host"] += 1

    event_bus.on("factory-failure", on_host)

    def working_factory(pi) -> None:
        pi.register_provider("working-provider", PROVIDER_CONFIG)

    await load_extension_from_factory(working_factory, os.getcwd(), event_bus, runtime, "<working>")

    def failing_factory(pi) -> None:
        captured.append(pi)

        async def on_factory_failure(_data) -> None:
            counts["extension"] += 1

        pi.events.on("factory-failure", on_factory_failure)
        pi.register_flag("failed-flag", type="boolean", default=True)
        flag_during_load.append(pi.get_flag("failed-flag"))
        pi.unregister_provider("working-provider")
        pi.register_provider("failed-provider", PROVIDER_CONFIG)
        raise RuntimeError("factory failed")

    with pytest.raises(RuntimeError, match="factory failed"):
        await load_extension_from_factory(failing_factory, os.getcwd(), event_bus, runtime, "<failing>")

    event_bus.emit("factory-failure", None)
    await tonio.time.sleep(0.01)

    assert flag_during_load == [True]
    assert "failed-flag" not in runtime.flag_values
    assert [entry["name"] for entry in runtime.pending_provider_registrations] == ["working-provider"]
    assert counts == {"extension": 0, "host": 1}
    assert len(captured) == 1
    with pytest.raises(RuntimeError, match='Extension "<failing>" failed to load and its API is no longer active.'):
        captured[0].register_flag("late-flag", type="boolean", default=True)


@pytest.mark.tonio
async def test_does_not_discard_a_concurrently_loaded_factorys_provider():
    runtime = create_extension_runtime()
    event_bus = EventBus()
    registered = tonio.Event()
    release_failure = tonio.Event()

    async def failing_factory(pi) -> None:
        pi.register_provider("failed-provider", PROVIDER_CONFIG)
        registered.set()
        await release_failure.wait()
        raise RuntimeError("factory failed")

    async def run_failing_load() -> None:
        with pytest.raises(RuntimeError, match="factory failed"):
            await load_extension_from_factory(failing_factory, os.getcwd(), event_bus, runtime, "<failing>")

    async def load_working() -> None:
        await registered.wait()

        def working_factory(pi) -> None:
            pi.register_provider("working-provider", PROVIDER_CONFIG)

        await load_extension_from_factory(working_factory, os.getcwd(), event_bus, runtime, "<working>")
        release_failure.set()

    await tonio.spawn(run_failing_load(), load_working())

    assert [entry["name"] for entry in runtime.pending_provider_registrations] == ["working-provider"]
