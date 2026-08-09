"""Mirror of pi's suite/regressions/7193-event-bus-lifecycle.test.ts.

`EventBus.emit` spawns each handler, so pi's `setImmediate` drain becomes a
short real sleep (the plugin's `sleep(0)` would not let the spawned handler
run).
"""

import os

import pytest
import tonio.colored as tonio

from pidrei.core.agent_session import ExtensionBindings
from pidrei.core.event_bus import EventBus
from pidrei.core.extensions.loader import create_extension_runtime, load_extension_from_factory
from pidrei.core.extensions.types import LoadExtensionsResult

from .harness import create_harness


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


@pytest.mark.tonio
async def test_removes_extension_owned_event_bus_listeners_on_reload_and_dispose(harnesses):
    event_bus = EventBus()
    counts = {"extension": 0, "host": 0}
    first_api: list = []

    def factory(pi) -> None:
        if not first_api:
            first_api.append(pi)

        async def on_reload_test(_data) -> None:
            counts["extension"] += 1

        pi.events.on("reload:test", on_reload_test)

    async def on_host(_data) -> None:
        counts["host"] += 1

    event_bus.on("reload:test", on_host)

    state: dict = {}

    async def load_extensions() -> LoadExtensionsResult:
        runtime = create_extension_runtime()
        extension = await load_extension_from_factory(factory, os.getcwd(), event_bus, runtime)
        return LoadExtensionsResult(extensions=[extension], runtime=runtime)

    state["extensions"] = await load_extensions()

    class _ResourceLoader:
        def get_extensions(self):
            return state["extensions"]

        def get_skills(self):
            from pidrei.core.skills import LoadSkillsResult

            return LoadSkillsResult(skills=[], diagnostics=[])

        def get_prompts(self):
            from pidrei.core.resource_loader import LoadPromptsResult

            return LoadPromptsResult(prompts=[], diagnostics=[])

        def get_agents_files(self):
            return []

        def get_system_prompt(self):
            return None

        def get_append_system_prompt(self):
            return []

        def extend_resources(self, *_args, **_kwargs):
            pass

        async def reload(self, **_kwargs):
            state["extensions"] = await load_extensions()

    harness = await create_harness(resource_loader=_ResourceLoader())
    harnesses.append(harness)
    await harness.session.bind_extensions(ExtensionBindings())

    async def emit() -> dict:
        before = dict(counts)
        event_bus.emit("reload:test", None)
        await tonio.time.sleep(0.01)
        return {key: counts[key] - before[key] for key in counts}

    assert await emit() == {"extension": 1, "host": 1}

    await harness.session.reload()
    with pytest.raises(Exception, match="stale after session replacement or reload"):
        first_api[0].get_commands()
    assert await emit() == {"extension": 1, "host": 1}

    await harness.session.reload()
    assert await emit() == {"extension": 1, "host": 1}

    harness.session.dispose()
    assert await emit() == {"extension": 0, "host": 1}
