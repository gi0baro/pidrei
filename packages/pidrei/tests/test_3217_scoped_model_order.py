"""Mirror of pi's suite/regressions/3217-scoped-model-order.test.ts.

pi waits for the catalog refresh with `vi.waitFor`; here the render output is
polled the way the other model-selector mirrors do — but bounded, so a broken
refresh fails the test instead of hanging the run.
"""

from types import SimpleNamespace

import pytest
import tonio.colored as tonio

from pidrei.core.keybindings import KeybindingsManager
from pidrei.modes.interactive.components.model_selector import ModelSelectorComponent
from pidrei.modes.interactive.components.scoped_models_selector import ScopedModelsSelectorComponent
from pidrei.modes.interactive.theme import init_theme_sync
from pidrei.utils.ansi import strip_ansi
from pidrei_tui import set_keybindings

from .harness import create_harness


async def _wait_until(condition, timeout=2.0):
    waited = 0.0
    while not condition():
        await tonio.time.sleep(0.005)
        waited += 0.005
        if waited >= timeout:
            raise AssertionError("condition not reached before timeout")


THREE_MODELS = [
    {"id": "faux-1", "name": "One", "reasoning": True},
    {"id": "faux-2", "name": "Two", "reasoning": True},
    {"id": "faux-3", "name": "Three", "reasoning": True},
]


@pytest.fixture(autouse=True)
def _setup():
    init_theme_sync("dark")
    # Keybindings are a global singleton; reset per test.
    set_keybindings(KeybindingsManager())


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


def fake_tui():
    return SimpleNamespace(request_render=lambda: None)


@pytest.mark.tonio
async def test_propagates_reordered_scoped_models_back_to_the_session_state(harnesses):
    harness = await create_harness(models=THREE_MODELS)
    harnesses.append(harness)

    ordered_ids = [f"{model.provider}/{model.id}" for model in harness.models]
    changes: list = []
    selector = ScopedModelsSelectorComponent(
        {"allModels": list(harness.models), "enabledModelIds": ordered_ids},
        {
            "onChange": changes.append,
            "onPersist": lambda *args: None,
            "onCancel": lambda *args: None,
        },
    )

    await selector.handle_input("\x1b[1;3B")

    assert changes == [[ordered_ids[1], ordered_ids[0], ordered_ids[2]]]


@pytest.mark.tonio
async def test_preserves_scoped_model_order_in_the_model_scoped_tab(harnesses):
    harness = await create_harness(models=THREE_MODELS)
    harnesses.append(harness)

    model_one = harness.get_model("faux-1")
    model_two = harness.get_model("faux-2")
    model_three = harness.get_model("faux-3")
    selector = ModelSelectorComponent(
        fake_tui(),
        model_one,
        harness.session.model_runtime,
        [{"model": model_two}, {"model": model_one}, {"model": model_three}],
        lambda *args: None,
        lambda *args: None,
    )

    def render() -> str:
        return strip_ansi("\n".join(selector.render(120)))

    await _wait_until(lambda: f"[{model_one.provider}]" in render() and "Model catalogs refreshed." in render())

    rendered_lines = [line for line in render().split("\n") if f"[{model_one.provider}]" in line]
    ordered_ids = [line.strip().removeprefix("→").strip().split(" [")[0].strip() for line in rendered_lines[:3]]

    assert ordered_ids == [model_two.id, model_one.id, model_three.id]
