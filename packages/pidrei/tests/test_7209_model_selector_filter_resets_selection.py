"""Mirror of pi's suite/regressions/7209-model-selector-filter-resets-selection.test.ts."""

from types import SimpleNamespace

import pytest
import tonio.colored as tonio

from pidrei.core.keybindings import KeybindingsManager
from pidrei.modes.interactive.components.model_selector import ModelSelectorComponent
from pidrei.modes.interactive.theme import init_theme_sync
from pidrei.utils.ansi import strip_ansi
from pidrei_tui import set_keybindings

from .harness import create_harness


DOWN = "\x1b[B"

MODELS = [
    {"id": "alpha-1", "name": "Alpha One", "reasoning": True},
    {"id": "alpha-2", "name": "Alpha Two", "reasoning": True},
    {"id": "alpha-3", "name": "Alpha Three", "reasoning": True},
    {"id": "beta-1", "name": "Beta One", "reasoning": True},
]


@pytest.fixture(autouse=True)
def _setup():
    init_theme_sync("dark")
    # Keybindings are a global singleton; reset per test.
    set_keybindings(KeybindingsManager())


def fake_tui():
    return SimpleNamespace(request_render=lambda: None)


def selected_model_id(rendered: str) -> str | None:
    """Return the model id of the highlighted (→) row in the rendered selector."""
    line = next((candidate for candidate in rendered.split("\n") if candidate.startswith("→ ")), None)
    if line is None:
        return None
    rest = line.removeprefix("→").lstrip()
    model_id = rest.split(" [")[0].strip()
    return model_id or None


def render(selector: ModelSelectorComponent) -> str:
    return strip_ansi("\n".join(selector.render(120)))


async def wait_for_refresh(selector: ModelSelectorComponent) -> None:
    while "Model catalogs refreshed." not in render(selector):
        await tonio.time.sleep(0.005)


@pytest.mark.tonio
async def test_moves_selection_to_the_first_row_in_the_all_tab_when_typing_a_query(tmp_path):
    harness = await create_harness(models=MODELS)
    try:
        current = harness.get_model("alpha-1")
        assert current is not None
        selector = ModelSelectorComponent(
            fake_tui(),
            current,
            harness.session.model_runtime,
            [],
            lambda *args: None,
            lambda *args: None,
        )

        await wait_for_refresh(selector)

        # Current model (alpha-1) is sorted first, so selection starts on row 0.
        assert selected_model_id(render(selector)) == "alpha-1"

        # Move selection down two rows to alpha-3.
        await selector.handle_input(DOWN)
        await selector.handle_input(DOWN)
        assert selected_model_id(render(selector)) == "alpha-3"

        # Type a query that matches the three alpha models. The selection must
        # move back to the top row (alpha-1), not stay clamped at index 2.
        for char in "alpha":
            await selector.handle_input(char)

        rendered = render(selector)
        assert selected_model_id(rendered) == "alpha-1"
        # Sanity: the filter actually narrowed the list.
        assert "beta-1" not in rendered
    finally:
        harness.cleanup()


@pytest.mark.tonio
async def test_moves_selection_to_the_first_row_in_the_scoped_tab_when_typing_a_query(tmp_path):
    harness = await create_harness(models=MODELS[:3])
    try:
        alpha1 = harness.get_model("alpha-1")
        alpha2 = harness.get_model("alpha-2")
        alpha3 = harness.get_model("alpha-3")
        assert alpha1 is not None and alpha2 is not None and alpha3 is not None

        # Scoped list is intentionally not in current-model-first order; the
        # current model (alpha-1) sits at index 2.
        selector = ModelSelectorComponent(
            fake_tui(),
            alpha1,
            harness.session.model_runtime,
            [{"model": alpha2}, {"model": alpha3}, {"model": alpha1}],
            lambda *args: None,
            lambda *args: None,
        )

        await wait_for_refresh(selector)

        # Selection starts on the current model (alpha-1), which is row 2 here.
        assert selected_model_id(render(selector)) == "alpha-1"

        # Type a query matching all three scoped models. Selection must move to
        # the top row (alpha-2), not stay clamped at index 2 (alpha-1).
        for char in "alpha":
            await selector.handle_input(char)

        assert selected_model_id(render(selector)) == "alpha-2"
    finally:
        harness.cleanup()
