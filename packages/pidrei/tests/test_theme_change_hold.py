"""pidrei-specific: a theme change swaps the global theme inside the host's
change callback (UI_ISLAND_DESIGN §4.5c).

pi's `setTheme` swaps the theme and notifies in one synchronous block. Here
every component reads the global theme while rendering under the UI state
lock, so the host runs the swap under that lock together with its refresh:
a frame never mixes the old theme and the new one.
"""

import importlib

import pytest

from pidrei.modes.interactive.theme import init_theme, theme


# The package re-exports the `theme` proxy under the module's own name.
theme_module = importlib.import_module("pidrei.modes.interactive.theme.theme")


@pytest.fixture
def change_callback():
    previous = theme_module._on_theme_change_callback
    seen: list = []

    def on_change(apply) -> None:
        seen.append(("before", theme.name))
        seen.append(("apply", apply()))
        seen.append(("after", theme.name))

    theme_module.on_theme_change(on_change)
    yield seen
    theme_module.on_theme_change(previous)


@pytest.mark.tonio
async def test_set_theme_swaps_inside_the_registered_change_callback(change_callback):
    await init_theme("dark")

    assert (await theme_module.set_theme("light"))["success"] is True

    assert change_callback == [("before", "dark"), ("apply", True), ("after", "light")]


@pytest.mark.tonio
async def test_a_failed_theme_falls_back_to_dark_inside_the_callback_without_a_refresh(change_callback):
    # pi's fallback swaps to dark without notifying: `apply` says so.
    await init_theme("light")

    result = await theme_module.set_theme("no-such-theme")

    assert result["success"] is False
    assert change_callback == [("before", "light"), ("apply", False), ("after", "dark")]


@pytest.mark.tonio
async def test_set_theme_swaps_directly_with_no_callback_registered():
    previous = theme_module._on_theme_change_callback
    theme_module.on_theme_change(None)
    try:
        await init_theme("dark")
        await theme_module.set_theme("light")
        assert theme.name == "light"
    finally:
        theme_module.on_theme_change(previous)
