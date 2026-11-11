"""Mirror of pi coding-agent test/extension-selector.test.ts."""

import pytest

from pidrei.core.keybindings import KeybindingsManager
from pidrei.modes.interactive.components.extension_selector import ExtensionSelectorComponent
from pidrei.modes.interactive.theme import init_theme
from pidrei.utils.ansi import strip_ansi
from pidrei_tui import set_keybindings


@pytest.fixture(autouse=True)
async def _setup():
    await init_theme("dark")
    # Keybindings are a global singleton; reset per test.
    set_keybindings(KeybindingsManager.in_memory())


def _render(selector: ExtensionSelectorComponent) -> str:
    return strip_ansi("\n".join(selector.render(80)))


def _visible_options(output: str) -> list[str]:
    return [line.strip().removeprefix("→ ") for line in output.split("\n") if "option-" in line]


# #6688: long extension selectors must keep the selected option visible.
def test_windows_long_option_lists_around_the_selection():
    options = [f"option-{index + 1}" for index in range(15)]
    selector = ExtensionSelectorComponent("Choose", options, lambda _option: None, lambda: None)

    output = _render(selector)
    assert _visible_options(output) == options[0:10]
    assert "→ option-1" in output
    assert "  (1/15)" in output

    for _ in range(7):
        selector.handle_input("\x1b[B")
    output = _render(selector)
    assert _visible_options(output) == options[2:12]
    assert "→ option-8" in output
    assert "  (8/15)" in output

    for _ in range(7):
        selector.handle_input("\x1b[B")
    output = _render(selector)
    assert _visible_options(output) == options[5:15]
    assert "→ option-15" in output
    assert "  (15/15)" in output


def test_renders_short_and_empty_option_lists_without_a_scroll_indicator():
    short_options = ["option-1", "option-2", "option-3"]
    short_output = _render(ExtensionSelectorComponent("Choose", short_options, lambda _option: None, lambda: None))
    assert _visible_options(short_output) == short_options
    assert "→ option-1" in short_output
    assert "(1/3)" not in short_output

    empty_output = _render(ExtensionSelectorComponent("Choose", [], lambda _option: None, lambda: None))
    assert "Choose" in empty_output
    assert "↑↓ navigate" in empty_output
    assert "→" not in empty_output
