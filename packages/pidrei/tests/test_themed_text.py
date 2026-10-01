"""Mirror of pi coding-agent test/themed-text.test.ts."""

import pytest

from pidrei.modes.interactive.components.themed_text import ThemedText
from pidrei.modes.interactive.theme import init_theme, theme


@pytest.fixture(autouse=True)
async def _reset_theme():
    yield
    await init_theme("dark")


@pytest.mark.tonio
async def test_builds_lazily_and_rebuilds_with_the_current_theme_after_invalidation():
    await init_theme("dark")
    builds = 0

    def build() -> str:
        nonlocal builds
        builds += 1
        return theme.fg("accent", "hello")

    text = ThemedText(build)
    assert builds == 0
    dark = "".join(text.render(20))

    await init_theme("light")
    assert "".join(text.render(20)) == dark
    text.invalidate()
    assert theme.get_fg_ansi("accent") in "".join(text.render(20))
    assert builds == 2
