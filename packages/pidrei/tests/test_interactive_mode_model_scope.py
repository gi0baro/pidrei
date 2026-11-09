"""Mirror of pi coding-agent test/interactive-mode-model-scope.test.ts.

pi calls the prototype methods on a plain object; here the unbound methods
get a namespace with the attributes they read. pi's `console.log` spy is a
recorder on the module's `write_stdout`.
"""

from functools import partial
from types import SimpleNamespace

import pytest

from pidrei.modes.interactive import interactive_mode
from pidrei.modes.interactive.interactive_mode import InteractiveMode
from pidrei.modes.interactive.theme import init_theme
from pidrei.utils.ansi import strip_ansi
from pidrei_tui import Container


def _create_context(show_details: bool = True) -> SimpleNamespace:
    context = SimpleNamespace(
        session=SimpleNamespace(
            scoped_models=[
                SimpleNamespace(model=SimpleNamespace(id="model-a"), thinking_level="high"),
                SimpleNamespace(model=SimpleNamespace(id="model-b"), thinking_level=None),
            ]
        ),
        _keybindings=SimpleNamespace(get_keys=lambda _keybinding: ["ctrl+p"]),
        _header_container=Container(),
        _should_show_startup_details=lambda: show_details,
    )
    context._get_model_scope_notice = partial(InteractiveMode._get_model_scope_notice, context)
    return context


# Regression test for #11508: fullscreen must not print the notice to the shell's main screen.
@pytest.mark.tonio
async def test_renders_the_model_scope_inside_the_fullscreen_header_instead_of_printing_it(monkeypatch):
    await init_theme("dark")
    printed: list[str] = []
    monkeypatch.setattr(interactive_mode, "write_stdout", printed.append)
    context = _create_context()

    InteractiveMode._show_fullscreen_model_scope_notice(context)

    assert printed == []
    rendered = strip_ansi("\n".join(context._header_container.render(120)))
    assert "Model scope: model-a:high, model-b (Ctrl+P to cycle)" in rendered


@pytest.mark.tonio
async def test_adds_nothing_when_startup_details_are_hidden():
    await init_theme("dark")
    context = _create_context(False)

    InteractiveMode._show_fullscreen_model_scope_notice(context)

    assert context._header_container.children == []
