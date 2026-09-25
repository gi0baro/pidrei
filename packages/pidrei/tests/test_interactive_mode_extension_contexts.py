"""pidrei-specific: the contexts interactive mode builds for extension
shortcut handlers and for in-session project-trust prompts.

pi builds both as object literals typed `ExtensionContext` /
`ProjectTrustContext`; their consumers use attribute access (`ctx.ui`,
`ctx.cwd`, `ctx.has_ui`), which a dict translation broke.
"""

from types import SimpleNamespace

import pytest
import tonio.colored as tonio

from pidrei.core.project_trust import _select_project_trust_option
from pidrei.modes.interactive.interactive_mode import InteractiveMode


@pytest.mark.tonio
async def test_a_shortcut_handler_gets_an_attribute_context():
    ui = SimpleNamespace(name="extension-ui")
    received = tonio.Result()
    handled = tonio.Event()

    async def handler(ctx) -> None:
        received.store((ctx.ui, ctx.cwd, ctx.has_ui, ctx.mode))
        handled.set()

    editor = SimpleNamespace(on_extension_shortcut=None)
    fake = SimpleNamespace(
        _keybindings=SimpleNamespace(get_effective_config=dict),
        _default_editor=editor,
        _create_extension_ui_context=lambda: ui,
        session_manager=SimpleNamespace(get_cwd=lambda: "/project"),
        session=SimpleNamespace(model=None, scoped_models=[], thinking_level="off", agent=SimpleNamespace(signal=None)),
        show_error=lambda message: received.store(("error", message)) or handled.set(),
    )
    runner = SimpleNamespace(
        get_shortcuts=lambda _config: {"ctrl+x": SimpleNamespace(handler=handler)},
        get_model_registry=lambda: None,
    )

    InteractiveMode._setup_extension_shortcuts(fake, runner)
    assert editor.on_extension_shortcut("\x18") is True
    await handled.wait(5)

    assert received.fetch() == (ui, "/project", True, "tui")


@pytest.mark.tonio
async def test_the_in_session_trust_prompt_reads_its_context_by_attribute():
    # `/resume` into an untrusted project: core's trust flow asks through
    # `ctx.ui.select` on the context interactive mode built.
    prompts: list = []

    async def select(title, options, _opts=None):
        prompts.append(title)
        return options[0]

    fake = SimpleNamespace(
        _create_extension_ui_context=lambda: SimpleNamespace(select=select, confirm=None, input=None, notify=None)
    )
    ctx = InteractiveMode._create_project_trust_context(fake, "/project")

    assert ctx.has_ui is True
    option = await _select_project_trust_option("/project", ctx)
    assert option is not None and option.label == "Trust"
    assert len(prompts) == 1
