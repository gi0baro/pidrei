"""Mirror of pi coding-agent test/interactive-mode-startup-input.test.ts."""

import threading
from functools import partial
from types import SimpleNamespace

import pytest

from pidrei.core.extensions import Extension, ExtensionRunner, MarkdownTransformerRegistration
from pidrei.core.settings_manager import SettingsManager
from pidrei.modes.interactive.interactive_mode import InteractiveMode
from pidrei.modes.interactive.theme import init_theme
from pidrei.utils.ansi import strip_ansi
from pidrei_tui import Container


def _create_submit_context():
    context = SimpleNamespace(
        _default_editor=SimpleNamespace(on_submit=None),
        session=SimpleNamespace(
            is_compacting=False,
            is_streaming=False,
            is_bash_running=False,
        ),
        _on_input_callback=None,
        _pending_user_inputs=[],
        _user_input_guard=threading.Lock(),
        flush_calls=[],
        history=[],
        set_text_calls=[],
    )
    context.editor = SimpleNamespace(
        add_to_history=context.history.append,
        set_text=context.set_text_calls.append,
    )
    context._flush_pending_bash_components = lambda: context.flush_calls.append(True)
    context._handle_editor_submit = partial(InteractiveMode._handle_editor_submit, context)
    context.ui = SimpleNamespace(state_lock=threading.RLock(), request_render=lambda force=False: None)
    context._set_editor_text = partial(InteractiveMode._set_editor_text, context)
    context._apply_editor_history = partial(InteractiveMode._apply_editor_history, context)
    return context


@pytest.mark.tonio
async def test_queues_a_normal_prompt_submitted_before_the_input_callback_is_installed():
    context = _create_submit_context()
    InteractiveMode._setup_editor_submit_handler(context)

    context._default_editor.on_submit(" early prompt ")

    assert context._pending_user_inputs == ["early prompt"]
    assert context.flush_calls == [True]
    assert context.history == ["early prompt"]


@pytest.mark.tonio
async def test_returns_queued_startup_input_before_installing_a_new_input_callback():
    context = SimpleNamespace(
        _on_input_callback=None,
        _pending_user_inputs=["queued prompt"],
        _user_input_guard=threading.Lock(),
    )

    assert await InteractiveMode._get_user_input(context) == "queued prompt"
    assert context._on_input_callback is None
    assert context._pending_user_inputs == []


@pytest.mark.tonio
async def test_startup_submit_wiring_passes_the_editor_text_through():
    """pidrei-specific regression (macOS CI, 0.85.1): the startup `on_submit`
    was wrapped in `sync_action`, whose handler takes no arguments, so the
    editor's `on_submit(text)` call raised and killed the input pump."""
    statuses: list[str] = []
    set_texts: list[str] = []
    actions: dict = {}
    context = SimpleNamespace(
        _default_editor=SimpleNamespace(on_action=actions.__setitem__, on_ctrl_d=None, on_submit=None),
        editor=SimpleNamespace(set_text=set_texts.append),
        show_status=statuses.append,
        _handle_ctrl_c=lambda: None,
        _handle_ctrl_d=lambda: None,
    )
    context._handle_startup_submit = partial(InteractiveMode._handle_startup_submit, context)

    InteractiveMode._setup_startup_input_handlers(context)
    context._default_editor.on_submit("early prompt")

    assert set_texts == ["early prompt"]
    assert statuses == ["Startup is still in progress"]
    assert set(actions) == {"app.clear"}


@pytest.mark.tonio
async def test_restores_a_prompt_submitted_while_managed_tool_setup_is_running():
    statuses: list[str] = []
    set_texts: list[str] = []
    context = SimpleNamespace(
        editor=SimpleNamespace(set_text=set_texts.append),
        show_status=statuses.append,
    )

    InteractiveMode._handle_startup_submit(context, "early prompt")

    assert set_texts == ["early prompt"]
    assert statuses == ["Startup is still in progress"]


# Startup notices: the two "InteractiveMode startup notices" cases of pi's
# interactive-mode-extension-changelog.test.ts (a dropped file: PiDrei has no
# "Extension Updates" notice), with their What's New assertions only.


def _extension_with_transformer(transformer, message_types) -> Extension:
    return Extension(
        path="<inline:test>",
        markdown_transformer=MarkdownTransformerRegistration(
            transformer=transformer, message_types=frozenset(message_types)
        ),
    )


def _render_notices(extensions: list[Extension]) -> str:
    # get_markdown_transformers only reads the extensions' transformer registrations.
    runner = SimpleNamespace(_extensions=extensions)
    context = SimpleNamespace(
        _startup_notices_shown=False,
        _changelog_markdown="## [2.0.0]\n\n- :warning: PiDrei change",
        _version="2.0.0",
        _chat_container=Container(),
        settings_manager=SettingsManager.in_memory(),
        session=SimpleNamespace(
            extension_runner=SimpleNamespace(
                get_markdown_transformers=partial(ExtensionRunner.get_markdown_transformers, runner)
            )
        ),
        ui=SimpleNamespace(state_lock=threading.RLock()),
    )
    context._get_markdown_transformers = partial(InteractiveMode._get_markdown_transformers, context)
    context._get_markdown_theme_with_settings = partial(InteractiveMode._get_markdown_theme_with_settings, context)
    context._create_changelog_markdown = partial(InteractiveMode._create_changelog_markdown, context)
    InteractiveMode._show_startup_notices_if_needed(context)
    return strip_ansi("\n".join(context._chat_container.render(120)))


def _replace_shortcodes(markdown, _context):
    return markdown.replace(":warning:", "⚠")


@pytest.mark.tonio
async def test_runs_changelog_transformers_on_whats_new():
    await init_theme("dark")
    rendered = _render_notices([_extension_with_transformer(_replace_shortcodes, ["changelog"])])

    assert "⚠ PiDrei change" in rendered
    assert ":warning:" not in rendered


@pytest.mark.tonio
async def test_skips_transformers_registered_for_the_default_message_types():
    await init_theme("dark")
    rendered = _render_notices(
        [_extension_with_transformer(_replace_shortcodes, ["user", "assistant", "assistant-thinking"])]
    )

    assert ":warning: PiDrei change" in rendered
