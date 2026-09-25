"""Mirror of pi coding-agent test/interactive-mode-startup-input.test.ts."""

import threading
from functools import partial
from types import SimpleNamespace

import pytest
import tonio.colored as tonio

from pidrei.modes.interactive.interactive_mode import InteractiveMode
from pidrei_tui._owner import OwnerTask


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
    # Editor mutations route through the owner helpers; an unstarted owner
    # exercises their direct-call fallback.
    context.ui = SimpleNamespace(input_owner=OwnerTask(), request_render=lambda force=False: None)
    context._post_editor_mutation = partial(InteractiveMode._post_editor_mutation, context)
    context._set_editor_text = partial(InteractiveMode._set_editor_text, context)
    context._add_editor_history = partial(InteractiveMode._add_editor_history, context)
    return context


@pytest.mark.tonio
async def test_queues_a_normal_prompt_submitted_before_the_input_callback_is_installed():
    context = _create_submit_context()
    # on_submit spawns the async submit handler detached; the test waits for
    # that handler to finish.
    handled = tonio.Event()
    handle_editor_submit = context._handle_editor_submit

    async def handle_and_signal(text: str) -> None:
        try:
            await handle_editor_submit(text)
        finally:
            handled.set()

    context._handle_editor_submit = handle_and_signal
    InteractiveMode._setup_editor_submit_handler(context)

    context._default_editor.on_submit(" early prompt ")
    await handled.wait(5)
    assert handled.is_set()

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
