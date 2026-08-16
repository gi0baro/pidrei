"""Mirror of pi coding-agent test/interactive-mode-startup-input.test.ts."""

from functools import partial
from types import SimpleNamespace

import pytest
import tonio.colored as tonio

from pidrei.modes.interactive.interactive_mode import InteractiveMode


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
    return context


@pytest.mark.tonio
async def test_queues_a_normal_prompt_submitted_before_the_input_callback_is_installed():
    context = _create_submit_context()
    InteractiveMode._setup_editor_submit_handler(context)

    context._default_editor.on_submit(" early prompt ")
    # on_submit spawns the async submit handler; let it run.
    await tonio.time.sleep(0.01)

    assert context._pending_user_inputs == ["early prompt"]
    assert context.flush_calls == [True]
    assert context.history == ["early prompt"]


@pytest.mark.tonio
async def test_returns_queued_startup_input_before_installing_a_new_input_callback():
    context = SimpleNamespace(
        _on_input_callback=None,
        _pending_user_inputs=["queued prompt"],
    )

    assert await InteractiveMode._get_user_input(context) == "queued prompt"
    assert context._on_input_callback is None
    assert context._pending_user_inputs == []


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
