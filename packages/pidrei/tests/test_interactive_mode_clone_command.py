"""Mirror of pi coding-agent test/interactive-mode-clone-command.test.ts.

pi calls InteractiveMode.prototype.handleCloneCommand with a fake `this`;
here the unbound method runs against a SimpleNamespace context.
"""

from types import SimpleNamespace

import pytest

from pidrei.modes.interactive.interactive_mode import InteractiveMode


def _create_context(leaf_id, fork_calls):
    async def fork(entry_id, *, position="before"):
        fork_calls.append((entry_id, position))
        return {"cancelled": False}

    return SimpleNamespace(
        session_manager=SimpleNamespace(get_leaf_id=lambda: leaf_id),
        runtime_host=SimpleNamespace(fork=fork),
        render_current_session_state_calls=[],
        editor=SimpleNamespace(set_text_calls=[]),
        show_status_calls=[],
        show_error_calls=[],
        request_render_calls=[],
    )


def _wire_recorders(context):
    context.editor.set_text = context.editor.set_text_calls.append
    context.show_status = context.show_status_calls.append
    context.show_error = context.show_error_calls.append
    context.ui = SimpleNamespace(request_render=lambda: context.request_render_calls.append(True))
    return context


@pytest.mark.tonio
async def test_clones_the_current_leaf_into_a_new_session():
    fork_calls: list = []
    context = _wire_recorders(_create_context("leaf-123", fork_calls))

    await InteractiveMode.handle_clone_command(context)

    assert fork_calls == [("leaf-123", "at")]
    assert context.editor.set_text_calls == [""]
    assert context.show_status_calls == ["Cloned to new session"]
    assert context.show_error_calls == []
    assert context.request_render_calls == []


@pytest.mark.tonio
async def test_shows_a_status_message_when_there_is_nothing_to_clone():
    fork_calls: list = []
    context = _wire_recorders(_create_context(None, fork_calls))

    await InteractiveMode.handle_clone_command(context)

    assert fork_calls == []
    assert context.show_status_calls == ["Nothing to clone yet"]
    assert context.show_error_calls == []
