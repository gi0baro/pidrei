"""Mirror of pi coding-agent test/interactive-mode-import-command.test.ts."""

from functools import partial
from types import SimpleNamespace

import pytest

from pidrei.core.agent_session_runtime import SessionImportFileNotFoundError
from pidrei.modes.interactive.interactive_mode import InteractiveMode


_get_path_command_argument = partial(InteractiveMode._get_path_command_argument, None)


def test_strips_quotes_from_import_path_arguments():
    assert _get_path_command_argument('/import "path/to/session.jsonl"', "/import") == "path/to/session.jsonl"
    assert (
        _get_path_command_argument('/import "path with spaces/session.jsonl"', "/import")
        == "path with spaces/session.jsonl"
    )


def test_preserves_apostrophes_in_unquoted_import_path_arguments():
    assert _get_path_command_argument("/import john's/session.jsonl", "/import") == "john's/session.jsonl"


def test_enforces_command_token_boundaries():
    assert _get_path_command_argument("/important /tmp/session.jsonl", "/import") is None
    assert _get_path_command_argument("/exporter out.html", "/export") is None
    assert _get_path_command_argument("/import /tmp/session.jsonl", "/import") == "/tmp/session.jsonl"


def _create_import_context(import_from_jsonl):
    context = SimpleNamespace(
        clear_status_calls=[],
        runtime_host=SimpleNamespace(import_from_jsonl=import_from_jsonl),
        show_error_calls=[],
        show_status_calls=[],
        confirm_calls=[],
    )
    context._clear_status_indicator = lambda kind=None: context.clear_status_calls.append(kind)
    context.show_error = context.show_error_calls.append
    context.show_status = context.show_status_calls.append

    async def show_extension_confirm(title, message, opts=None):
        context.confirm_calls.append((title, message))
        return True

    context._show_extension_confirm = show_extension_confirm

    async def handle_fatal_runtime_error(prefix, error):
        raise AssertionError("unexpected fatal error")

    context._handle_fatal_runtime_error = handle_fatal_runtime_error

    async def prompt_for_missing_session_cwd(error):
        return None

    context._prompt_for_missing_session_cwd = prompt_for_missing_session_cwd
    context._get_path_command_argument = partial(InteractiveMode._get_path_command_argument, context)
    return context


@pytest.mark.tonio
async def test_passes_unquoted_path_to_runtime_host_import_from_jsonl():
    import_calls: list = []

    async def import_from_jsonl(input_path, cwd_override=None):
        import_calls.append((input_path, cwd_override))
        return {"cancelled": False}

    context = _create_import_context(import_from_jsonl)

    await InteractiveMode.handle_import_command(context, '/import "path/to/session.jsonl"')

    assert context.confirm_calls == [("Import session", "Replace current session with path/to/session.jsonl?")]
    assert import_calls == [("path/to/session.jsonl", None)]
    assert context.show_error_calls == []
    assert context.show_status_calls == ["Session imported from: path/to/session.jsonl"]


@pytest.mark.tonio
async def test_passes_unquoted_apostrophe_path_to_runtime_host_import_from_jsonl_unchanged():
    import_calls: list = []

    async def import_from_jsonl(input_path, cwd_override=None):
        import_calls.append((input_path, cwd_override))
        return {"cancelled": False}

    context = _create_import_context(import_from_jsonl)

    await InteractiveMode.handle_import_command(context, "/import john's/session.jsonl")

    assert import_calls == [("john's/session.jsonl", None)]
    assert context.show_error_calls == []
    assert context.show_status_calls == ["Session imported from: john's/session.jsonl"]


@pytest.mark.tonio
async def test_shows_a_non_fatal_error_when_import_path_does_not_exist():
    async def import_from_jsonl(input_path, cwd_override=None):
        raise SessionImportFileNotFoundError("/tmp/missing-session.jsonl")

    context = _create_import_context(import_from_jsonl)

    await InteractiveMode.handle_import_command(context, "/import /tmp/missing-session.jsonl")

    assert context.show_error_calls == ["Failed to import session: File not found: /tmp/missing-session.jsonl"]
    assert context.show_status_calls == []
