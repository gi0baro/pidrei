"""Mirror of pi's extensions-runner.test.ts.

Two translations run through the file. Extensions are `.py` modules with an
`extension(pi)` factory (see `loader.py` for why). And where pi spies on
`console.warn` to observe shortcut conflicts, this asserts on
`runner.get_shortcut_diagnostics()` — the same messages, from the list the
runner keeps for the UI, rather than through a stderr capture that would
fight the tonio runtime.

pi's `getCommandDiagnostics()` is not mirrored: it is reset by
`getRegisteredCommands()` and never written to, so pi's assertion that it
equals `[]` holds vacuously and pidrei has no such accessor.
"""

import os
import re
import shutil
import tempfile
from types import SimpleNamespace

import pytest

from pidrei.core.auth_storage import AuthStorage, FileAuthStorageBackend
from pidrei.core.event_bus import create_event_bus
from pidrei.core.extensions.loader import (
    create_extension_runtime,
    discover_and_load_extensions,
    load_extension_from_factory,
    load_extensions,
)
from pidrei.core.extensions.runner import ExtensionRunner, emit_project_trust_event
from pidrei.core.extensions.types import BoundaryContextPreview
from pidrei.core.keybindings import KeybindingsManager
from pidrei.core.session_manager import ProjectedSessionEntry, SessionManager
from pidrei.core.system_prompt import BuildSystemPromptOptions, build_system_prompt
from pidrei_ai.types import ModelCostTier

from .model_runtime_helpers import create_in_memory_model_registry


DEFAULT_KEYBINDINGS = KeybindingsManager().get_effective_config()

PROVIDER_MODEL_CONFIG = {
    "baseUrl": "https://provider.test/v1",
    "apiKey": "provider-test-key",
    "api": "openai-completions",
    "models": [
        {
            "id": "instant-model",
            "name": "Instant Model",
            "reasoning": False,
            "input": ["text"],
            "cost": {
                "input": 1,
                "output": 2,
                "cacheRead": 0.1,
                "cacheWrite": 1.25,
                "tiers": [
                    {
                        "inputTokensAbove": 272000,
                        "input": 2,
                        "output": 3,
                        "cacheRead": 0.2,
                        "cacheWrite": 2.5,
                    }
                ],
            },
            "contextWindow": 128000,
            "maxTokens": 4096,
        }
    ],
}


def extension_actions() -> dict:
    return {
        "send_message": lambda *args: None,
        "send_user_message": lambda *args: None,
        "append_entry": lambda *args: None,
        "set_session_name": lambda *args: None,
        "get_session_name": lambda: None,
        "set_label": lambda *args: None,
        "get_active_tools": list,
        "get_all_tools": list,
        "set_active_tools": lambda *args: None,
        "refresh_tools": lambda: None,
        "get_commands": list,
        "set_model": _false,
        "get_thinking_level": lambda: "off",
        "set_thinking_level": lambda *args: None,
    }


async def _false(*_args) -> bool:
    return False


def extension_context_actions() -> dict:
    return {
        "get_model": lambda: None,
        "get_scoped_models": list,
        "is_idle": lambda: True,
        "is_project_trusted": lambda: True,
        "get_signal": lambda: None,
        "abort": lambda: None,
        "has_pending_messages": lambda: False,
        "shutdown": lambda: None,
        "get_context_usage": lambda: None,
        "compact": lambda *args: None,
        "get_system_prompt": lambda: "",
    }


class _Fixture:
    def __init__(self) -> None:
        self.root = tempfile.mkdtemp(prefix="pidrei-runner-test-")
        self.extensions = os.path.join(self.root, "extensions")
        os.makedirs(self.extensions)
        self.session_manager = SessionManager.in_memory()
        # Constructor: cannot await. The temp root is fresh, so there is no
        # auth.json to load and the loaded state would be empty regardless.
        self.auth_storage = AuthStorage.from_storage(FileAuthStorageBackend(os.path.join(self.root, "auth.json")))
        self.model_registry = None

    def write(self, name: str, content: str) -> str:
        path = os.path.join(self.extensions, name)
        with open(path, "w") as handle:
            handle.write(content)
        return path

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


@pytest.fixture
def fx(request):
    holder = _Fixture()
    request.addfinalizer(holder.cleanup)
    return holder


async def make_runner(fx, extensions=None, runtime=None) -> ExtensionRunner:
    if fx.model_registry is None:
        fx.model_registry = await create_in_memory_model_registry(fx.auth_storage)
    if extensions is None:
        result = await discover_and_load_extensions([], fx.root, fx.root)
        extensions, runtime = result.extensions, result.runtime
    return ExtensionRunner(extensions, runtime, fx.root, fx.session_manager, fx.model_registry)


def shortcut_extension(shortcut: str, description: str) -> str:
    return f"""
def extension(pi):
    pi.register_shortcut({shortcut!r}, description={description!r}, handler=lambda ctx: None)
"""


def tool_extension(name: str, description: str = "Test tool") -> str:
    return f"""
from pidrei.core.extensions import ToolDefinition


async def run(*args):
    return {{"content": [{{"type": "text", "text": "ok"}}], "details": {{}}}}


def extension(pi):
    pi.register_tool(
        ToolDefinition(
            name={name!r},
            label={name!r},
            description={description!r},
            parameters={{"type": "object", "properties": {{}}}},
            execute=run,
        )
    )
"""


def command_extension(name: str, description: str) -> str:
    return f"""
async def run(args, ctx):
    return None


def extension(pi):
    pi.register_command({name!r}, description={description!r}, handler=run)
"""


def diagnostic_messages(runner: ExtensionRunner) -> list[str]:
    return [diagnostic.message for diagnostic in runner.get_shortcut_diagnostics()]


# -- project_trust ---------------------------------------------------------------


@pytest.mark.tonio
async def test_continues_past_undecided_handlers_and_returns_the_first_decision(fx):
    undecided = fx.write(
        "undecided.py",
        "\nasync def handler(event, ctx):\n"
        '    return {"trusted": "undecided", "remember": True}\n'
        "\n\ndef extension(pi):\n"
        '    pi.on("project_trust", handler)\n',
    )
    decided = fx.write(
        "decided.py",
        "\nasync def handler(event, ctx):\n"
        '    return {"trusted": "no", "remember": True}\n'
        "\n\ndef extension(pi):\n"
        '    pi.on("project_trust", handler)\n',
    )

    extensions_result = await load_extensions([undecided, decided], fx.root)
    result, errors = await emit_project_trust_event(
        extensions_result,
        {"type": "project_trust", "cwd": fx.root},
        {"cwd": fx.root, "mode": "tui", "has_ui": False},
    )

    assert result == {"trusted": "no", "remember": True}
    assert errors == []


# -- shortcut conflicts ----------------------------------------------------------


@pytest.mark.tonio
async def test_warns_when_extension_shortcut_conflicts_with_builtin(fx):
    fx.write("conflict.py", shortcut_extension("ctrl+c", "Conflicts with built-in"))

    runner = await make_runner(fx)
    shortcuts = runner.get_shortcuts(DEFAULT_KEYBINDINGS)

    assert any("conflicts with built-in" in message for message in diagnostic_messages(runner))
    assert "ctrl+c" not in shortcuts


@pytest.mark.tonio
async def test_allows_a_shortcut_when_the_reserved_set_no_longer_contains_the_default_key(fx):
    fx.write("rebinding.py", shortcut_extension("ctrl+p", "Uses freed default"))

    runner = await make_runner(fx)
    keybindings = {**DEFAULT_KEYBINDINGS, "app.model.cycleForward": "ctrl+n"}
    shortcuts = runner.get_shortcuts(keybindings)

    assert "ctrl+p" in shortcuts
    assert not any("conflicts with built-in" in message for message in diagnostic_messages(runner))


@pytest.mark.tonio
async def test_warns_but_allows_when_extension_uses_non_reserved_builtin_shortcut(fx):
    paste_image_key = DEFAULT_KEYBINDINGS["app.clipboard.pasteImage"]
    if isinstance(paste_image_key, list):
        paste_image_key = paste_image_key[0]
    fx.write("non-reserved.py", shortcut_extension(paste_image_key, "Overrides non-reserved"))

    runner = await make_runner(fx)
    shortcuts = runner.get_shortcuts(DEFAULT_KEYBINDINGS)

    assert any("built-in shortcut for app.clipboard.pasteImage" in message for message in diagnostic_messages(runner))
    assert paste_image_key in shortcuts


@pytest.mark.tonio
async def test_blocks_shortcuts_for_reserved_actions_even_when_rebound(fx):
    fx.write("rebound-reserved.py", shortcut_extension("ctrl+x", "Conflicts with rebound reserved"))

    runner = await make_runner(fx)
    shortcuts = runner.get_shortcuts({**DEFAULT_KEYBINDINGS, "app.interrupt": "ctrl+x"})

    assert any("conflicts with built-in" in message for message in diagnostic_messages(runner))
    assert "ctrl+x" not in shortcuts


@pytest.mark.tonio
async def test_blocks_shortcuts_when_reserved_key_is_also_bound_to_non_reserved_actions(fx):
    fx.write("shared-reserved.py", shortcut_extension("ctrl+p", "Conflicts with shared reserved default"))

    runner = await make_runner(fx)
    shortcuts = runner.get_shortcuts(DEFAULT_KEYBINDINGS)

    assert any("conflicts with built-in" in message for message in diagnostic_messages(runner))
    assert "ctrl+p" not in shortcuts


@pytest.mark.tonio
async def test_blocks_shortcuts_when_reserved_action_has_multiple_keys(fx):
    fx.write("multi-reserved.py", shortcut_extension("ctrl+y", "Conflicts with multi-key reserved"))

    runner = await make_runner(fx)
    shortcuts = runner.get_shortcuts({**DEFAULT_KEYBINDINGS, "app.clear": ["ctrl+x", "ctrl+y"]})

    assert any("conflicts with built-in" in message for message in diagnostic_messages(runner))
    assert "ctrl+y" not in shortcuts


@pytest.mark.tonio
async def test_warns_but_allows_when_non_reserved_action_has_multiple_keys(fx):
    fx.write("multi-non-reserved.py", shortcut_extension("ctrl+y", "Overrides multi-key non-reserved"))

    runner = await make_runner(fx)
    shortcuts = runner.get_shortcuts({**DEFAULT_KEYBINDINGS, "app.clipboard.pasteImage": ["ctrl+x", "ctrl+y"]})

    assert any("built-in shortcut for app.clipboard.pasteImage" in message for message in diagnostic_messages(runner))
    assert "ctrl+y" in shortcuts


@pytest.mark.tonio
async def test_warns_when_two_extensions_register_the_same_shortcut(fx):
    fx.write("ext1.py", shortcut_extension("ctrl+shift+x", "First extension"))
    fx.write("ext2.py", shortcut_extension("ctrl+shift+x", "Second extension"))

    runner = await make_runner(fx)
    shortcuts = runner.get_shortcuts(DEFAULT_KEYBINDINGS)

    assert any("shortcut conflict" in message for message in diagnostic_messages(runner))
    # Last one wins.
    assert "ctrl+shift+x" in shortcuts
    assert shortcuts["ctrl+shift+x"].description == "Second extension"


@pytest.mark.tonio
async def test_matches_reserved_keys_case_insensitively(fx):
    """Not in pi: every default keybinding is already lowercase there, so the
    normalisation both sides do is never exercised. A user's keybindings.json
    is hand-written and can say "Ctrl+C"."""
    # ctrl+shift+q is bound by nothing else, so the reserved binding under test
    # is the only thing that can block it.
    fx.write("cased.py", shortcut_extension("Ctrl+Shift+Q", "Cased conflict"))

    runner = await make_runner(fx)
    shortcuts = runner.get_shortcuts({**DEFAULT_KEYBINDINGS, "app.clear": "CTRL+SHIFT+Q"})

    assert any("conflicts with built-in" in message for message in diagnostic_messages(runner))
    assert shortcuts == {}


# -- tool collection -------------------------------------------------------------


@pytest.mark.tonio
async def test_collects_tools_from_multiple_extensions(fx):
    fx.write("tool-a.py", tool_extension("tool_a"))
    fx.write("tool-b.py", tool_extension("tool_b"))

    runner = await make_runner(fx)
    tools = runner.get_all_registered_tools()

    assert len(tools) == 2
    assert sorted(tool.definition.name for tool in tools) == ["tool_a", "tool_b"]


# Regression test for #9300. pi omits `parameters`; ToolDefinition requires the
# field, so the Python shape of the malformed tool is an explicit None.
@pytest.mark.tonio
async def test_rejects_extension_tools_without_a_parameter_schema(fx):
    extension_path = fx.write(
        "missing-parameters.py",
        """
from pidrei.core.extensions import ToolDefinition


async def run(*args):
    return {"content": [{"type": "text", "text": "ok"}]}


def extension(pi):
    pi.register_tool(ToolDefinition(name="noop", label="No-op", description="Do nothing", parameters=None, execute=run))
""",
    )

    result = await load_extensions([extension_path], fx.root)

    assert len(result.extensions) == 0
    assert [(error.path, error.error) for error in result.errors] == [
        (
            extension_path,
            (
                f'Failed to load extension: Tool "noop" registered by extension "{extension_path}" '
                "must define an object parameter schema."
            ),
        )
    ]


@pytest.mark.tonio
async def test_keeps_first_tool_when_two_extensions_register_the_same_name(fx):
    fx.write("a-first.py", tool_extension("shared", "first"))
    fx.write("b-second.py", tool_extension("shared", "second"))

    runner = await make_runner(fx)
    tools = runner.get_all_registered_tools()

    assert len(tools) == 1
    assert tools[0].definition.description == "first"


# -- command collection ----------------------------------------------------------


@pytest.mark.tonio
async def test_collects_commands_from_multiple_extensions(fx):
    fx.write("cmd-a.py", command_extension("cmd-a", "Test command"))
    fx.write("cmd-b.py", command_extension("cmd-b", "Test command"))

    runner = await make_runner(fx)
    commands = runner.get_registered_commands()

    assert len(commands) == 2
    assert sorted(command.name for command in commands) == ["cmd-a", "cmd-b"]
    assert sorted(command.invocation_name for command in commands) == ["cmd-a", "cmd-b"]


@pytest.mark.tonio
async def test_gets_command_by_invocation_name(fx):
    fx.write("cmd.py", command_extension("my-cmd", "My command"))

    runner = await make_runner(fx)

    command = runner.get_command("my-cmd")
    assert command is not None
    assert command.name == "my-cmd"
    assert command.invocation_name == "my-cmd"
    assert command.description == "My command"

    assert runner.get_command("not-exists") is None


@pytest.mark.tonio
async def test_suffixes_duplicate_extension_commands_in_insertion_order(fx):
    fx.write("cmd-a.py", command_extension("shared-cmd", "First command"))
    fx.write("cmd-b.py", command_extension("shared-cmd", "Second command"))

    runner = await make_runner(fx)
    commands = runner.get_registered_commands()

    assert len(commands) == 2
    assert [command.name for command in commands] == ["shared-cmd", "shared-cmd"]
    assert [command.invocation_name for command in commands] == ["shared-cmd:1", "shared-cmd:2"]
    assert [command.description for command in commands] == ["First command", "Second command"]
    assert runner.get_command("shared-cmd:1").description == "First command"
    assert runner.get_command("shared-cmd:2").description == "Second command"


# -- context creation ------------------------------------------------------------


@pytest.mark.tonio
async def test_reflects_the_get_scoped_models_context_action_on_ctx_scoped_models(fx):
    runner = await make_runner(fx)

    # Before bind_core the default is an empty list (never None).
    assert runner.create_context().scoped_models == []

    # After bind_core wires a get_scoped_models action, ctx.scoped_models
    # returns it live (same reference, lazy getter).
    scoped = [SimpleNamespace(model=SimpleNamespace(id="scoped-test"), thinking_level="high")]
    runner.bind_core(extension_actions(), {**extension_context_actions(), "get_scoped_models": lambda: scoped})
    assert runner.create_context().scoped_models is scoped


@pytest.mark.tonio
async def test_exposes_the_current_cancel_token_on_extension_context(fx):
    from pidrei_ai.utils.cancel import CancelToken

    runner = await make_runner(fx)
    cancel = CancelToken()

    runner.bind_core(extension_actions(), {**extension_context_actions(), "get_signal": lambda: cancel})

    ctx = runner.create_context()
    assert ctx.signal is cancel
    assert ctx.signal.cancelled is False

    cancel.cancel()
    assert ctx.signal.cancelled is True


@pytest.mark.tonio
async def test_exposes_print_mode_and_has_ui_false_by_default(fx):
    runner = await make_runner(fx)
    runner.bind_core(extension_actions(), extension_context_actions())

    ctx = runner.create_context()
    assert ctx.mode == "print"
    assert ctx.has_ui is False


@pytest.mark.tonio
async def test_exposes_project_trust_state_on_extension_context(fx):
    runner = await make_runner(fx)
    runner.bind_core(extension_actions(), {**extension_context_actions(), "is_project_trusted": lambda: False})

    assert runner.create_context().is_project_trusted() is False


@pytest.mark.tonio
async def test_exposes_rpc_mode_with_has_ui_true_when_an_rpc_ui_context_is_provided(fx):
    runner = await make_runner(fx)
    runner.bind_core(extension_actions(), extension_context_actions())
    runner.set_ui_context(object(), "rpc")

    ctx = runner.create_context()
    assert ctx.mode == "rpc"
    assert ctx.has_ui is True


@pytest.mark.tonio
async def test_exposes_tui_mode_with_has_ui_true_when_a_tui_ui_context_is_provided(fx):
    runner = await make_runner(fx)
    runner.bind_core(extension_actions(), extension_context_actions())
    runner.set_ui_context(object(), "tui")

    ctx = runner.create_context()
    assert ctx.mode == "tui"
    assert ctx.has_ui is True


# -- error handling --------------------------------------------------------------


@pytest.mark.tonio
async def test_calls_error_listeners_when_handler_raises(fx):
    fx.write(
        "throws.py",
        """
async def boom(event, ctx):
    raise RuntimeError("Handler error!")


def extension(pi):
    pi.on("context", boom)
""",
    )

    runner = await make_runner(fx)
    errors = []
    runner.on_error(errors.append)

    await runner.emit_context([])

    assert len(errors) == 1
    assert "Handler error!" in errors[0].error
    assert errors[0].event == "context"


def _user_bash_event(fx, command: str = "pwd") -> dict:
    return {"type": "user_bash", "command": command, "excludeFromContext": False, "cwd": fx.root}


# Regression test for #9068.
@pytest.mark.tonio
async def test_fails_closed_when_a_user_bash_handler_throws(fx):
    fx.write(
        "throws.py",
        """
async def handler(event, ctx):
    raise RuntimeError("Routing failed")


def extension(pi):
    pi.on("user_bash", handler)
""",
    )

    runner = await make_runner(fx)
    errors = []
    runner.on_error(errors.append)

    with pytest.raises(RuntimeError, match="Routing failed"):
        await runner.emit_user_bash(_user_bash_event(fx))
    assert [(error.event, error.error) for error in errors] == [("user_bash", "Routing failed")]


# Regression test for #9068.
@pytest.mark.tonio
@pytest.mark.parametrize(
    "handler_result",
    [
        "{}",
        '{"operations": None}',
        '{"operations": SimpleNamespace()}',
        '{"result": None}',
        '{"result": {"output": "handled"}}',
        (
            '{"operations": SimpleNamespace(exec=run), '
            '"result": {"output": "handled", "exitCode": 0, "cancelled": False, "truncated": False}}'
        ),
    ],
    ids=[
        "an empty dict",
        "None operations",
        "operations without exec",
        "a None result",
        "an incomplete result",
        "operations and a result",
    ],
)
async def test_fails_closed_when_a_user_bash_handler_returns_an_invalid_result(fx, handler_result):
    fx.write(
        "invalid_result.py",
        f"""
from types import SimpleNamespace


async def run(*args, **kwargs):
    return {{"exitCode": 0}}


async def handler(event, ctx):
    return {handler_result}


def extension(pi):
    pi.on("user_bash", handler)
""",
    )

    runner = await make_runner(fx)
    errors = []
    runner.on_error(errors.append)

    with pytest.raises(Exception, match="Invalid user_bash handler result"):
        await runner.emit_user_bash(_user_bash_event(fx))
    assert len(errors) == 1
    assert errors[0].event == "user_bash"
    assert "Invalid user_bash handler result" in errors[0].error


@pytest.mark.tonio
async def test_accepts_valid_user_bash_operations_and_result_overrides(fx):
    fx.write(
        "valid_results.py",
        """
from types import SimpleNamespace


async def run(*args, **kwargs):
    return {"exitCode": 0}


async def handler(event, ctx):
    if event["command"] == "operations":
        return {"operations": SimpleNamespace(exec=run)}
    return {"result": {"output": "handled", "exitCode": 0, "cancelled": False, "truncated": False}}


def extension(pi):
    pi.on("user_bash", handler)
""",
    )

    runner = await make_runner(fx)

    operations = await runner.emit_user_bash(_user_bash_event(fx, "operations"))
    assert list(operations) == ["operations"]
    assert callable(operations["operations"].exec)
    assert await runner.emit_user_bash(_user_bash_event(fx, "result")) == {
        "result": {"output": "handled", "exitCode": 0, "cancelled": False, "truncated": False}
    }


# -- renderers -------------------------------------------------------------------


@pytest.mark.tonio
async def test_gets_markdown_transformers_in_extension_load_order(fx):
    source = "\ndef extension(pi):\n    pi.register_markdown_transformer(lambda markdown, context: markdown)\n"
    fx.write("markdown-renderer-a.py", source)
    fx.write("markdown-renderer-b.py", source)

    runner = await make_runner(fx)
    assert len(runner.get_markdown_transformers()) == 2


@pytest.mark.tonio
async def test_gets_message_renderer_by_type(fx):
    fx.write(
        "renderer.py",
        '\ndef extension(pi):\n    pi.register_message_renderer("my-type", lambda *args: None)\n',
    )

    runner = await make_runner(fx)
    assert runner.get_message_renderer("my-type") is not None
    assert runner.get_message_renderer("not-exists") is None


@pytest.mark.tonio
async def test_gets_entry_renderer_by_type(fx):
    fx.write(
        "entry-renderer.py",
        '\ndef extension(pi):\n    pi.register_entry_renderer("my-entry", lambda *args: None)\n',
    )

    runner = await make_runner(fx)
    assert runner.get_entry_renderer("my-entry") is not None
    assert runner.get_entry_renderer("not-exists") is None


# -- flags -----------------------------------------------------------------------


@pytest.mark.tonio
async def test_collects_flags_from_extensions(fx):
    fx.write(
        "with-flag.py",
        '\ndef extension(pi):\n    pi.register_flag("my-flag", type="boolean", description="My flag")\n',
    )

    runner = await make_runner(fx)
    assert "my-flag" in runner.get_flags()


@pytest.mark.tonio
async def test_keeps_first_flag_when_two_extensions_register_the_same_name(fx):
    fx.write(
        "a-first.py",
        '\ndef extension(pi):\n    pi.register_flag("shared-flag", type="boolean", description="first", default=True)\n',
    )
    fx.write(
        "b-second.py",
        '\ndef extension(pi):\n    pi.register_flag("shared-flag", type="boolean", description="second", default=False)\n',
    )

    result = await discover_and_load_extensions([], fx.root, fx.root)
    runner = await make_runner(fx, result.extensions, result.runtime)
    flags = runner.get_flags()

    assert flags["shared-flag"].description == "first"
    assert result.runtime.flag_values["shared-flag"] is True


@pytest.mark.tonio
async def test_rejects_default_values_that_do_not_match_the_flag_type(fx):
    fx.write(
        "bad-flag-default.py",
        '\ndef extension(pi):\n    pi.register_flag("safe-mode", type="boolean", default="false")\n',
    )

    result = await discover_and_load_extensions([], fx.root, fx.root)

    assert len(result.extensions) == 0
    assert 'Invalid default for flag "safe-mode": expected boolean, got string' in result.errors[0].error
    assert "safe-mode" not in result.runtime.flag_values


@pytest.mark.tonio
async def test_can_set_flag_values(fx):
    fx.write(
        "flag.py",
        '\ndef extension(pi):\n    pi.register_flag("test-flag", type="boolean", description="Test flag")\n',
    )

    result = await discover_and_load_extensions([], fx.root, fx.root)
    runner = await make_runner(fx, result.extensions, result.runtime)

    runner.set_flag_value("--test-flag", True)

    assert result.runtime.flag_values["--test-flag"] is True


# -- before_agent_start ----------------------------------------------------------


@pytest.mark.tonio
async def test_keeps_ctx_get_system_prompt_in_sync_with_chained_updates(fx):
    fx.write(
        "before-agent-start-1.py",
        """
async def handler(event, ctx):
    return {"systemPrompt": ctx.get_system_prompt() + "\\nfirst"}


def extension(pi):
    pi.on("before_agent_start", handler)
""",
    )
    fx.write(
        "before-agent-start-2.py",
        """
async def handler(event, ctx):
    return {"systemPrompt": ctx.get_system_prompt() + "\\nsecond"}


def extension(pi):
    pi.on("before_agent_start", handler)
""",
    )

    result = await discover_and_load_extensions([], fx.root, fx.root)
    assert result.errors == []
    assert len(result.extensions) == 2

    runner = await make_runner(fx, result.extensions, result.runtime)
    errors = []
    runner.on_error(lambda error: errors.append(error.error))
    runner.bind_core(extension_actions(), extension_context_actions())

    chained = await runner.emit_before_agent_start(
        "hello", None, BuildSystemPromptOptions(custom_prompt="base", cwd=fx.root)
    )

    assert errors == []
    assert chained["messages"] == []
    assert re.search(r"base[\s\S]*\nfirst\nsecond$", build_system_prompt(chained["systemPromptOptions"]))


# -- boundary chaining -------------------------------------------------------------

BEFORE_SETTLE = {"type": "agent_before_settle", "outcome": "completed"}


def _empty_preview(entries=()) -> BoundaryContextPreview:
    return BoundaryContextPreview(
        context_entries=list(entries), context_messages=[], llm_messages=[], pending_messages=[], can_continue=False
    )


async def _load_factories(fx, *factories) -> ExtensionRunner:
    runtime = create_extension_runtime()
    extensions = [
        await load_extension_from_factory(factory, fx.root, create_event_bus(), runtime, f"<inline:{index}>")
        for index, factory in enumerate(factories)
    ]
    return await make_runner(fx, extensions, runtime)


@pytest.mark.tonio
async def test_boundary_chains_shared_draft_proposals_and_preserves_omitted_result_fields(fx):
    observations: list[dict] = []

    def first(pi) -> None:
        async def handler(event, _ctx):
            observations.append(
                {
                    "entries": len(event["entries"]),
                    "continuation": event["continue"],
                    "preview": len(event["context"].context_entries),
                }
            )
            event["entries"].append({"type": "custom", "customType": "first", "data": 1})
            return {"continue": True}

        pi.on("agent_before_settle", handler)

    def second(pi) -> None:
        async def handler(event, _ctx):
            observations.append(
                {
                    "entries": len(event["entries"]),
                    "continuation": event["continue"],
                    "preview": len(event["context"].context_entries),
                }
            )
            return {"entries": []}

        pi.on("agent_before_settle", handler)

    runner = await _load_factories(fx, first, second)

    async def build_context(entries):
        return _empty_preview(
            ProjectedSessionEntry(
                source_entry={
                    "type": "custom",
                    "id": f"draft-{index}",
                    "parentId": None,
                    "timestamp": "",
                    "customType": entry["type"],
                },
                messages=[],
            )
            for index, entry in enumerate(entries)
        )

    result = await runner.emit_boundary(BEFORE_SETTLE, build_context)

    assert observations == [
        {"entries": 0, "continuation": False, "preview": 0},
        {"entries": 1, "continuation": True, "preview": 1},
    ]
    assert result.entries == []
    assert result.continue_ is True


@pytest.mark.tonio
async def test_boundary_reports_invalid_previews_and_lets_later_handlers_repair_the_proposal(fx):
    second_ran = False

    def invalid(pi) -> None:
        async def handler(_event, _ctx):
            return {"entries": [{"type": "context_edit", "targetId": "missing", "replacement": None}]}

        pi.on("agent_before_settle", handler)

    def repair(pi) -> None:
        async def handler(event, _ctx):
            nonlocal second_ran
            second_ran = True
            assert len(event["entries"]) == 1
            return {"entries": []}

        pi.on("agent_before_settle", handler)

    runner = await _load_factories(fx, invalid, repair)
    errors: list[str] = []
    runner.on_error(lambda error: errors.append(error.error))

    async def build_context(entries):
        if any(entry["type"] == "context_edit" for entry in entries):
            raise Exception("Entry missing not found")
        return _empty_preview()

    result = await runner.emit_boundary(BEFORE_SETTLE, build_context)

    assert second_ran is True
    assert "Invalid boundary entries: Entry missing not found" in errors
    assert result.entries == []
    assert result.valid is True


@pytest.mark.tonio
async def test_boundary_keeps_shared_mutations_made_before_a_handler_throws(fx):
    def failing(pi) -> None:
        async def handler(event, _ctx):
            event["entries"].append({"type": "custom", "customType": "kept"})
            raise Exception("boundary failed")

        pi.on("agent_before_settle", handler)

    runner = await _load_factories(fx, failing)
    errors: list[str] = []
    runner.on_error(lambda error: errors.append(error.error))

    async def build_context(_entries):
        return _empty_preview()

    result = await runner.emit_boundary(BEFORE_SETTLE, build_context)

    assert result.entries == [{"type": "custom", "customType": "kept"}]
    assert errors == ["boundary failed"]


# -- tool_result chaining --------------------------------------------------------


@pytest.mark.tonio
async def test_chains_content_modifications_across_handlers(fx):
    fx.write(
        "tool-result-1.py",
        """
async def handler(event, ctx):
    return {"content": [*event["content"], {"type": "text", "text": "ext1"}]}


def extension(pi):
    pi.on("tool_result", handler)
""",
    )
    fx.write(
        "tool-result-2.py",
        """
async def handler(event, ctx):
    return {"content": [*event["content"], {"type": "text", "text": "ext2"}]}


def extension(pi):
    pi.on("tool_result", handler)
""",
    )

    runner = await make_runner(fx)
    chained = await runner.emit_tool_result(
        {
            "type": "tool_result",
            "toolName": "my_tool",
            "toolCallId": "call-1",
            "input": {},
            "content": [{"type": "text", "text": "base"}],
            "details": {"initial": True},
            "isError": False,
        }
    )

    assert chained is not None
    content = chained["content"]
    assert content[0] == {"type": "text", "text": "base"}
    assert len(content) == 3
    assert sorted(item["text"] for item in content[1:]) == ["ext1", "ext2"]


@pytest.mark.tonio
async def test_preserves_previous_modifications_when_later_handlers_return_partial_patches(fx):
    fx.write(
        "tool-result-partial-1.py",
        """
async def handler(event, ctx):
    return {"content": [{"type": "text", "text": "first"}], "details": {"source": "ext1"}}


def extension(pi):
    pi.on("tool_result", handler)
""",
    )
    fx.write(
        "tool-result-partial-2.py",
        """
async def handler(event, ctx):
    return {"isError": True}


def extension(pi):
    pi.on("tool_result", handler)
""",
    )

    runner = await make_runner(fx)
    chained = await runner.emit_tool_result(
        {
            "type": "tool_result",
            "toolName": "my_tool",
            "toolCallId": "call-2",
            "input": {},
            "content": [{"type": "text", "text": "base"}],
            "details": {"initial": True},
            "isError": False,
        }
    )

    # pi asserts with toEqual, which drops the `usage: undefined` pi also
    # returns; None is a value in Python, so it is spelled out here.
    assert chained == {
        "content": [{"type": "text", "text": "first"}],
        "details": {"source": "ext1"},
        "isError": True,
        "usage": None,
    }


# -- provider registration -------------------------------------------------------


@pytest.mark.tonio
async def test_bind_core_ignores_invalid_queued_registrations_and_reports_extension_error(fx):
    runtime = create_extension_runtime()
    runtime.register_provider(
        "broken-provider",
        {"streamSimple": lambda *args: None},
        "/tmp/broken-extension.py",
    )

    runner = await make_runner(fx, [], runtime)
    errors = []
    runner.on_error(lambda error: errors.append(f"{error.extension_path}: {error.error}"))

    runner.bind_core(extension_actions(), extension_context_actions())

    assert errors == [
        '/tmp/broken-extension.py: Provider broken-provider: "api" is required when registering streamSimple.'
    ]
    assert (await fx.model_registry.refresh()).aborted is False


@pytest.mark.tonio
async def test_pre_bind_unregister_removes_all_queued_registrations_for_a_provider(fx):
    runtime = create_extension_runtime()

    runtime.register_provider("queued-provider", PROVIDER_MODEL_CONFIG)
    runtime.register_provider(
        "queued-provider",
        {
            **PROVIDER_MODEL_CONFIG,
            "models": [
                {
                    "id": "instant-model-2",
                    "name": "Instant Model 2",
                    "reasoning": False,
                    "input": ["text"],
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    "contextWindow": 128000,
                    "maxTokens": 4096,
                }
            ],
        },
    )
    assert len(runtime.pending_provider_registrations) == 2

    runtime.unregister_provider("queued-provider")
    assert runtime.pending_provider_registrations == []


@pytest.mark.tonio
async def test_post_bind_register_and_unregister_take_effect_immediately(fx):
    runtime = create_extension_runtime()
    runner = await make_runner(fx, [], runtime)

    runner.bind_core(extension_actions(), extension_context_actions())
    assert runtime.pending_provider_registrations == []

    runtime.register_provider("instant-provider", PROVIDER_MODEL_CONFIG)
    assert runtime.pending_provider_registrations == []
    model = fx.model_registry.find("instant-provider", "instant-model")
    # pi compares the raw config object; pidrei parses the wire tier into a
    # ModelCostTier on the way in, so the assertion is on the parsed shape.
    assert model.cost.tiers == [
        ModelCostTier(input=2, output=3, cache_read=0.2, cache_write=2.5, input_tokens_above=272000)
    ]

    # Deliberately asserted with the refresh that register_provider() detached
    # still in flight: on the free-threaded runtime its provider rebuild runs
    # on another thread, and before ModelRuntime._composition_guard it could
    # undo this unregister. See test_model_runtime_provider_composition_race.
    runtime.unregister_provider("instant-provider")
    assert fx.model_registry.find("instant-provider", "instant-model") is None


# -- command context -------------------------------------------------------------


@pytest.mark.tonio
async def test_passes_fork_options_through_to_the_bound_handler(fx):
    runtime = create_extension_runtime()
    runner = await make_runner(fx, [], runtime)
    calls = []

    async def fork(entry_id, options=None):
        calls.append((entry_id, options))
        return {"cancelled": False}

    async def cancelled(*_args, **_kwargs):
        return {"cancelled": False}

    async def noop(*_args, **_kwargs):
        return None

    runner.bind_command_context(
        {
            "wait_for_idle": noop,
            "new_session": cancelled,
            "fork": fork,
            "navigate_tree": cancelled,
            "switch_session": cancelled,
            "reload": noop,
        }
    )

    command_context = runner.create_command_context()
    await command_context.fork("entry-1")
    assert calls[0] == ("entry-1", None)

    await command_context.fork("entry-2", {"position": "at"})
    assert calls[1] == ("entry-2", {"position": "at"})


# -- event subscriptions ---------------------------------------------------------
# #8967: event handler unsubscription must not disturb other registrations.


async def load_subscription_extension(fx, factory):
    runtime = create_extension_runtime()
    extension = await load_extension_from_factory(factory, fx.root, create_event_bus(), runtime)
    runner = await make_runner(fx, [extension], runtime)
    return extension, runner


def recorder(calls: list[str], name: str):
    async def handler(_event, _ctx):
        calls.append(name)

    return handler


AGENT_END = {"type": "agent_end", "messages": []}


@pytest.mark.tonio
async def test_allows_self_removal_without_skipping_neighboring_handlers(fx):
    calls: list[str] = []

    def factory(pi) -> None:
        async def first(_event, _ctx):
            calls.append("A")
            unsubscribe()

        unsubscribe = pi.on("agent_end", first)
        pi.on("agent_end", recorder(calls, "B"))

    _extension, runner = await load_subscription_extension(fx, factory)

    await runner.emit(AGENT_END)
    assert calls == ["A", "B"]

    await runner.emit(AGENT_END)
    assert calls == ["A", "B", "B"]


@pytest.mark.tonio
async def test_removes_duplicate_registrations_independently_and_cleans_up_the_last_handler(fx):
    calls: list[str] = []
    unsubscribers = []

    def factory(pi) -> None:
        shared = recorder(calls, "shared")
        unsubscribers.append(pi.on("agent_end", shared))
        unsubscribers.append(pi.on("agent_end", recorder(calls, "B")))
        unsubscribers.append(pi.on("agent_end", shared))

    extension, runner = await load_subscription_extension(fx, factory)
    stop_first, stop_b, stop_second = unsubscribers

    stop_second()
    stop_second()
    await runner.emit(AGENT_END)
    assert calls == ["shared", "B"]

    stop_first()
    await runner.emit(AGENT_END)
    assert calls == ["shared", "B", "B"]

    stop_b()
    assert "agent_end" not in extension.handlers


@pytest.mark.tonio
async def test_keeps_removed_pending_handlers_in_the_current_dispatch(fx):
    calls: list[str] = []

    def factory(pi) -> None:
        async def first(_event, _ctx):
            calls.append("A")
            stop_b()

        pi.on("agent_end", first)
        stop_b = pi.on("agent_end", recorder(calls, "B"))
        pi.on("agent_end", recorder(calls, "C"))

    _extension, runner = await load_subscription_extension(fx, factory)

    await runner.emit(AGENT_END)
    assert calls == ["A", "B", "C"]
    await runner.emit(AGENT_END)
    assert calls == ["A", "B", "C", "A", "C"]


@pytest.mark.tonio
async def test_defers_registrations_made_during_dispatch_until_the_next_dispatch(fx):
    calls: list[str] = []

    def factory(pi) -> None:
        async def first(_event, _ctx):
            calls.append("A")
            pi.on("agent_end", recorder(calls, "C"))

        pi.on("agent_end", first)
        pi.on("agent_end", recorder(calls, "B"))

    _extension, runner = await load_subscription_extension(fx, factory)

    await runner.emit(AGENT_END)
    assert calls == ["A", "B"]
    await runner.emit(AGENT_END)
    assert calls == ["A", "B", "A", "B", "C"]


@pytest.mark.tonio
async def test_uses_a_fresh_handler_list_for_nested_dispatches(fx):
    calls: list[str] = []
    holder: dict = {}

    def factory(pi) -> None:
        async def first(_event, _ctx):
            calls.append("A")
            stop_a()
            stop_b()
            pi.on("agent_end", recorder(calls, "C"))
            await holder["runner"].emit(AGENT_END)

        stop_a = pi.on("agent_end", first)
        stop_b = pi.on("agent_end", recorder(calls, "B"))

    _extension, runner = await load_subscription_extension(fx, factory)
    holder["runner"] = runner

    await runner.emit(AGENT_END)
    assert calls == ["A", "C", "B"]


# -- has_handlers ----------------------------------------------------------------


@pytest.mark.tonio
async def test_returns_true_when_handlers_exist_for_event_type(fx):
    fx.write(
        "handler.py",
        '\nasync def handler(event, ctx):\n    return None\n\n\ndef extension(pi):\n    pi.on("tool_call", handler)\n',
    )

    runner = await make_runner(fx)

    assert runner.has_handlers("tool_call") is True
    assert runner.has_handlers("agent_end") is False


# -- before_provider_headers -----------------------------------------------------


@pytest.mark.tonio
async def test_lets_a_handler_mutate_headers_in_place_and_preserves_existing_headers(fx):
    fx.write(
        "headers.py",
        """
async def handler(event, ctx):
    event["headers"]["X-Turn-Index"] = "3"


def extension(pi):
    pi.on("before_provider_headers", handler)
""",
    )

    runner = await make_runner(fx)
    assert runner.has_handlers("before_provider_headers") is True

    headers = await runner.emit_before_provider_headers({"User-Agent": "kimchi/1.0"})
    assert headers["X-Turn-Index"] == "3"
    assert headers["User-Agent"] == "kimchi/1.0"


@pytest.mark.tonio
async def test_isolates_a_throwing_handler_and_still_applies_the_others(fx):
    fx.write(
        "a-throwing.py",
        """
async def handler(event, ctx):
    raise RuntimeError("header handler boom")


def extension(pi):
    pi.on("before_provider_headers", handler)
""",
    )
    fx.write(
        "b-good.py",
        """
async def handler(event, ctx):
    event["headers"]["X-Good"] = "yes"


def extension(pi):
    pi.on("before_provider_headers", handler)
""",
    )

    runner = await make_runner(fx)
    errors = []
    runner.on_error(errors.append)

    headers = await runner.emit_before_provider_headers({"User-Agent": "x"})

    assert headers["X-Good"] == "yes"
    assert headers["User-Agent"] == "x"
    assert len(errors) == 1
    assert errors[0].event == "before_provider_headers"
    assert "header handler boom" in errors[0].error
