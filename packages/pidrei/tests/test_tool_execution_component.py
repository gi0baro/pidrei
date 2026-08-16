"""Mirror of pi coding-agent test/tool-execution-component.test.ts."""

import os
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pidrei.config import get_readme_path
from pidrei.core.extensions.types import ToolDefinition
from pidrei.core.tools.bash import BashExecResult, create_bash_tool_definition
from pidrei.core.tools.read import create_read_tool, create_read_tool_definition
from pidrei.core.tools.write import create_write_tool_definition
from pidrei.modes.interactive.components import ToolExecutionComponent
from pidrei.modes.interactive.theme import init_theme_sync, theme
from pidrei.utils.ansi import strip_ansi
from pidrei_tui import Text


def create_base_tool_definition(name: str = "custom_tool") -> ToolDefinition:
    async def execute(_tool_call_id, _params, cancel=None, on_update=None, ctx=None):
        return SimpleNamespace(content=[{"type": "text", "text": "ok"}], details={})

    return ToolDefinition(
        name=name,
        label=name,
        description="custom tool",
        parameters={},
        execute=execute,
    )


def create_fake_tui():
    return SimpleNamespace(request_render=lambda: None)


@pytest.fixture(autouse=True)
def _theme():
    init_theme_sync("dark")


CWD = os.getcwd()


class TestToolExecutionComponentParity:
    def test_stacks_custom_call_and_result_renderers_like_the_old_implementation(self):
        tool_definition = replace(
            create_base_tool_definition(),
            render_call=lambda args, theme, context: Text("custom call", 0, 0),
            render_result=lambda result, options, theme, context: Text("custom result", 0, 0),
        )

        component = ToolExecutionComponent("custom_tool", "tool-1", {}, {}, tool_definition, create_fake_tui(), CWD)
        assert "custom call" in strip_ansi("\n".join(component.render(120)))

        component.update_result({"content": [{"type": "text", "text": "done"}], "details": {}, "isError": False}, False)

        rendered = strip_ansi("\n".join(component.render(120)))
        assert "custom call" in rendered
        assert "custom result" in rendered

    def test_self_rendered_empty_tool_rows_take_no_layout_space(self):
        tool_definition = replace(
            create_base_tool_definition(),
            render_shell="self",
            render_call=lambda args, theme, context: Text("", 0, 0),
            render_result=lambda result, options, theme, context: Text("", 0, 0),
        )

        component = ToolExecutionComponent(
            "custom_tool", "tool-empty-self-render", {}, {}, tool_definition, create_fake_tui(), CWD
        )
        assert component.render(120) == []

        component.update_result({"content": [], "details": {}, "isError": False}, False)

        assert component.render(120) == []

    def test_uses_built_in_rendering_for_built_in_overrides_without_custom_renderers(self):
        override_definition = create_base_tool_definition("edit")

        component = ToolExecutionComponent(
            "edit",
            "tool-2",
            {"path": "README.md", "oldText": "before", "newText": "after"},
            {},
            override_definition,
            create_fake_tui(),
            CWD,
        )
        component.update_result(
            {"content": [], "details": {"diff": "+1 after", "firstChangedLine": 1}, "isError": False}
        )
        rendered = strip_ansi("\n".join(component.render(120)))
        assert "edit" in rendered
        assert "README.md" in rendered
        assert ":1" not in rendered

    def test_preserves_legacy_file_path_rendering_compatibility_for_built_in_tools(self):
        component = ToolExecutionComponent(
            "read", "tool-3", {"file_path": "README.md"}, {}, None, create_fake_tui(), CWD
        )
        rendered = strip_ansi("\n".join(component.render(120)))
        assert "read" in rendered
        assert "README.md" in rendered

    @pytest.mark.tonio
    async def test_bash_execute_emits_an_initial_empty_partial_update_before_output_arrives(self):
        import tonio.colored as tonio

        updates = []

        class Operations:
            async def exec(self, _command, _cwd, *, on_data, cancel=None, timeout=None, env=None):
                await tonio.sleep(0.01)
                return BashExecResult(exit_code=0)

        tool = create_bash_tool_definition(CWD, operations=Operations(), expose_session_environment=False)
        coro = tool.execute("tool-bash-1", {"command": "sleep 10"}, None, lambda update: updates.append(update), None)
        # pi asserts the update is emitted before the exec settles
        task = tonio.spawn(coro)
        await tonio.time.sleep(0.001)
        assert len(updates) == 1
        first = updates[0]
        content = first["content"] if isinstance(first, dict) else first.content
        assert content == []
        await task

    @pytest.mark.tonio
    async def test_bash_renderer_does_not_duplicate_final_full_output_truncation_details(self):
        import re

        class Operations:
            async def exec(self, _command, _cwd, *, on_data, cancel=None, timeout=None, env=None):
                for i in range(1, 4001):
                    on_data(f"line-{i:04d}\n".encode())
                return BashExecResult(exit_code=0)

        tool = create_bash_tool_definition(CWD, operations=Operations(), expose_session_environment=False)
        result = await tool.execute("tool-bash-1b", {"command": "generate output"}, None, None, None)
        component = ToolExecutionComponent(
            "bash", "tool-bash-1b", {"command": "generate output"}, {}, tool, create_fake_tui(), CWD
        )
        component.set_expanded(True)
        component.update_result({"content": result.content, "details": result.details, "isError": False}, False)

        rendered = strip_ansi("\n".join(component.render(200)))
        assert len(re.findall(r"Full output:", rendered)) == 1
        assert re.search(r"line-4000[^\n]*\n[^\S\n]*\n \[Full output:", rendered)
        assert not re.search(r"line-4000[^\n]*\n[^\S\n]*\n[^\S\n]*\n \[Full output:", rendered)
        assert "Truncated: showing 2000 of 4000 lines" in rendered
        assert "[Showing lines 2001-4000 of 4000. Full output:" not in rendered

    def test_does_not_duplicate_built_in_headers_when_passed_the_active_built_in_definition(self):
        import re

        component = ToolExecutionComponent(
            "read", "tool-4", {"path": "README.md"}, {}, create_read_tool_definition(CWD), create_fake_tui(), CWD
        )
        component.update_result(
            {"content": [{"type": "text", "text": "hello"}], "details": None, "isError": False}, False
        )
        rendered = strip_ansi("\n".join(component.render(120)))
        assert len(re.findall(r"\bread\b", rendered)) == 1

    def test_inherits_missing_built_in_result_renderer_slot_from_the_built_in_tool(self):
        override_definition = replace(
            create_base_tool_definition("read"),
            render_call=lambda args, theme, context: Text("override call", 0, 0),
        )

        component = ToolExecutionComponent(
            "read", "tool-4b", {"path": "notes.txt"}, {}, override_definition, create_fake_tui(), CWD
        )
        component.update_result(
            {"content": [{"type": "text", "text": "hello"}], "details": None, "isError": False}, False
        )
        component.set_expanded(True)
        rendered = strip_ansi("\n".join(component.render(120)))
        assert "override call" in rendered
        assert "hello" in rendered

    def test_inherits_missing_built_in_call_renderer_slot_from_the_built_in_tool(self):
        override_definition = replace(
            create_base_tool_definition("read"),
            render_result=lambda result, options, theme, context: Text("override result", 0, 0),
        )

        component = ToolExecutionComponent(
            "read", "tool-4c", {"path": "README.md"}, {}, override_definition, create_fake_tui(), CWD
        )
        component.update_result(
            {"content": [{"type": "text", "text": "hello"}], "details": None, "isError": False}, False
        )
        rendered = strip_ansi("\n".join(component.render(120)))
        assert "read" in rendered
        assert "README.md" in rendered
        assert "override result" in rendered

    def test_uses_custom_renderers_for_built_in_overrides_that_reuse_built_in_definition_parameters(self):
        built_in_definition = create_read_tool_definition(CWD)
        component = ToolExecutionComponent(
            "read",
            "tool-4d",
            {"path": "README.md"},
            {},
            replace(
                built_in_definition,
                render_call=lambda args, theme, context: Text("override call", 0, 0),
                render_result=lambda result, options, theme, context: Text("override result", 0, 0),
            ),
            create_fake_tui(),
            CWD,
        )
        component.update_result(
            {"content": [{"type": "text", "text": "hello"}], "details": None, "isError": False}, False
        )
        rendered = strip_ansi("\n".join(component.render(120)))
        assert "override call" in rendered
        assert "override result" in rendered
        assert "read README.md" not in rendered

    def test_uses_custom_renderers_for_built_in_overrides_that_reuse_wrapped_built_in_tool_parameters(self):
        built_in_tool = create_read_tool(CWD)
        component = ToolExecutionComponent(
            "read",
            "tool-4e",
            {"path": "README.md"},
            {},
            replace(
                create_base_tool_definition("read"),
                parameters=built_in_tool.parameters,
                render_call=lambda args, theme, context: Text("wrapped override call", 0, 0),
                render_result=lambda result, options, theme, context: Text("wrapped override result", 0, 0),
            ),
            create_fake_tui(),
            CWD,
        )
        component.update_result(
            {"content": [{"type": "text", "text": "hello"}], "details": None, "isError": False}, False
        )
        rendered = strip_ansi("\n".join(component.render(120)))
        assert "wrapped override call" in rendered
        assert "wrapped override result" in rendered

    def test_shares_renderer_state_across_custom_call_and_result_slots(self):
        def render_call(_args, _theme, context):
            if context["state"].get("token") is None:
                context["state"]["token"] = "shared-token"
            return Text(f"custom call {context['state']['token']}", 0, 0)

        def render_result(_result, _options, _theme, context):
            return Text(f"custom result {context['state'].get('token')}", 0, 0)

        tool_definition = replace(create_base_tool_definition(), render_call=render_call, render_result=render_result)

        component = ToolExecutionComponent("custom_tool", "tool-5", {}, {}, tool_definition, create_fake_tui(), CWD)
        component.update_result({"content": [{"type": "text", "text": "done"}], "details": {}, "isError": False}, False)
        rendered = strip_ansi("\n".join(component.render(120)))
        assert "custom call shared-token" in rendered
        assert "custom result shared-token" in rendered

    def test_exposes_args_in_render_result_context(self):
        tool_definition = replace(
            create_base_tool_definition(),
            render_call=lambda args, theme, context: Text("call", 0, 0),
            render_result=lambda result, options, theme, context: Text(f"arg:{context['args']['foo']}", 0, 0),
        )

        component = ToolExecutionComponent(
            "custom_tool", "tool-5b", {"foo": "bar"}, {}, tool_definition, create_fake_tui(), CWD
        )
        component.update_result({"content": [{"type": "text", "text": "done"}], "details": {}, "isError": False}, False)
        rendered = strip_ansi("\n".join(component.render(120)))
        assert "arg:bar" in rendered

    def test_collapses_fallback_results_until_expanded(self):
        tool_definition = create_base_tool_definition()

        component = ToolExecutionComponent(
            "custom_tool", "tool-6", {"foo": "bar"}, {}, tool_definition, create_fake_tui(), CWD
        )
        output = "\n".join(f"line-{index + 1}" for index in range(15))
        component.update_result({"content": [{"type": "text", "text": output}], "details": {}, "isError": False}, False)

        collapsed = strip_ansi("\n".join(component.render(120)))
        assert "custom_tool" in collapsed
        assert "line-10" in collapsed
        assert "line-11" not in collapsed
        assert "5 more lines" in collapsed
        assert "to expand" in collapsed

        component.set_expanded(True)
        expanded = strip_ansi("\n".join(component.render(120)))
        assert "line-15" in expanded
        assert "more lines" not in expanded

    def test_trims_trailing_blank_display_lines_from_write_previews(self):
        component = ToolExecutionComponent(
            "write",
            "tool-7",
            {"path": "README.md", "content": "one\ntwo\n"},
            {},
            create_write_tool_definition(CWD),
            create_fake_tui(),
            CWD,
        )
        rendered = strip_ansi("\n".join(component.render(120)))
        assert "one" in rendered
        assert "two" in rendered
        assert "two\n\n" not in rendered

    def test_trims_trailing_blank_display_lines_from_read_results(self):
        component = ToolExecutionComponent(
            "read", "tool-8", {"path": "notes.txt"}, {}, create_read_tool_definition(CWD), create_fake_tui(), CWD
        )
        component.update_result(
            {"content": [{"type": "text", "text": "one\ntwo\n"}], "details": None, "isError": False}, False
        )
        component.set_expanded(True)
        rendered = strip_ansi("\n".join(component.render(120)))
        assert "one" in rendered
        assert "two" in rendered
        assert "two\n\n" not in rendered

    def test_does_not_syntax_highlight_read_errors_based_on_the_requested_file_path(self):
        component = ToolExecutionComponent(
            "read",
            "tool-read-error-highlighting",
            {"path": "config.exs", "offset": 120, "limit": 130},
            {},
            create_read_tool_definition(CWD),
            create_fake_tui(),
            CWD,
        )
        error = "Offset 120 is beyond end of file (96 lines total)"
        component.update_result({"content": [{"type": "text", "text": error}], "details": None, "isError": True}, False)

        rendered = "\n".join(component.render(120))
        assert error in strip_ansi(rendered)
        assert theme.fg("toolOutput", error) in rendered

    def test_collapses_ordinary_read_results_until_expanded(self):
        component = ToolExecutionComponent(
            "read",
            "tool-ordinary-read-collapsed",
            {"path": "notes.txt"},
            {},
            create_read_tool_definition(CWD),
            create_fake_tui(),
            CWD,
        )
        component.update_result(
            {"content": [{"type": "text", "text": "hidden content"}], "details": None, "isError": False}, False
        )

        collapsed = strip_ansi("\n".join(component.render(120)))
        assert "read" in collapsed
        assert "notes.txt" in collapsed
        assert "hidden content" not in collapsed

        component.set_expanded(True)
        expanded = strip_ansi("\n".join(component.render(120)))
        assert "hidden content" in expanded

    @pytest.mark.parametrize(
        ("title", "path", "content", "compact", "hidden", "absent"),
        [
            (
                "SKILL.md",
                os.path.join(CWD, "attio", "SKILL.md"),
                "---\nname: attio\ndescription: CRM helper\n---\n\n# Hidden skill instructions",
                "[skill] attio",
                "Hidden skill instructions",
                "read skill attio",
            ),
            (
                "AGENTS.md",
                os.path.join(CWD, ".pidrei", "AGENTS.md"),
                "Hidden resource instructions",
                "read resource .pidrei/AGENTS.md",
                "Hidden resource instructions",
                None,
            ),
            (
                "AGENTS.override.md",
                os.path.join(CWD, ".pidrei", "AGENTS.override.md"),
                "Hidden override instructions",
                "read resource .pidrei/AGENTS.override.md",
                "Hidden override instructions",
                None,
            ),
            (
                "outside AGENTS.md",
                os.path.abspath(os.path.join(CWD, "..", "AGENTS.md")),
                "Hidden outside resource instructions",
                f"read resource {os.path.abspath(os.path.join(CWD, '..', 'AGENTS.md'))}",
                "Hidden outside resource instructions",
                None,
            ),
            (
                "Pi documentation",
                get_readme_path(),
                "Hidden docs content",
                "read docs README.md",
                "Hidden docs content",
                None,
            ),
        ],
    )
    def test_renders_read_results_compactly_until_expanded(self, title, path, content, compact, hidden, absent):
        component = ToolExecutionComponent(
            "read",
            f"tool-compact-{title}",
            {"path": path},
            {},
            create_read_tool_definition(CWD),
            create_fake_tui(),
            CWD,
        )
        component.update_result(
            {"content": [{"type": "text", "text": content}], "details": None, "isError": False}, False
        )

        collapsed = strip_ansi("\n".join(component.render(120)))
        assert compact in collapsed
        assert hidden not in collapsed
        if absent:
            assert absent not in collapsed

        component.set_expanded(True)
        expanded = strip_ansi("\n".join(component.render(120)))
        assert hidden in expanded

    @pytest.mark.parametrize(
        ("title", "path", "compact"),
        [
            ("SKILL.md", os.path.join(CWD, "attio", "SKILL.md"), "[skill] attio:120-329"),
            ("Pi documentation", get_readme_path(), "read docs README.md:120-329"),
        ],
    )
    def test_shows_the_read_line_range_in_compact_reads_before_the_expand_hint(self, title, path, compact):
        component = ToolExecutionComponent(
            "read",
            f"tool-compact-range-{title}",
            {"path": path, "offset": 120, "limit": 210},
            {},
            create_read_tool_definition(CWD),
            create_fake_tui(),
            CWD,
        )

        collapsed = strip_ansi("\n".join(component.render(120)))
        assert compact in collapsed
        assert collapsed.index(":120-329") < collapsed.index("to expand")
