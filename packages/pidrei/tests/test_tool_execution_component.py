"""Mirror of pi coding-agent test/tool-execution-component.test.ts."""

import os
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
import tonio.colored as tonio

from pidrei.config import get_readme_path
from pidrei.core.extensions.types import ToolDefinition
from pidrei.core.tools.bash import BashExecResult, create_bash_tool_definition
from pidrei.core.tools.read import create_read_tool, create_read_tool_definition
from pidrei.core.tools.renderers import bash as bash_renderers, with_built_in_renderers
from pidrei.core.tools.write import create_write_tool_definition
from pidrei.modes.interactive.components import ToolExecutionComponent, tool_execution
from pidrei.modes.interactive.theme import init_theme_sync, theme
from pidrei.utils.ansi import strip_ansi
from pidrei_tui import Text, TuiMouseEvent, reset_capabilities_cache, set_capabilities


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
    # `post_ui` applies inline: there is no UI owner in these tests.
    return SimpleNamespace(request_render=lambda: None, post_ui=lambda fn: fn())


@pytest.fixture(autouse=True)
def _theme():
    init_theme_sync("dark")


CWD = os.getcwd()


class TestToolExecutionComponentParity:
    # Issue #8577: ignore conversions that finish after the image was replaced.
    @pytest.mark.tonio
    async def test_keeps_the_final_tool_image_when_a_partial_image_conversion_finishes_late(self, monkeypatch):
        release = threading.Event()
        returned = tonio.Event()

        def convert_to_png(_data, _mime_type):
            # Runs on the blocking pool, so waiting here does not stall the runtime.
            release.wait(5)
            returned.set()
            return {"data": "converted-partial", "mimeType": "image/png"}

        monkeypatch.setattr(tool_execution, "convert_to_png", convert_to_png)
        rendered = tonio.Event()
        applied = tonio.Event()

        def post_ui(fn) -> None:
            # Stands in for the UI owner: runs the late conversion's apply, then
            # signals that it has run, so the checks below see its outcome.
            fn()
            applied.set()

        set_capabilities({"images": "kitty", "trueColor": True, "hyperlinks": True})
        try:
            component = ToolExecutionComponent(
                "custom_tool",
                "tool-image-race",
                {},
                {},
                None,
                SimpleNamespace(request_render=rendered.set, post_ui=post_ui),
                CWD,
            )

            component.update_result(
                {"content": [{"type": "image", "data": "partial-jpeg", "mimeType": "image/jpeg"}], "isError": False},
                True,
            )
            component.update_result(
                {"content": [{"type": "image", "data": "final-png", "mimeType": "image/png"}], "isError": False}
            )
            assert "final-png" in "\n".join(component.render(120))

            release.set()
            await returned.wait(5)
            assert returned.is_set()
            # The late conversion's apply has run; had it been applied, it would
            # have re-rendered.
            await applied.wait(5)
            assert applied.is_set()
            assert not rendered.is_set()

            output = "\n".join(component.render(120))
            assert "final-png" in output
            assert "converted-partial" not in output
        finally:
            release.set()
            reset_capabilities_cache()

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
            with_built_in_renderers("edit", override_definition),
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
        exec_started = tonio.Event()
        release_exec = tonio.Event()

        # The exec stays pending until released, so the check below runs while
        # it is in flight rather than inside a fixed window.
        class Operations:
            async def exec(self, _command, _cwd, *, on_data, cancel=None, timeout=None, env=None):
                exec_started.set()
                await release_exec.wait()
                return BashExecResult(exit_code=0)

        tool = create_bash_tool_definition(CWD, operations=Operations(), expose_session_environment=False)
        coro = tool.execute("tool-bash-1", {"command": "sleep 10"}, None, lambda update: updates.append(update), None)
        # pi asserts the update is emitted before the exec settles
        task = tonio.spawn(coro)
        await exec_started.wait(5)
        assert exec_started.is_set()
        assert len(updates) == 1
        first = updates[0]
        content = first["content"] if isinstance(first, dict) else first.content
        assert content == []
        release_exec.set()
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

    # Issue #9628: keep short durations precise and make long shell durations readable.
    @pytest.mark.tonio
    @pytest.mark.parametrize(
        ("ms", "formatted"),
        [
            (0, "0.0s"),
            (4_200, "4.2s"),
            (59_900, "59.9s"),
            (59_999, "60.0s"),
            (60_000, "1m 0s"),
            (90_900, "1m 30s"),
            (1_592_200, "26m 32s"),
            (3_599_999, "59m 59s"),
            (3_600_000, "1h 0m 0s"),
            (7_384_900, "2h 3m 4s"),
        ],
    )
    async def test_bash_renderer_formats_durations_while_running_and_after_completion(self, monkeypatch, ms, formatted):
        # pi drives `Date.now()` with vi.useFakeTimers; the renderer reads the clock
        # through its module's `time`, which is swapped for a settable one here.
        clock = {"now_s": 0.0}
        monkeypatch.setattr(bash_renderers, "time", SimpleNamespace(time=lambda: clock["now_s"]))
        component = ToolExecutionComponent(
            "bash",
            "tool-bash-duration",
            {"command": "long-running-command"},
            {},
            create_bash_tool_definition(CWD, expose_session_environment=False),
            create_fake_tui(),
            CWD,
        )
        component.mark_execution_started()
        component.update_result({"content": [], "isError": False}, True)

        clock["now_s"] += ms / 1000
        component.invalidate()
        running = strip_ansi("\n".join(component.render(120)))

        component.update_result({"content": [], "isError": False}, False)
        completed = strip_ansi("\n".join(component.render(120)))

        clock["now_s"] += 1
        component.invalidate()
        assert strip_ansi("\n".join(component.render(120))) == completed
        assert f"Elapsed {formatted}" in running
        assert f"Took {formatted}" in completed

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
            "read",
            "tool-4b",
            {"path": "notes.txt"},
            {},
            with_built_in_renderers("read", override_definition),
            create_fake_tui(),
            CWD,
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
            "read",
            "tool-4c",
            {"path": "README.md"},
            {},
            with_built_in_renderers("read", override_definition),
            create_fake_tui(),
            CWD,
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

    @pytest.mark.tonio
    async def test_expands_a_collapsed_tool_result_when_clicked(self):
        component = ToolExecutionComponent(
            "read",
            "tool-click-expand",
            {"path": "notes.txt"},
            {},
            create_read_tool_definition(CWD),
            create_fake_tui(),
            CWD,
        )
        component.update_result(
            {"content": [{"type": "text", "text": "hidden content"}], "details": None, "isError": False}, False
        )
        width = 120
        lines = component.render(width)
        result_row = next((index for index, line in enumerate(lines) if "notes.txt" in strip_ansi(line)), -1)
        assert result_row >= 0
        event = TuiMouseEvent(
            type="click",
            button="left",
            x=2,
            y=result_row,
            screen_x=2,
            screen_y=result_row,
            width=width,
            height=len(lines),
            click_count=1,
        )
        result = await component.handle_mouse(event)
        assert result is not None and result.handled is True
        assert "hidden content" in strip_ansi("\n".join(component.render(width)))

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
