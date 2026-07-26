"""Mirror of pi coding-agent test/export-html-whitespace.test.ts."""

import os
import re
from types import SimpleNamespace

from pidrei.config import get_export_template_dir
from pidrei.core.export_html import ansi_lines_to_html, create_tool_html_renderer


class TestExportHtmlToolOutputWhitespace:
    def test_preserves_whitespace_for_plain_text_tool_output_lines_without_preserving_template_whitespace(self):
        with open(os.path.join(get_export_template_dir(), "template.css"), encoding="utf-8") as f:
            css = f.read()

        assert re.search(
            r"\.output-preview > div:not\(\.expand-hint\),\s*"
            r"\.output-full > div:not\(\.expand-hint\) \{[\s\S]*?white-space:\s*pre-wrap;",
            css,
        )
        assert re.search(r"\.ansi-line\s*\{[\s\S]*?white-space:\s*pre;", css)
        assert not re.search(r"\.output-preview,\s*\.output-full\s*\{[\s\S]*?white-space:\s*pre-wrap;", css)

    def test_does_not_insert_source_whitespace_between_ansi_rendered_lines(self):
        assert ansi_lines_to_html(["one", "two"]) == '<div class="ansi-line">one</div><div class="ansi-line">two</div>'

    def test_trims_tui_spacing_lines_from_custom_tool_result_html(self):
        component = SimpleNamespace(
            render=lambda width: ["", "\x1b[31mone\x1b[0m", "two", ""],
            invalidate=lambda: None,
        )
        tool = SimpleNamespace(
            name="custom",
            label="custom",
            description="custom",
            render_call=None,
            render_result=lambda result, options, theme, context: component,
        )
        renderer = create_tool_html_renderer({"getToolDefinition": lambda name: tool, "theme": object(), "cwd": "/tmp"})

        rendered = renderer.render_result("id", "custom", [], None, False)
        assert rendered["expanded"] == (
            '<div class="ansi-line"><span style="color:#800000">one</span></div><div class="ansi-line">two</div>'
        )
