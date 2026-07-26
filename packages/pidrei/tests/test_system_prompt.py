"""Mirror of pi coding-agent test/system-prompt.test.ts (harness name pi → pidrei)."""

import os

from pidrei.core.system_prompt import BuildSystemPromptOptions, build_system_prompt


class TestEmptyTools:
    def test_shows_none_for_empty_tools_list(self):
        prompt = build_system_prompt(BuildSystemPromptOptions(selected_tools=[], cwd=os.getcwd()))
        assert "Available tools:\n(none)" in prompt

    def test_shows_file_paths_guideline_even_with_no_tools(self):
        prompt = build_system_prompt(BuildSystemPromptOptions(selected_tools=[], cwd=os.getcwd()))
        assert "Show file paths clearly" in prompt


class TestDefaultTools:
    def test_includes_all_default_tools_when_snippets_are_provided(self):
        prompt = build_system_prompt(
            BuildSystemPromptOptions(
                tool_snippets={
                    "read": "Read file contents",
                    "bash": "Execute bash commands",
                    "edit": "Make surgical edits",
                    "write": "Create or overwrite files",
                },
                cwd=os.getcwd(),
            )
        )

        assert "- read:" in prompt
        assert "- bash:" in prompt
        assert "- edit:" in prompt
        assert "- write:" in prompt

    def test_instructs_models_to_resolve_docs_and_examples_under_absolute_base_paths(self):
        prompt = build_system_prompt(BuildSystemPromptOptions(cwd=os.getcwd()))

        assert (
            "- When reading pidrei docs or examples, resolve docs/... under Additional docs and "
            "examples/... under Examples, not the current working directory"
        ) in prompt
        assert "environment variables (docs/environment-variables.md)" in prompt


class TestCustomToolSnippets:
    def test_includes_custom_tools_when_prompt_snippet_is_provided(self):
        prompt = build_system_prompt(
            BuildSystemPromptOptions(
                selected_tools=["read", "dynamic_tool"],
                tool_snippets={"dynamic_tool": "Run dynamic test behavior"},
                cwd=os.getcwd(),
            )
        )

        assert "- dynamic_tool: Run dynamic test behavior" in prompt

    def test_omits_custom_tools_when_prompt_snippet_is_not_provided(self):
        prompt = build_system_prompt(BuildSystemPromptOptions(selected_tools=["read", "dynamic_tool"], cwd=os.getcwd()))

        assert "dynamic_tool" not in prompt


class TestPromptGuidelines:
    def test_appends_prompt_guidelines_to_default_guidelines(self):
        prompt = build_system_prompt(
            BuildSystemPromptOptions(
                selected_tools=["read", "dynamic_tool"],
                prompt_guidelines=["Use dynamic_tool for project summaries."],
                cwd=os.getcwd(),
            )
        )

        assert "- Use dynamic_tool for project summaries." in prompt

    def test_deduplicates_and_trims_prompt_guidelines(self):
        prompt = build_system_prompt(
            BuildSystemPromptOptions(
                selected_tools=["read", "dynamic_tool"],
                prompt_guidelines=[
                    "Use dynamic_tool for summaries.",
                    "  Use dynamic_tool for summaries.  ",
                    "   ",
                ],
                cwd=os.getcwd(),
            )
        )

        assert prompt.count("- Use dynamic_tool for summaries.") == 1
