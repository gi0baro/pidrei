"""Mirror of pi coding-agent test/system-prompt.test.ts (harness name pi → pidrei)."""

import os

import pytest

from pidrei.core.skills import Skill
from pidrei.core.source_info import create_synthetic_source_info
from pidrei.core.system_prompt import BuildSystemPromptOptions, ContextFile, build_system_prompt


TEST_SKILL = Skill(
    name="test-skill",
    description="A test skill.",
    file_path="/skills/test-skill/SKILL.md",
    base_dir="/skills/test-skill",
    source_info=create_synthetic_source_info("/skills/test-skill/SKILL.md", source="test"),
    disable_model_invocation=False,
)


class TestEmptyTools:
    def test_shows_none_for_empty_tools_list(self):
        prompt = build_system_prompt(BuildSystemPromptOptions(selected_tools=[], cwd=os.getcwd()))
        assert "<tools>\n(none)\n" in prompt

    def test_shows_file_paths_guideline_even_with_no_tools(self):
        prompt = build_system_prompt(BuildSystemPromptOptions(selected_tools=[], cwd=os.getcwd()))
        assert "Show file paths clearly" in prompt


class TestPromptStructure:
    def test_keeps_the_default_and_custom_prompt_prefixes_exact(self):
        default_prompt = build_system_prompt(
            BuildSystemPromptOptions(cwd="/tmp", selected_tools=[], context_files=[], skills=[])
        )
        custom_prompt = build_system_prompt(
            BuildSystemPromptOptions(
                custom_prompt="You are Exact.", cwd="/tmp", selected_tools=[], context_files=[], skills=[]
            )
        )

        assert default_prompt.startswith("You are an expert coding assistant operating inside pidrei")
        assert custom_prompt.startswith("You are Exact.\n\n<cwd>")

    def test_preserves_an_exact_forced_prompt_without_sections(self):
        assert build_system_prompt(BuildSystemPromptOptions(force_system_prompt="exact", cwd="/tmp")) == "exact"

    def test_maps_appended_instructions_and_project_context_to_stable_sections(self):
        prompt = build_system_prompt(
            BuildSystemPromptOptions(
                custom_prompt="You are Exact.",
                append_system_prompt="Additional instructions.",
                context_files=[ContextFile(path="/tmp/AGENTS.md", content="Project instructions.")],
                selected_tools=[],
                skills=[],
                cwd="/tmp",
            )
        )

        assert "<addendum>\nAdditional instructions.\n</addendum>" in prompt
        assert (
            '<project_context>\nProject-specific instructions and guidelines:\n\n<project_instructions path="/tmp/AGENTS.md">'
        ) in prompt
        assert "<cwd>\n/tmp\n</cwd>" in prompt


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


class TestSkills:
    @pytest.mark.parametrize("custom_prompt", [None, "Custom system prompt"], ids=["default prompt", "custom prompt"])
    def test_includes_skills_with_only_bash(self, custom_prompt):
        prompt = build_system_prompt(
            BuildSystemPromptOptions(
                custom_prompt=custom_prompt,
                selected_tools=["bash"],
                context_files=[],
                skills=[TEST_SKILL],
                cwd=os.getcwd(),
            )
        )

        assert "<skills>" in prompt
        assert "<available_skills>" in prompt
        assert "<name>test-skill</name>" in prompt
        assert "Use bash to load a skill's file" in prompt

    def test_omits_skills_without_read_or_bash(self):
        prompt = build_system_prompt(
            BuildSystemPromptOptions(selected_tools=["write"], context_files=[], skills=[TEST_SKILL], cwd=os.getcwd())
        )

        assert "<available_skills>" not in prompt
