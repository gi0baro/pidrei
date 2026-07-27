"""Mirror of pi coding-agent test/prompt-templates.test.ts."""

import pytest

from pidrei.core.prompt_templates import (
    PromptTemplate,
    expand_prompt_template,
    load_prompt_templates,
    parse_command_args,
    substitute_args,
)
from pidrei.core.source_info import create_synthetic_source_info


class TestSubstituteArgs:
    def test_replaces_arguments_with_all_args_joined(self):
        assert substitute_args("Test: $ARGUMENTS", ["a", "b", "c"]) == "Test: a b c"

    def test_replaces_at_with_all_args_joined(self):
        assert substitute_args("Test: $@", ["a", "b", "c"]) == "Test: a b c"

    def test_replaces_at_and_arguments_identically(self):
        args = ["foo", "bar", "baz"]
        assert substitute_args("Test: $@", args) == substitute_args("Test: $ARGUMENTS", args)

    def test_does_not_recursively_substitute_patterns_in_argument_values(self):
        assert substitute_args("$ARGUMENTS", ["$1", "$ARGUMENTS"]) == "$1 $ARGUMENTS"
        assert substitute_args("$@", ["$100", "$1"]) == "$100 $1"
        assert substitute_args("$ARGUMENTS", ["$100", "$1"]) == "$100 $1"

    def test_supports_mixed_positional_and_arguments(self):
        assert substitute_args("$1: $ARGUMENTS", ["prefix", "a", "b"]) == "prefix: prefix a b"

    def test_supports_mixed_positional_and_at(self):
        assert substitute_args("$1: $@", ["prefix", "a", "b"]) == "prefix: prefix a b"

    def test_handles_empty_arguments_array(self):
        assert substitute_args("Test: $ARGUMENTS", []) == "Test: "
        assert substitute_args("Test: $@", []) == "Test: "
        assert substitute_args("Test: $1", []) == "Test: "

    def test_handles_multiple_occurrences(self):
        assert substitute_args("$ARGUMENTS and $ARGUMENTS", ["a", "b"]) == "a b and a b"
        assert substitute_args("$@ and $@", ["a", "b"]) == "a b and a b"
        assert substitute_args("$@ and $ARGUMENTS", ["a", "b"]) == "a b and a b"

    def test_handles_special_characters_in_arguments(self):
        assert substitute_args("$1 $2: $ARGUMENTS", ["arg100", "@user"]) == "arg100 @user: arg100 @user"

    def test_handles_out_of_range_numbered_placeholders(self):
        assert substitute_args("$1 $2 $3 $4 $5", ["a", "b"]) == "a b   "

    def test_handles_unicode_characters(self):
        assert substitute_args("$ARGUMENTS", ["日本語", "🎉", "café"]) == "日本語 🎉 café"

    def test_preserves_newlines_and_tabs_in_argument_values(self):
        assert substitute_args("$1 $2", ["line1\nline2", "tab\tthere"]) == "line1\nline2 tab\tthere"

    def test_handles_consecutive_dollar_patterns(self):
        assert substitute_args("$1$2", ["a", "b"]) == "ab"

    def test_handles_quoted_arguments_with_spaces(self):
        assert substitute_args("$ARGUMENTS", ["first arg", "second arg"]) == "first arg second arg"

    def test_handles_single_argument(self):
        assert substitute_args("Test: $ARGUMENTS", ["only"]) == "Test: only"
        assert substitute_args("Test: $@", ["only"]) == "Test: only"

    def test_handles_zero_index(self):
        assert substitute_args("$0", ["a", "b"]) == ""

    def test_handles_decimal_number_in_pattern(self):
        assert substitute_args("$1.5", ["a"]) == "a.5"

    def test_handles_arguments_as_part_of_word(self):
        assert substitute_args("pre$ARGUMENTS", ["a", "b"]) == "prea b"
        assert substitute_args("pre$@", ["a", "b"]) == "prea b"

    def test_handles_empty_arguments_in_middle_of_list(self):
        assert substitute_args("$ARGUMENTS", ["a", "", "c"]) == "a  c"

    def test_handles_trailing_and_leading_spaces_in_arguments(self):
        assert substitute_args("$ARGUMENTS", ["  leading  ", "trailing  "]) == "  leading   trailing  "

    def test_handles_argument_containing_pattern_partially(self):
        assert substitute_args("Prefix $ARGUMENTS suffix", ["ARGUMENTS"]) == "Prefix ARGUMENTS suffix"

    def test_handles_non_matching_patterns(self):
        assert substitute_args("$A $$ $ $ARGS", ["a"]) == "$A $$ $ $ARGS"

    def test_handles_case_variations(self):
        assert substitute_args("$arguments $Arguments $ARGUMENTS", ["a", "b"]) == "$arguments $Arguments a b"

    def test_handles_both_syntaxes_in_same_command_with_same_result(self):
        args = ["x", "y", "z"]
        result1 = substitute_args("$@ and $ARGUMENTS", args)
        result2 = substitute_args("$ARGUMENTS and $@", args)
        assert result1 == result2
        assert result1 == "x y z and x y z"

    def test_handles_very_long_argument_lists(self):
        args = [f"arg{i}" for i in range(100)]
        assert substitute_args("$ARGUMENTS", args) == " ".join(args)

    def test_handles_numbered_placeholders_with_single_digit(self):
        assert substitute_args("$1 $2 $3", ["a", "b", "c"]) == "a b c"

    def test_handles_numbered_placeholders_with_multiple_digits(self):
        args = [f"val{i}" for i in range(15)]
        assert substitute_args("$10 $12 $15", args) == "val9 val11 val14"

    def test_handles_escaped_dollar_signs_backslash_preserved(self):
        # Note: No escape mechanism exists - backslash is treated literally
        assert substitute_args("Price: \\$100", []) == "Price: \\"

    def test_handles_mixed_numbered_and_wildcard_placeholders(self):
        assert substitute_args("$1: $@ ($ARGUMENTS)", ["first", "second", "third"]) == (
            "first: first second third (first second third)"
        )

    def test_handles_command_with_no_placeholders(self):
        assert substitute_args("Just plain text", ["a", "b"]) == "Just plain text"

    def test_handles_command_with_only_placeholders(self):
        assert substitute_args("$1 $2 $@", ["a", "b", "c"]) == "a b a b c"


class TestSubstituteArgsPositionalDefaults:
    def test_uses_default_when_positional_arg_is_missing(self):
        assert substitute_args("List exactly ${1:-7} next steps", []) == "List exactly 7 next steps"

    def test_supports_defaults_for_all_arguments(self):
        template = "${@:-default}\n${ARGUMENTS:-default}"
        assert substitute_args(template, []) == "default\ndefault"
        assert substitute_args(template, ["This", "would", "be", "the", "arguments"]) == (
            "This would be the arguments\nThis would be the arguments"
        )

    def test_uses_positional_arg_when_present(self):
        assert substitute_args("List exactly ${1:-7} next steps", ["3"]) == "List exactly 3 next steps"

    def test_uses_default_when_positional_arg_is_empty(self):
        assert substitute_args("Mode: ${1:-brief}", [""]) == "Mode: brief"

    def test_supports_multiple_positional_defaults(self):
        assert substitute_args("${1:-7} ${2:-brief}", []) == "7 brief"
        assert substitute_args("${1:-7} ${2:-brief}", ["3"]) == "3 brief"
        assert substitute_args("${1:-7} ${2:-brief}", ["3", "verbose"]) == "3 verbose"

    def test_does_not_recursively_substitute_patterns_in_arg_values(self):
        assert substitute_args("${1:-7}", ["$ARGUMENTS"]) == "$ARGUMENTS"
        assert substitute_args("${1:-7}", ["$1"]) == "$1"

    def test_does_not_recursively_substitute_patterns_in_default_values(self):
        assert substitute_args("${1:-$ARGUMENTS}", ["a", "b"]) == "a"
        assert substitute_args("${3:-$ARGUMENTS}", ["a", "b"]) == "$ARGUMENTS"

    def test_supports_defaults_with_spaces(self):
        assert substitute_args("${1:-seven steps}", []) == "seven steps"

    def test_supports_out_of_range_positional_defaults(self):
        assert substitute_args("${3:-fallback}", ["a", "b"]) == "fallback"

    def test_mixes_positional_defaults_with_existing_placeholders(self):
        assert substitute_args("$1 ${2:-x} $ARGUMENTS", ["a"]) == "a x a"


class TestSubstituteArgsArraySlicing:
    def test_slices_from_index(self):
        assert substitute_args("${@:2}", ["a", "b", "c", "d"]) == "b c d"
        assert substitute_args("${@:1}", ["a", "b", "c"]) == "a b c"
        assert substitute_args("${@:3}", ["a", "b", "c", "d"]) == "c d"

    def test_slices_with_length(self):
        assert substitute_args("${@:2:2}", ["a", "b", "c", "d"]) == "b c"
        assert substitute_args("${@:1:1}", ["a", "b", "c"]) == "a"
        assert substitute_args("${@:3:1}", ["a", "b", "c", "d"]) == "c"
        assert substitute_args("${@:2:3}", ["a", "b", "c", "d", "e"]) == "b c d"

    def test_handles_out_of_range_slices(self):
        assert substitute_args("${@:99}", ["a", "b"]) == ""
        assert substitute_args("${@:5}", ["a", "b"]) == ""
        assert substitute_args("${@:10:5}", ["a", "b"]) == ""

    def test_handles_zero_length_slices(self):
        assert substitute_args("${@:2:0}", ["a", "b", "c"]) == ""
        assert substitute_args("${@:1:0}", ["a", "b"]) == ""

    def test_handles_length_exceeding_array(self):
        assert substitute_args("${@:2:99}", ["a", "b", "c"]) == "b c"
        assert substitute_args("${@:1:10}", ["a", "b"]) == "a b"

    def test_processes_slice_before_simple_at(self):
        assert substitute_args("${@:2} vs $@", ["a", "b", "c"]) == "b c vs a b c"
        assert substitute_args("First: ${@:1:1}, All: $@", ["x", "y", "z"]) == "First: x, All: x y z"

    def test_does_not_recursively_substitute_slice_patterns_in_args(self):
        assert substitute_args("${@:1}", ["${@:2}", "test"]) == "${@:2} test"
        assert substitute_args("${@:2}", ["a", "${@:3}", "c"]) == "${@:3} c"

    def test_handles_mixed_usage_with_positional_args(self):
        assert substitute_args("$1: ${@:2}", ["cmd", "arg1", "arg2"]) == "cmd: arg1 arg2"
        assert substitute_args("$1 $2 ${@:3}", ["a", "b", "c", "d"]) == "a b c d"

    def test_treats_slice_zero_as_all_args(self):
        assert substitute_args("${@:0}", ["a", "b", "c"]) == "a b c"

    def test_handles_empty_args_array(self):
        assert substitute_args("${@:2}", []) == ""
        assert substitute_args("${@:1}", []) == ""

    def test_handles_single_arg_array(self):
        assert substitute_args("${@:1}", ["only"]) == "only"
        assert substitute_args("${@:2}", ["only"]) == ""

    def test_handles_slice_in_middle_of_text(self):
        assert substitute_args("Process ${@:2} with $1", ["tool", "file1", "file2"]) == "Process file1 file2 with tool"

    def test_handles_multiple_slices_in_one_template(self):
        assert substitute_args("${@:1:1} and ${@:2}", ["a", "b", "c"]) == "a and b c"
        assert substitute_args("${@:1:2} vs ${@:3:2}", ["a", "b", "c", "d", "e"]) == "a b vs c d"

    def test_handles_quoted_arguments_in_slices(self):
        assert substitute_args("${@:2}", ["cmd", "first arg", "second arg"]) == "first arg second arg"

    def test_handles_special_characters_in_sliced_args(self):
        assert substitute_args("${@:2}", ["cmd", "$100", "@user", "#tag"]) == "$100 @user #tag"

    def test_handles_unicode_in_sliced_args(self):
        assert substitute_args("${@:1}", ["日本語", "🎉", "café"]) == "日本語 🎉 café"

    def test_combines_positional_slice_and_wildcard_placeholders(self):
        template = "Run $1 on ${@:2:2}, then process $@"
        args = ["eslint", "file1.ts", "file2.ts", "file3.ts"]
        assert substitute_args(template, args) == (
            "Run eslint on file1.ts file2.ts, then process eslint file1.ts file2.ts file3.ts"
        )

    def test_handles_slice_with_no_spacing(self):
        assert substitute_args("prefix${@:2}suffix", ["a", "b", "c"]) == "prefixb csuffix"

    def test_handles_large_slice_lengths_gracefully(self):
        args = [f"arg{i + 1}" for i in range(10)]
        assert substitute_args("${@:5:100}", args) == "arg5 arg6 arg7 arg8 arg9 arg10"


class TestParseCommandArgs:
    def test_parses_simple_space_separated_arguments(self):
        assert parse_command_args("a b c") == ["a", "b", "c"]

    def test_parses_quoted_arguments_with_spaces(self):
        assert parse_command_args('"first arg" second') == ["first arg", "second"]

    def test_parses_single_quoted_arguments(self):
        assert parse_command_args("'first arg' second") == ["first arg", "second"]

    def test_parses_mixed_quote_styles(self):
        assert parse_command_args('"double" \'single\' "double again"') == ["double", "single", "double again"]

    def test_handles_empty_string(self):
        assert parse_command_args("") == []

    def test_handles_extra_spaces(self):
        assert parse_command_args("a  b   c") == ["a", "b", "c"]

    def test_handles_tabs_as_separators(self):
        assert parse_command_args("a\tb\tc") == ["a", "b", "c"]

    def test_handles_quoted_empty_string(self):
        # Note: Empty quotes are skipped by current implementation
        assert parse_command_args('"" " "') == [" "]

    def test_handles_arguments_with_special_characters(self):
        assert parse_command_args("$100 @user #tag") == ["$100", "@user", "#tag"]

    def test_handles_unicode_characters(self):
        assert parse_command_args("日本語 🎉 café") == ["日本語", "🎉", "café"]

    def test_handles_newlines_in_quoted_arguments(self):
        assert parse_command_args('"line1\nline2" second') == ["line1\nline2", "second"]

    def test_treats_unquoted_newlines_as_separators(self):
        assert parse_command_args("label-2\n\nHere is some description #2.") == [
            "label-2",
            "Here",
            "is",
            "some",
            "description",
            "#2.",
        ]

    def test_collapses_mixed_unquoted_whitespace(self):
        assert parse_command_args("a\n\n\tb  c") == ["a", "b", "c"]

    def test_handles_escaped_quotes_inside_quoted_strings(self):
        # Note: This implementation doesn't handle escaped quotes - backslash is literal
        assert parse_command_args('"quoted \\"text\\""') == ["quoted \\text\\"]

    def test_handles_trailing_spaces(self):
        assert parse_command_args("a b c   ") == ["a", "b", "c"]

    def test_handles_leading_spaces(self):
        assert parse_command_args("   a b c") == ["a", "b", "c"]


def make_template(name, content):
    return PromptTemplate(
        name=name,
        description="test",
        content=content,
        source_info=create_synthetic_source_info(f"/tmp/{name}.md", source="local"),
        file_path=f"/tmp/{name}.md",
    )


class TestExpandPromptTemplate:
    def test_splits_template_arguments_on_unquoted_newlines(self):
        result = expand_prompt_template(
            "/arg-test label-2\n\nHere is some description #2.",
            [make_template("arg-test", "- arg1: $1\n- rest: ${@:2}")],
        )
        assert result == "- arg1: label-2\n- rest: Here is some description #2."

    def test_supports_template_command_separated_from_args_by_newline(self):
        result = expand_prompt_template("/arg-test\nlabel-2", [make_template("arg-test", "arg1: $1")])
        assert result == "arg1: label-2"


class TestIntegration:
    def test_parses_and_substitutes_together_correctly(self):
        input = 'Button "onClick handler" "disabled support"'
        args = parse_command_args(input)
        template = "Create component $1 with features: $ARGUMENTS"
        assert substitute_args(template, args) == (
            "Create component Button with features: Button onClick handler disabled support"
        )

    def test_handles_the_example_from_readme(self):
        input = 'Button "onClick handler" "disabled support"'
        args = parse_command_args(input)
        template = "Create a React component named $1 with features: $ARGUMENTS"
        assert substitute_args(template, args) == (
            "Create a React component named Button with features: Button onClick handler disabled support"
        )

    def test_produces_same_result_with_at_and_arguments(self):
        args = parse_command_args("feature1 feature2 feature3")
        assert substitute_args("Implement: $@", args) == substitute_args("Implement: $ARGUMENTS", args)


class TestLoadPromptTemplatesArgumentHint:
    @pytest.fixture
    def prompts_dir(self, tmp_dir):
        # `tmp_dir`, not pytest's `tmp_path`: these tests are tonio tests now
        # that `load_prompt_templates` is async, and the tonio plugin cannot
        # wrap a yield fixture.
        return tmp_dir / "prompts"

    def write_template(self, prompts_dir, name, content):
        prompts_dir.mkdir(parents=True, exist_ok=True)
        (prompts_dir / f"{name}.md").write_text(content, encoding="utf-8")

    def load(self, tmp_dir, prompts_dir):
        return load_prompt_templates(
            cwd=str(tmp_dir / "cwd"),
            agent_dir=str(tmp_dir / "agent"),
            prompt_paths=[str(prompts_dir)],
            include_defaults=False,
        )

    @pytest.mark.tonio
    async def test_parses_required_argument_hint_from_frontmatter(self, tmp_dir, prompts_dir):
        self.write_template(
            prompts_dir,
            "pr",
            "---\ndescription: Review PRs from URLs with structured issue and code analysis\n"
            'argument-hint: "<PR-URL>"\n---\nYou are given one or more GitHub PR URLs: $@',
        )

        templates = await self.load(tmp_dir, prompts_dir)
        pr = next(t for t in templates if t.name == "pr")
        assert pr.argument_hint == "<PR-URL>"
        assert pr.description == "Review PRs from URLs with structured issue and code analysis"

    @pytest.mark.tonio
    async def test_parses_optional_argument_hint_from_frontmatter(self, tmp_dir, prompts_dir):
        self.write_template(
            prompts_dir,
            "wr",
            "---\ndescription: Finish the current task end-to-end with changelog, commit, and push\n"
            'argument-hint: "[instructions]"\n---\nWrap it. Additional instructions: $ARGUMENTS',
        )

        templates = await self.load(tmp_dir, prompts_dir)
        wr = next(t for t in templates if t.name == "wr")
        assert wr.argument_hint == "[instructions]"
        assert wr.description == "Finish the current task end-to-end with changelog, commit, and push"

    @pytest.mark.tonio
    async def test_leaves_argument_hint_none_when_not_specified(self, tmp_dir, prompts_dir):
        self.write_template(
            prompts_dir,
            "cl",
            "---\ndescription: Audit changelog entries before release\n---\n"
            "Audit changelog entries for all commits since the last release.",
        )

        templates = await self.load(tmp_dir, prompts_dir)
        cl = next(t for t in templates if t.name == "cl")
        assert cl.argument_hint is None

    @pytest.mark.tonio
    async def test_ignores_empty_argument_hint(self, tmp_dir, prompts_dir):
        self.write_template(
            prompts_dir,
            "empty-hint",
            '---\ndescription: A command with empty hint\nargument-hint: ""\n---\nDo something',
        )

        templates = await self.load(tmp_dir, prompts_dir)
        tmpl = next(t for t in templates if t.name == "empty-hint")
        assert tmpl.argument_hint is None

    @pytest.mark.tonio
    async def test_preserves_argument_hint_with_special_characters(self, tmp_dir, prompts_dir):
        self.write_template(
            prompts_dir,
            "is",
            "---\ndescription: Analyze GitHub issues (bugs or feature requests)\n"
            'argument-hint: "<issue>"\n---\nAnalyze GitHub issue(s): $ARGUMENTS',
        )

        templates = await self.load(tmp_dir, prompts_dir)
        is_template = next(t for t in templates if t.name == "is")
        assert is_template.argument_hint == "<issue>"
