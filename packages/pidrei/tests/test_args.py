"""Mirrors pi coding-agent test/args.test.ts."""

from pidrei.cli.args import parse_args


class TestVersionFlag:
    def test_parses_version_flag(self):
        result = parse_args(["--version"])
        assert result.version is True

    def test_parses_v_shorthand(self):
        result = parse_args(["-v"])
        assert result.version is True

    def test_version_takes_precedence_over_other_args(self):
        result = parse_args(["--version", "--help", "some message"])
        assert result.version is True
        assert result.help is True
        assert "some message" in result.messages


class TestHelpFlag:
    def test_parses_help_flag(self):
        result = parse_args(["--help"])
        assert result.help is True

    def test_parses_h_shorthand(self):
        result = parse_args(["-h"])
        assert result.help is True


class TestPrintFlag:
    def test_parses_print_flag(self):
        result = parse_args(["--print"])
        assert result.print is True

    def test_parses_p_shorthand(self):
        result = parse_args(["-p"])
        assert result.print is True

    def test_parses_prompt_after_p_even_when_it_starts_with_yaml_frontmatter(self):
        prompt = "---\ntitle: hello\n---\nSay hi."
        result = parse_args(["-p", prompt])
        assert result.print is True
        assert result.messages == [prompt]
        assert len(result.unknown_flags) == 0

    def test_does_not_consume_options_after_p_as_prompts(self):
        result = parse_args(["-p", "--provider", "openai", "Say hi."])
        assert result.print is True
        assert result.provider == "openai"
        assert result.messages == ["Say hi."]


class TestContinueFlag:
    def test_parses_continue_flag(self):
        result = parse_args(["--continue"])
        assert result.continue_ is True

    def test_parses_c_shorthand(self):
        result = parse_args(["-c"])
        assert result.continue_ is True


class TestResumeFlag:
    def test_parses_resume_flag(self):
        result = parse_args(["--resume"])
        assert result.resume is True

    def test_parses_r_shorthand(self):
        result = parse_args(["-r"])
        assert result.resume is True


class TestFlagsWithValues:
    def test_parses_provider(self):
        result = parse_args(["--provider", "openai"])
        assert result.provider == "openai"

    def test_parses_model(self):
        result = parse_args(["--model", "gpt-4o"])
        assert result.model == "gpt-4o"

    def test_parses_api_key(self):
        result = parse_args(["--api-key", "sk-test-key"])
        assert result.api_key == "sk-test-key"

    def test_parses_system_prompt(self):
        result = parse_args(["--system-prompt", "You are a helpful assistant"])
        assert result.system_prompt == "You are a helpful assistant"

    def test_parses_append_system_prompt(self):
        result = parse_args(["--append-system-prompt", "Additional context"])
        assert result.append_system_prompt == ["Additional context"]

    def test_parses_multiple_append_system_prompt_flags(self):
        result = parse_args(["--append-system-prompt", "Context A", "--append-system-prompt", "Context B"])
        assert result.append_system_prompt == ["Context A", "Context B"]

    def test_parses_mode(self):
        result = parse_args(["--mode", "json"])
        assert result.mode == "json"

    def test_parses_mode_rpc(self):
        result = parse_args(["--mode", "rpc"])
        assert result.mode == "rpc"

    def test_parses_session(self):
        result = parse_args(["--session", "/path/to/session.jsonl"])
        assert result.session == "/path/to/session.jsonl"

    def test_parses_session_id(self):
        result = parse_args(["--session-id", "orchestrated-session"])
        assert result.session_id == "orchestrated-session"

    def test_parses_fork(self):
        result = parse_args(["--fork", "1234abcd"])
        assert result.fork == "1234abcd"
        assert result.messages == []

    def test_parses_export(self):
        result = parse_args(["--export", "session.jsonl"])
        assert result.export == "session.jsonl"

    def test_parses_thinking(self):
        result = parse_args(["--thinking", "high"])
        assert result.thinking == "high"

    def test_parses_models_as_comma_separated_list(self):
        result = parse_args(["--models", "gpt-4o,claude-sonnet,gemini-pro"])
        assert result.models == ["gpt-4o", "claude-sonnet", "gemini-pro"]


class TestNameFlag:
    def test_parses_name_flag_with_value(self):
        result = parse_args(["--name", "my-session"])
        assert result.name == "my-session"

    def test_parses_n_shorthand(self):
        result = parse_args(["-n", "quick-session"])
        assert result.name == "quick-session"

    def test_preserves_empty_values_for_main_validation(self):
        result = parse_args(["--name", ""])
        assert result.name == ""

    def test_reports_missing_value(self):
        result = parse_args(["--name"])
        assert result.diagnostics == [{"type": "error", "message": "--name requires a value"}]

    def test_works_alongside_other_flags(self):
        result = parse_args(["--name", "named-run", "--print", "--model", "gpt-4o", "hello"])
        assert result.name == "named-run"
        assert result.print is True
        assert result.model == "gpt-4o"
        assert result.messages == ["hello"]


class TestNoSessionFlag:
    def test_parses_no_session_flag(self):
        result = parse_args(["--no-session"])
        assert result.no_session is True


class TestExtensionFlag:
    def test_parses_single_extension(self):
        result = parse_args(["--extension", "./my-extension.ts"])
        assert result.extensions == ["./my-extension.ts"]

    def test_parses_e_shorthand(self):
        result = parse_args(["-e", "./my-extension.ts"])
        assert result.extensions == ["./my-extension.ts"]

    def test_parses_multiple_extension_flags(self):
        result = parse_args(["--extension", "./ext1.ts", "-e", "./ext2.ts"])
        assert result.extensions == ["./ext1.ts", "./ext2.ts"]


class TestNoExtensionsFlag:
    def test_parses_no_extensions_flag(self):
        result = parse_args(["--no-extensions"])
        assert result.no_extensions is True

    def test_parses_no_extensions_with_explicit_e_flags(self):
        result = parse_args(["--no-extensions", "-e", "foo.ts", "-e", "bar.ts"])
        assert result.no_extensions is True
        assert result.extensions == ["foo.ts", "bar.ts"]


class TestSkillFlag:
    def test_parses_single_skill(self):
        result = parse_args(["--skill", "./skill-dir"])
        assert result.skills == ["./skill-dir"]

    def test_parses_multiple_skill_flags(self):
        result = parse_args(["--skill", "./skill-a", "--skill", "./skill-b"])
        assert result.skills == ["./skill-a", "./skill-b"]


class TestPromptTemplateFlag:
    def test_parses_single_prompt_template(self):
        result = parse_args(["--prompt-template", "./prompts"])
        assert result.prompt_templates == ["./prompts"]

    def test_parses_multiple_prompt_template_flags(self):
        result = parse_args(["--prompt-template", "./one", "--prompt-template", "./two"])
        assert result.prompt_templates == ["./one", "./two"]


class TestThemeFlag:
    def test_parses_single_theme(self):
        result = parse_args(["--theme", "./theme.json"])
        assert result.themes == ["./theme.json"]

    def test_parses_multiple_theme_flags(self):
        result = parse_args(["--theme", "./dark.json", "--theme", "./light.json"])
        assert result.themes == ["./dark.json", "./light.json"]


class TestNoSkillsFlag:
    def test_parses_no_skills_flag(self):
        result = parse_args(["--no-skills"])
        assert result.no_skills is True


class TestNoPromptTemplatesFlag:
    def test_parses_no_prompt_templates_flag(self):
        result = parse_args(["--no-prompt-templates"])
        assert result.no_prompt_templates is True


class TestNoThemesFlag:
    def test_parses_no_themes_flag(self):
        result = parse_args(["--no-themes"])
        assert result.no_themes is True


class TestNoContextFilesFlag:
    def test_parses_no_context_files_flag(self):
        result = parse_args(["--no-context-files"])
        assert result.no_context_files is True

    def test_parses_nc_shorthand(self):
        result = parse_args(["-nc"])
        assert result.no_context_files is True


class TestProjectApprovalFlags:
    def test_parses_approve(self):
        result = parse_args(["--approve"])
        assert result.project_trust_override is True

    def test_parses_a_shorthand(self):
        result = parse_args(["-a"])
        assert result.project_trust_override is True

    def test_parses_no_approve(self):
        result = parse_args(["--no-approve"])
        assert result.project_trust_override is False

    def test_parses_na_shorthand(self):
        result = parse_args(["-na"])
        assert result.project_trust_override is False


class TestVerboseFlag:
    def test_parses_verbose_flag(self):
        result = parse_args(["--verbose"])
        assert result.verbose is True


class TestOfflineFlag:
    def test_parses_offline_flag(self):
        result = parse_args(["--offline"])
        assert result.offline is True


class TestToolFlags:
    def test_parses_no_tools_flag(self):
        result = parse_args(["--no-tools"])
        assert result.no_tools is True

    def test_parses_nt_shorthand(self):
        result = parse_args(["-nt"])
        assert result.no_tools is True

    def test_parses_no_builtin_tools_flag(self):
        result = parse_args(["--no-builtin-tools"])
        assert result.no_builtin_tools is True

    def test_parses_nbt_shorthand(self):
        result = parse_args(["-nbt"])
        assert result.no_builtin_tools is True

    def test_parses_tools_flag(self):
        result = parse_args(["--tools", "read,bash"])
        assert result.tools == ["read", "bash"]

    def test_parses_t_shorthand(self):
        result = parse_args(["-t", "read,bash"])
        assert result.tools == ["read", "bash"]

    def test_parses_exclude_tools_flag(self):
        result = parse_args(["--exclude-tools", "read,bash"])
        assert result.exclude_tools == ["read", "bash"]

    def test_parses_xt_shorthand(self):
        result = parse_args(["-xt", "read,bash"])
        assert result.exclude_tools == ["read", "bash"]

    def test_parses_no_tools_with_explicit_tools_flags(self):
        result = parse_args(["--no-tools", "--tools", "read,bash"])
        assert result.no_tools is True
        assert result.tools == ["read", "bash"]

    def test_parses_no_builtin_tools_with_explicit_tools_flags(self):
        result = parse_args(["--no-builtin-tools", "--tools", "read,bash"])
        assert result.no_builtin_tools is True
        assert result.tools == ["read", "bash"]


class TestMessagesAndFileArgs:
    def test_parses_plain_text_messages(self):
        result = parse_args(["hello", "world"])
        assert result.messages == ["hello", "world"]

    def test_parses_file_arguments(self):
        result = parse_args(["@README.md", "@src/main.ts"])
        assert result.file_args == ["README.md", "src/main.ts"]

    def test_parses_mixed_messages_and_file_args(self):
        result = parse_args(["@file.txt", "explain this", "@image.png"])
        assert result.file_args == ["file.txt", "image.png"]
        assert result.messages == ["explain this"]

    def test_captures_unknown_long_flags_with_string_values(self):
        result = parse_args(["--unknown-flag", "message"])
        assert result.messages == []
        assert result.unknown_flags.get("unknown-flag") == "message"

    def test_captures_unknown_boolean_long_flags(self):
        result = parse_args(["--unknown-flag"])
        assert result.unknown_flags.get("unknown-flag") is True

    def test_captures_unknown_long_flags_with_equals_syntax(self):
        result = parse_args(["--unknown-flag=value"])
        assert result.unknown_flags.get("unknown-flag") == "value"


class TestComplexCombinations:
    def test_parses_multiple_flags_together(self):
        result = parse_args(
            [
                "--provider",
                "anthropic",
                "--model",
                "claude-sonnet",
                "--print",
                "--thinking",
                "high",
                "@prompt.md",
                "Do the task",
            ]
        )
        assert result.provider == "anthropic"
        assert result.model == "claude-sonnet"
        assert result.print is True
        assert result.thinking == "high"
        assert result.file_args == ["prompt.md"]
        assert result.messages == ["Do the task"]
