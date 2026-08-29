"""Mirror of pi coding-agent test/resource-loader.test.ts (Phase 3 subset).

Extension-loading and theme tests are deferred with their systems (Phase 5 /
Phase 4); the missing-extension-path diagnostic, skills/prompts discovery,
context files, SYSTEM.md/APPEND_SYSTEM.md, trust gating, overrides, and
extend_resources are mirrored. HOME is redirected per test (the auto-discovery
also scans ~/.agents/skills, which must stay hermetic).
"""

import contextlib
import os
from pathlib import Path

import pytest

from pidrei.core import resource_loader
from pidrei.core.resource_loader import DefaultResourceLoader, SourcedPath, load_project_context_files
from pidrei.core.settings_manager import SettingsManager
from pidrei.core.skills import LoadSkillsResult, Skill
from pidrei.core.source_info import PathMetadata, create_synthetic_source_info


@contextlib.contextmanager
def fake_home(path):
    original = os.environ.get("HOME")
    os.environ["HOME"] = str(path)
    try:
        yield
    finally:
        if original is None:
            del os.environ["HOME"]
        else:
            os.environ["HOME"] = original


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def skill_md(name: str, description: str) -> str:
    return f"---\nname: {name}\ndescription: {description}\n---\nSkill content here."


@pytest.fixture
def dirs(tmp_path):
    agent_dir = tmp_path / "agent"
    cwd = tmp_path / "project"
    home = tmp_path / "home"
    agent_dir.mkdir(parents=True)
    cwd.mkdir(parents=True)
    home.mkdir(parents=True)
    return tmp_path, agent_dir, cwd, home


class TestReload:
    @pytest.mark.tonio
    async def test_initializes_with_empty_results_before_reload(self, dirs):
        _tmp, agent_dir, cwd, _home = dirs
        loader = DefaultResourceLoader(cwd=str(cwd), agent_dir=str(agent_dir))

        assert loader.get_extensions().extensions == []
        assert loader.get_skills().skills == []
        assert loader.get_prompts().prompts == []

    @pytest.mark.tonio
    async def test_discovers_skills_from_agent_dir(self, dirs):
        _tmp, agent_dir, cwd, home = dirs
        write(agent_dir / "skills" / "test-skill.md", skill_md("test-skill", "A test skill"))

        loader = DefaultResourceLoader(cwd=str(cwd), agent_dir=str(agent_dir))
        with fake_home(home):
            await loader.reload()

        assert any(skill.name == "test-skill" for skill in loader.get_skills().skills)

    @pytest.mark.tonio
    async def test_ignores_extra_markdown_files_in_auto_discovered_skill_dirs(self, dirs):
        _tmp, agent_dir, cwd, home = dirs
        skill_dir = agent_dir / "skills" / "pi-skills" / "browser-tools"
        write(skill_dir / "SKILL.md", skill_md("browser-tools", "Browser tools"))
        write(skill_dir / "EFFICIENCY.md", "No frontmatter here")

        loader = DefaultResourceLoader(cwd=str(cwd), agent_dir=str(agent_dir))
        with fake_home(home):
            await loader.reload()

        result = loader.get_skills()
        assert any(skill.name == "browser-tools" for skill in result.skills)
        assert not any(d.path and d.path.endswith("EFFICIENCY.md") for d in result.diagnostics)

    @pytest.mark.tonio
    async def test_discovers_prompts_from_agent_dir(self, dirs):
        _tmp, agent_dir, cwd, home = dirs
        write(agent_dir / "prompts" / "test-prompt.md", "---\ndescription: A test prompt\n---\nPrompt content.")

        loader = DefaultResourceLoader(cwd=str(cwd), agent_dir=str(agent_dir))
        with fake_home(home):
            await loader.reload()

        assert any(prompt.name == "test-prompt" for prompt in loader.get_prompts().prompts)

    @pytest.mark.tonio
    async def test_prefers_project_resources_over_user_on_name_collisions(self, dirs):
        _tmp, agent_dir, cwd, home = dirs
        user_prompt_path = agent_dir / "prompts" / "commit.md"
        project_prompt_path = cwd / ".pidrei" / "prompts" / "commit.md"
        write(user_prompt_path, "User prompt")
        write(project_prompt_path, "Project prompt")

        user_skill_path = agent_dir / "skills" / "collision-skill" / "SKILL.md"
        project_skill_path = cwd / ".pidrei" / "skills" / "collision-skill" / "SKILL.md"
        write(user_skill_path, skill_md("collision-skill", "user"))
        write(project_skill_path, skill_md("collision-skill", "project"))

        loader = DefaultResourceLoader(cwd=str(cwd), agent_dir=str(agent_dir))
        with fake_home(home):
            await loader.reload()

        prompt = next(p for p in loader.get_prompts().prompts if p.name == "commit")
        assert prompt.file_path == str(project_prompt_path)

        skill = next(s for s in loader.get_skills().skills if s.name == "collision-skill")
        assert skill.file_path == str(project_skill_path)

    @pytest.mark.tonio
    async def test_reports_missing_explicit_extension_paths(self, dirs):
        _tmp, agent_dir, cwd, home = dirs
        missing = str(cwd / "missing-extension.py")

        loader = DefaultResourceLoader(cwd=str(cwd), agent_dir=str(agent_dir), additional_extension_paths=[missing])
        with fake_home(home):
            await loader.reload()

        errors = loader.get_extensions().errors
        assert any(error.path == missing and "does not exist" in error.error for error in errors)

    @pytest.mark.tonio
    async def test_honors_overrides_for_auto_discovered_resources(self, dirs):
        _tmp, agent_dir, cwd, home = dirs
        settings_manager = SettingsManager.in_memory()
        settings_manager.set_skill_paths(["-skills/skip-skill"])
        settings_manager.set_prompt_template_paths(["-prompts/skip.md"])

        write(agent_dir / "skills" / "skip-skill" / "SKILL.md", skill_md("skip-skill", "Skip me"))
        write(agent_dir / "prompts" / "skip.md", "Skip prompt")
        write(agent_dir / "skills" / "keep-skill" / "SKILL.md", skill_md("keep-skill", "Keep me"))

        loader = DefaultResourceLoader(cwd=str(cwd), agent_dir=str(agent_dir), settings_manager=settings_manager)
        with fake_home(home):
            await loader.reload()

        assert not any(skill.name == "skip-skill" for skill in loader.get_skills().skills)
        assert any(skill.name == "keep-skill" for skill in loader.get_skills().skills)
        assert not any(prompt.name == "skip" for prompt in loader.get_prompts().prompts)

    @pytest.mark.tonio
    async def test_discovers_agents_md_context_files(self, dirs):
        _tmp, agent_dir, cwd, home = dirs
        write(cwd / "AGENTS.md", "# Project Guidelines\n\nBe helpful.")

        loader = DefaultResourceLoader(cwd=str(cwd), agent_dir=str(agent_dir))
        with fake_home(home):
            await loader.reload()

        assert any("AGENTS.md" in file.path for file in loader.get_agents_files())

    @pytest.mark.tonio
    async def test_prefers_agents_override_md_per_directory_while_preserving_ancestor_layering(self, dirs):
        _tmp, agent_dir, cwd, home = dirs
        nested_cwd = cwd / "service"
        nested_cwd.mkdir()
        write(agent_dir / "AGENTS.md", "global instructions")
        write(agent_dir / "AGENTS.override.md", "global override")
        write(cwd / "AGENTS.md", "project instructions")
        write(nested_cwd / "AGENTS.md", "service instructions")
        write(nested_cwd / "AGENTS.override.md", "service override")

        loader = DefaultResourceLoader(cwd=str(nested_cwd), agent_dir=str(agent_dir))
        with fake_home(home):
            await loader.reload()

        assert [(file.path, file.content) for file in loader.get_agents_files()] == [
            (str(agent_dir / "AGENTS.override.md"), "global override"),
            (str(cwd / "AGENTS.md"), "project instructions"),
            (str(nested_cwd / "AGENTS.override.md"), "service override"),
        ]

    @pytest.mark.tonio
    async def test_ignores_context_file_candidates_that_are_directories(self, dirs):
        _tmp, agent_dir, cwd, home = dirs
        (cwd / "AGENTS.override.md").mkdir()
        (cwd / "AGENTS.md").mkdir()
        write(cwd / "CLAUDE.md", "Fallback instructions")
        # Hand swap (predates tonio 0.9.14; `monkeypatch` works in tonio tests now).
        warnings = []
        original_warn = resource_loader._warn
        resource_loader._warn = warnings.append
        try:
            loader = DefaultResourceLoader(cwd=str(cwd), agent_dir=str(agent_dir))
            with fake_home(home):
                await loader.reload()
        finally:
            resource_loader._warn = original_warn

        assert any(
            file.path == str(cwd / "CLAUDE.md") and file.content == "Fallback instructions"
            for file in loader.get_agents_files()
        )
        assert not any(str(cwd / "AGENTS.md") in warning for warning in warnings)
        assert not any(str(cwd / "AGENTS.override.md") in warning for warning in warnings)

    @pytest.mark.tonio
    async def test_skips_context_file_discovery_when_no_context_files_is_true(self, dirs):
        _tmp, agent_dir, cwd, home = dirs
        write(cwd / "AGENTS.override.md", "# Override Guidelines\n\nBe helpful.")
        write(cwd / "AGENTS.md", "# Project Guidelines\n\nBe helpful.")
        write(cwd / "CLAUDE.md", "# Claude Guidelines\n\nBe helpful.")

        loader = DefaultResourceLoader(cwd=str(cwd), agent_dir=str(agent_dir), no_context_files=True)
        with fake_home(home):
            await loader.reload()

        assert loader.get_agents_files() == []

    @pytest.mark.tonio
    async def test_discovers_system_md_from_project_config_dir(self, dirs):
        _tmp, agent_dir, cwd, home = dirs
        write(cwd / ".pidrei" / "SYSTEM.md", "You are a helpful assistant.")

        loader = DefaultResourceLoader(cwd=str(cwd), agent_dir=str(agent_dir))
        with fake_home(home):
            await loader.reload()

        assert loader.get_system_prompt() == "You are a helpful assistant."

    @pytest.mark.tonio
    async def test_skips_project_resources_that_require_trust_when_project_is_not_trusted(self, dirs):
        _tmp, agent_dir, cwd, home = dirs
        write(cwd / ".pidrei" / "SYSTEM.md", "Project system prompt.")
        write(agent_dir / "SYSTEM.md", "Global system prompt.")
        write(agent_dir / "AGENTS.md", "Global instructions")
        write(cwd / "AGENTS.md", "Project instructions")
        write(cwd / ".pidrei" / "skills" / "project-skill" / "SKILL.md", skill_md("project-skill", "Project skill"))
        write(cwd / ".pidrei" / "prompts" / "project.md", "Project prompt")
        settings_manager = await SettingsManager.create(str(cwd), str(agent_dir), project_trusted=False)

        loader = DefaultResourceLoader(cwd=str(cwd), agent_dir=str(agent_dir), settings_manager=settings_manager)
        with fake_home(home):
            await loader.reload()

        assert loader.get_system_prompt() == "Global system prompt."
        agents_files = loader.get_agents_files()
        assert any(file.path == str(agent_dir / "AGENTS.md") for file in agents_files)
        assert any(file.path == str(cwd / "AGENTS.md") for file in agents_files)
        assert not any(skill.name == "project-skill" for skill in loader.get_skills().skills)
        assert not any(prompt.name == "project" for prompt in loader.get_prompts().prompts)

    @pytest.mark.tonio
    async def test_discovers_append_system_md(self, dirs):
        _tmp, agent_dir, cwd, home = dirs
        write(cwd / ".pidrei" / "APPEND_SYSTEM.md", "Additional instructions.")

        loader = DefaultResourceLoader(cwd=str(cwd), agent_dir=str(agent_dir))
        with fake_home(home):
            await loader.reload()

        assert "Additional instructions." in loader.get_append_system_prompt()

    @pytest.mark.tonio
    async def test_resolve_project_trust_callback_controls_project_resources(self, dirs):
        _tmp, agent_dir, cwd, home = dirs
        write(cwd / ".pidrei" / "SYSTEM.md", "Project system prompt.")
        write(cwd / ".pidrei" / "skills" / "project-skill" / "SKILL.md", skill_md("project-skill", "Project skill"))

        loader = DefaultResourceLoader(cwd=str(cwd), agent_dir=str(agent_dir))

        async def trust(extensions_result):
            assert extensions_result.extensions == []
            return True

        with fake_home(home):
            await loader.reload(resolve_project_trust=trust)

        assert loader.get_system_prompt() == "Project system prompt."
        assert any(skill.name == "project-skill" for skill in loader.get_skills().skills)

        async def distrust(_extensions_result):
            return False

        with fake_home(home):
            await loader.reload(resolve_project_trust=distrust)

        assert loader.get_system_prompt() is None
        assert not any(skill.name == "project-skill" for skill in loader.get_skills().skills)


class TestExtendResources:
    @pytest.mark.tonio
    async def test_loads_skills_and_prompts_with_extension_metadata(self, dirs):
        tmp, agent_dir, cwd, home = dirs
        extra_skill_dir = tmp / "extra-skills" / "extra-skill"
        skill_path = extra_skill_dir / "SKILL.md"
        write(skill_path, skill_md("extra-skill", "Extra skill"))

        extra_prompt_dir = tmp / "extra-prompts"
        prompt_path = extra_prompt_dir / "extra.md"
        write(prompt_path, "---\ndescription: Extra prompt\n---\nExtra prompt content")

        loader = DefaultResourceLoader(cwd=str(cwd), agent_dir=str(agent_dir))
        with fake_home(home):
            await loader.reload()

            await loader.extend_resources(
                skill_paths=[
                    SourcedPath(
                        path=str(extra_skill_dir),
                        metadata=PathMetadata(
                            source="extension:extra",
                            scope="temporary",
                            origin="top-level",
                            base_dir=str(extra_skill_dir),
                        ),
                    )
                ],
                prompt_paths=[
                    SourcedPath(
                        path=str(prompt_path),
                        metadata=PathMetadata(
                            source="extension:extra",
                            scope="temporary",
                            origin="top-level",
                            base_dir=str(extra_prompt_dir),
                        ),
                    )
                ],
            )

        loaded_skill = next((s for s in loader.get_skills().skills if s.name == "extra-skill"), None)
        assert loaded_skill is not None
        assert loaded_skill.source_info.source == "extension:extra"
        assert loaded_skill.source_info.path == str(skill_path)

        loaded_prompt = next((p for p in loader.get_prompts().prompts if p.name == "extra"), None)
        assert loaded_prompt is not None
        assert loaded_prompt.source_info.source == "extension:extra"
        assert loaded_prompt.source_info.path == str(prompt_path)

    @pytest.mark.tonio
    async def test_keeps_package_metadata_for_skills_and_prompts(self, dirs):
        # Regression: extension discovery used to drop package scope/source,
        # collapsing every autocomplete source tag to [t]. See pi issue #6968.
        # pi's case also covers themes; extend_resources has no theme surface
        # in pidrei, so skills and prompts carry the mirror.
        tmp, agent_dir, cwd, home = dirs
        package_root = tmp / "metadata-pkg"
        write(package_root / "skills" / "package-skill" / "SKILL.md", skill_md("package-skill", "Package skill"))
        write(
            package_root / "prompts" / "package-prompt.md",
            "---\ndescription: Package prompt\n---\nPackage prompt content",
        )

        extension_resource_dir = tmp / "extension-resources"
        extension_skill_dir = extension_resource_dir / "extension-skill"
        write(extension_skill_dir / "SKILL.md", skill_md("extension-skill", "Extension skill"))
        extension_prompts_dir = extension_resource_dir / "prompts"
        write(
            extension_prompts_dir / "extension-prompt.md",
            "---\ndescription: Extension prompt\n---\nExtension prompt content",
        )

        settings_manager = SettingsManager.in_memory({"packages": [str(package_root)]})
        loader = DefaultResourceLoader(cwd=str(cwd), agent_dir=str(agent_dir), settings_manager=settings_manager)
        with fake_home(home):
            await loader.reload()

            extension_metadata = PathMetadata(source="extension:discovery", scope="temporary", origin="top-level")
            await loader.extend_resources(
                skill_paths=[SourcedPath(path=str(extension_skill_dir), metadata=extension_metadata)],
                prompt_paths=[SourcedPath(path=str(extension_prompts_dir), metadata=extension_metadata)],
            )

        package_skill = next((s for s in loader.get_skills().skills if s.name == "package-skill"), None)
        assert package_skill is not None
        assert package_skill.source_info.source == str(package_root)
        assert package_skill.source_info.scope == "user"
        assert package_skill.source_info.origin == "package"

        package_prompt = next((p for p in loader.get_prompts().prompts if p.name == "package-prompt"), None)
        assert package_prompt is not None
        assert package_prompt.source_info.source == str(package_root)
        assert package_prompt.source_info.scope == "user"
        assert package_prompt.source_info.origin == "package"

        extension_skill = next((s for s in loader.get_skills().skills if s.name == "extension-skill"), None)
        assert extension_skill is not None
        assert extension_skill.source_info.source == "extension:discovery"

        extension_prompt = next((p for p in loader.get_prompts().prompts if p.name == "extension-prompt"), None)
        assert extension_prompt is not None
        assert extension_prompt.source_info.source == "extension:discovery"

    @pytest.mark.tonio
    async def test_loads_extension_resources_returned_as_file_urls(self, dirs):
        tmp, agent_dir, cwd, home = dirs
        extra_skill_dir = tmp / "extra skills" / "file-url-skill"
        skill_path = extra_skill_dir / "SKILL.md"
        write(skill_path, skill_md("file-url-skill", "File URL skill"))

        loader = DefaultResourceLoader(cwd=str(cwd), agent_dir=str(agent_dir))
        with fake_home(home):
            await loader.reload()

            await loader.extend_resources(
                skill_paths=[
                    SourcedPath(
                        path=Path(extra_skill_dir).as_uri(),
                        metadata=PathMetadata(
                            source="extension:file-url",
                            scope="temporary",
                            origin="top-level",
                            base_dir=str(extra_skill_dir),
                        ),
                    )
                ]
            )

        result = loader.get_skills()
        assert result.diagnostics == []
        loaded_skill = next((s for s in result.skills if s.name == "file-url-skill"), None)
        assert loaded_skill is not None
        assert loaded_skill.file_path == str(skill_path)
        assert loaded_skill.source_info.source == "extension:file-url"


class TestNoSkillsOption:
    @pytest.mark.tonio
    async def test_skips_skill_discovery_when_no_skills_is_true(self, dirs):
        _tmp, agent_dir, cwd, home = dirs
        write(agent_dir / "skills" / "test-skill.md", skill_md("test-skill", "A test skill"))

        loader = DefaultResourceLoader(cwd=str(cwd), agent_dir=str(agent_dir), no_skills=True)
        with fake_home(home):
            await loader.reload()

        assert loader.get_skills().skills == []

    @pytest.mark.tonio
    async def test_still_loads_additional_skill_paths_when_no_skills_is_true(self, dirs):
        tmp, agent_dir, cwd, home = dirs
        custom_skill_dir = tmp / "custom-skills"
        write(custom_skill_dir / "custom.md", skill_md("custom", "Custom skill"))

        loader = DefaultResourceLoader(
            cwd=str(cwd), agent_dir=str(agent_dir), no_skills=True, additional_skill_paths=[str(custom_skill_dir)]
        )
        with fake_home(home):
            await loader.reload()

        assert any(skill.name == "custom" for skill in loader.get_skills().skills)


class TestOverrideFunctions:
    @pytest.mark.tonio
    async def test_applies_skills_override(self, dirs):
        _tmp, agent_dir, cwd, home = dirs
        injected_skill = Skill(
            name="injected",
            description="Injected skill",
            file_path="/fake/path",
            base_dir="/fake",
            source_info=create_synthetic_source_info("/fake/path", source="custom"),
            disable_model_invocation=False,
        )
        loader = DefaultResourceLoader(
            cwd=str(cwd),
            agent_dir=str(agent_dir),
            skills_override=lambda _base: LoadSkillsResult(skills=[injected_skill], diagnostics=[]),
        )
        with fake_home(home):
            await loader.reload()

        skills = loader.get_skills().skills
        assert len(skills) == 1
        assert skills[0].name == "injected"

    @pytest.mark.tonio
    async def test_applies_system_prompt_override(self, dirs):
        _tmp, agent_dir, cwd, home = dirs
        loader = DefaultResourceLoader(
            cwd=str(cwd), agent_dir=str(agent_dir), system_prompt_override=lambda _base: "Custom system prompt"
        )
        with fake_home(home):
            await loader.reload()

        assert loader.get_system_prompt() == "Custom system prompt"


class TestSystemPromptSources:
    """pi's "system prompt sources" cases (#7266)."""

    @pytest.mark.tonio
    async def test_exposes_discovered_project_system_md_as_the_system_prompt_source(self, dirs):
        _tmp, agent_dir, cwd, home = dirs
        system_prompt_path = cwd / ".pidrei" / "SYSTEM.md"
        write(system_prompt_path, "Project system prompt.")

        loader = DefaultResourceLoader(cwd=str(cwd), agent_dir=str(agent_dir))
        with fake_home(home):
            await loader.reload()

        assert loader.get_system_prompt() == "Project system prompt."
        source = loader.get_system_prompt_source()
        assert source is not None and source.path == str(system_prompt_path)

    @pytest.mark.tonio
    async def test_exposes_discovered_global_system_md_as_the_system_prompt_source(self, dirs):
        _tmp, agent_dir, cwd, home = dirs
        system_prompt_path = agent_dir / "SYSTEM.md"
        write(system_prompt_path, "Global system prompt.")

        loader = DefaultResourceLoader(cwd=str(cwd), agent_dir=str(agent_dir))
        with fake_home(home):
            await loader.reload()

        assert loader.get_system_prompt() == "Global system prompt."
        source = loader.get_system_prompt_source()
        assert source is not None and source.path == str(system_prompt_path)

    @pytest.mark.tonio
    async def test_does_not_expose_literal_system_prompt_text_as_a_source(self, dirs):
        _tmp, agent_dir, cwd, home = dirs
        loader = DefaultResourceLoader(cwd=str(cwd), agent_dir=str(agent_dir), system_prompt="Literal system prompt.")
        with fake_home(home):
            await loader.reload()

        assert loader.get_system_prompt() == "Literal system prompt."
        assert loader.get_system_prompt_source() is None

    @pytest.mark.tonio
    async def test_exposes_file_backed_system_prompt_options_as_a_source(self, dirs):
        tmp, agent_dir, cwd, home = dirs
        system_prompt_path = tmp / "custom-system.md"
        write(system_prompt_path, "Custom system prompt.")

        loader = DefaultResourceLoader(cwd=str(cwd), agent_dir=str(agent_dir), system_prompt=str(system_prompt_path))
        with fake_home(home):
            await loader.reload()

        assert loader.get_system_prompt() == "Custom system prompt."
        source = loader.get_system_prompt_source()
        assert source is not None and source.path == str(system_prompt_path)

    @pytest.mark.tonio
    async def test_exposes_discovered_append_system_md_as_an_append_system_prompt_source(self, dirs):
        _tmp, agent_dir, cwd, home = dirs
        append_system_prompt_path = cwd / ".pidrei" / "APPEND_SYSTEM.md"
        write(append_system_prompt_path, "Project append prompt.")

        loader = DefaultResourceLoader(cwd=str(cwd), agent_dir=str(agent_dir))
        with fake_home(home):
            await loader.reload()

        assert loader.get_append_system_prompt() == ["Project append prompt."]
        assert [source.path for source in loader.get_append_system_prompt_sources()] == [str(append_system_prompt_path)]

    @pytest.mark.tonio
    async def test_keeps_only_file_backed_append_prompts_as_sources(self, dirs):
        tmp, agent_dir, cwd, home = dirs
        append_system_prompt_path = tmp / "custom-append.md"
        write(append_system_prompt_path, "Custom append prompt.")

        loader = DefaultResourceLoader(
            cwd=str(cwd),
            agent_dir=str(agent_dir),
            append_system_prompt=[str(append_system_prompt_path), "Literal append prompt."],
        )
        with fake_home(home):
            await loader.reload()

        assert loader.get_append_system_prompt() == ["Custom append prompt.", "Literal append prompt."]
        assert [source.path for source in loader.get_append_system_prompt_sources()] == [str(append_system_prompt_path)]


class TestNestedWorktreeContextDedup:
    """loadProjectContextFiles — nested worktree dedup (pi #7221)."""

    @staticmethod
    def link_worktree(main_dir: Path, worktree_dir: Path, name: str) -> None:
        """Builds a linked-worktree skeleton (no git binary needed): the main
        repo's `.git/worktrees/<name>/` holds `HEAD` plus a `commondir` pointing
        back at the main `.git`, and the worktree's working tree carries a
        `.git` *file* whose `gitdir:` resolves to it."""
        git_dir = main_dir / ".git" / "worktrees" / name
        git_dir.mkdir(parents=True, exist_ok=True)
        # The main repo's own `.git` is a real git dir with a HEAD, as git writes it.
        write(main_dir / ".git" / "HEAD", "ref: refs/heads/main\n")
        write(git_dir / "HEAD", "ref: refs/heads/feat\n")
        # commondir is relative to the worktree gitdir and points at the main .git.
        write(git_dir / "commondir", "../..")
        write(worktree_dir / ".git", f"gitdir: {git_dir}\n")

    def setup_nested_worktree(self, tmp: Path) -> tuple[Path, Path, Path, Path]:
        """Main repo at <tmp>/outer/main with a linked worktree at main/worktrees/feat."""
        outer = tmp / "outer"
        main = outer / "main"
        worktree = main / "worktrees" / "feat"
        worktree_src = worktree / "src"
        worktree_src.mkdir(parents=True, exist_ok=True)
        self.link_worktree(main, worktree, "feat")
        return outer, main, worktree, worktree_src

    @pytest.mark.tonio
    async def test_skips_the_main_repos_duplicate_when_the_worktree_root_has_its_own_context(self, dirs):
        tmp, agent_dir, _cwd, _home = dirs
        _outer, main, worktree, worktree_src = self.setup_nested_worktree(tmp)
        write(main / "AGENTS.md", "main repo instructions")
        write(worktree / "AGENTS.md", "worktree instructions")

        files = await load_project_context_files(cwd=str(worktree_src), agent_dir=str(agent_dir))

        assert [f.content for f in files] == ["worktree instructions"]

    @pytest.mark.tonio
    async def test_still_inherits_the_main_repos_context_when_the_worktree_root_has_none(self, dirs):
        tmp, agent_dir, _cwd, _home = dirs
        _outer, main, _worktree, worktree_src = self.setup_nested_worktree(tmp)
        write(main / "AGENTS.md", "main repo instructions")

        files = await load_project_context_files(cwd=str(worktree_src), agent_dir=str(agent_dir))

        assert [f.content for f in files] == ["main repo instructions"]

    @pytest.mark.tonio
    async def test_only_skips_the_same_filename_not_a_differently_named_context_file(self, dirs):
        # The repo tracks CLAUDE.md; the worktree adds an AGENTS.md, which
        # _load_context_file_from_dir prefers. The main repo's CLAUDE.md is
        # nobody's duplicate, so dropping it would lose its content entirely.
        tmp, agent_dir, _cwd, _home = dirs
        _outer, main, worktree, worktree_src = self.setup_nested_worktree(tmp)
        write(main / "CLAUDE.md", "main repo instructions")
        write(worktree / "AGENTS.md", "worktree instructions")

        files = await load_project_context_files(cwd=str(worktree_src), agent_dir=str(agent_dir))

        assert [f.content for f in files] == ["main repo instructions", "worktree instructions"]

    @pytest.mark.tonio
    async def test_does_not_skip_the_containers_context_in_a_bare_layout(self, dirs):
        # `git clone --bare proj/.bare` + `git worktree add ../main` makes commondir
        # `../..`, so dirname(commonGitDir) is `proj` - a plain directory that tracks
        # nothing. Its AGENTS.md is not a duplicate of the worktree's. Layout below
        # matches what real git writes for this setup.
        tmp, agent_dir, _cwd, _home = dirs
        proj = tmp / "proj"
        bare = proj / ".bare"
        worktree = proj / "main"
        worktree_git_dir = bare / "worktrees" / "main"
        worktree_git_dir.mkdir(parents=True, exist_ok=True)
        worktree.mkdir(parents=True, exist_ok=True)
        write(bare / "HEAD", "ref: refs/heads/main\n")
        write(worktree_git_dir / "HEAD", "ref: refs/heads/main\n")
        write(worktree_git_dir / "commondir", "../..")
        write(worktree / ".git", f"gitdir: {worktree_git_dir}\n")
        write(proj / "AGENTS.md", "container instructions")
        write(worktree / "AGENTS.md", "worktree instructions")

        files = await load_project_context_files(cwd=str(worktree), agent_dir=str(agent_dir))

        assert [f.content for f in files] == ["container instructions", "worktree instructions"]

    @pytest.mark.tonio
    async def test_keeps_loading_ancestors_above_the_main_repo(self, dirs):
        tmp, agent_dir, _cwd, _home = dirs
        outer, main, worktree, worktree_src = self.setup_nested_worktree(tmp)
        write(outer / "AGENTS.md", "outer instructions")
        write(main / "AGENTS.md", "main repo instructions")
        write(worktree / "AGENTS.md", "worktree instructions")

        files = await load_project_context_files(cwd=str(worktree_src), agent_dir=str(agent_dir))

        # Only the main repo root's duplicate is dropped; the unrelated dir above it stays.
        assert [f.content for f in files] == ["outer instructions", "worktree instructions"]

    @pytest.mark.tonio
    async def test_does_not_skip_anything_for_a_sibling_worktree(self, dirs):
        # git worktree add ../feat puts the worktree beside the main repo, so no
        # duplicate is ever encountered and ancestors above it are unrelated.
        tmp, agent_dir, _cwd, _home = dirs
        outer = tmp / "outer"
        main = outer / "main"
        sib = outer / "sib-feat"
        sib_src = sib / "src"
        sib_src.mkdir(parents=True, exist_ok=True)
        main.mkdir(parents=True, exist_ok=True)
        write(outer / "AGENTS.md", "outer instructions")
        write(sib / "AGENTS.md", "sibling worktree instructions")
        self.link_worktree(main, sib, "sib")

        files = await load_project_context_files(cwd=str(sib_src), agent_dir=str(agent_dir))

        assert [f.content for f in files] == ["outer instructions", "sibling worktree instructions"]

    @pytest.mark.tonio
    async def test_does_not_skip_the_superprojects_context_from_inside_a_submodule(self, dirs):
        # A submodule's `.git` file is also `gitdir:`-style, but its gitdir has no
        # commondir, so it resolves under `.git/modules` - never an ancestor of cwd.
        tmp, agent_dir, _cwd, _home = dirs
        sup = tmp / "super"
        sub = sup / "vendor" / "lib"
        sub_src = sub / "src"
        sub_src.mkdir(parents=True, exist_ok=True)
        write(sup / "AGENTS.md", "superproject instructions")
        write(sub / "AGENTS.md", "submodule instructions")
        sub_git_dir = sup / ".git" / "modules" / "vendor" / "lib"
        sub_git_dir.mkdir(parents=True, exist_ok=True)
        write(sub_git_dir / "HEAD", "ref: refs/heads/main\n")
        write(sub / ".git", f"gitdir: {sub_git_dir}\n")

        files = await load_project_context_files(cwd=str(sub_src), agent_dir=str(agent_dir))

        assert [f.content for f in files] == ["superproject instructions", "submodule instructions"]

    @pytest.mark.tonio
    async def test_keeps_climbing_past_an_ordinary_repo_root(self, dirs):
        tmp, agent_dir, _cwd, _home = dirs
        outer = tmp / "outer"
        repo = outer / "repo"
        leaf = repo / "src"
        leaf.mkdir(parents=True, exist_ok=True)
        (repo / ".git").mkdir(parents=True, exist_ok=True)
        write(repo / ".git" / "HEAD", "ref: refs/heads/main\n")
        write(outer / "AGENTS.md", "outer instructions")
        write(repo / "AGENTS.md", "repo instructions")
        write(leaf / "AGENTS.md", "leaf instructions")

        files = await load_project_context_files(cwd=str(leaf), agent_dir=str(agent_dir))

        assert [f.content for f in files] == ["outer instructions", "repo instructions", "leaf instructions"]

    @pytest.mark.tonio
    async def test_climbs_normally_when_the_gitdir_target_does_not_exist(self, dirs):
        tmp, agent_dir, _cwd, _home = dirs
        repo = tmp / "corrupt"
        src = repo / "src"
        src.mkdir(parents=True, exist_ok=True)
        write(repo / ".git", "gitdir: /nonexistent/path/worktrees/feat\n")
        write(repo / "AGENTS.md", "repo instructions")
        write(src / "AGENTS.md", "src instructions")

        files = await load_project_context_files(cwd=str(src), agent_dir=str(agent_dir))

        assert [f.content for f in files] == ["repo instructions", "src instructions"]
