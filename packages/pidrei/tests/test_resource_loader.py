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

from pidrei.core.resource_loader import DefaultResourceLoader, SourcedPath
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
def dirs(tmp_dir):
    agent_dir = tmp_dir / "agent"
    cwd = tmp_dir / "project"
    home = tmp_dir / "home"
    agent_dir.mkdir(parents=True)
    cwd.mkdir(parents=True)
    home.mkdir(parents=True)
    return tmp_dir, agent_dir, cwd, home


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
    async def test_skips_context_file_discovery_when_no_context_files_is_true(self, dirs):
        _tmp, agent_dir, cwd, home = dirs
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
        settings_manager = SettingsManager.create(str(cwd), str(agent_dir), project_trusted=False)

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

            loader.extend_resources(
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
    async def test_loads_extension_resources_returned_as_file_urls(self, dirs):
        tmp, agent_dir, cwd, home = dirs
        extra_skill_dir = tmp / "extra skills" / "file-url-skill"
        skill_path = extra_skill_dir / "SKILL.md"
        write(skill_path, skill_md("file-url-skill", "File URL skill"))

        loader = DefaultResourceLoader(cwd=str(cwd), agent_dir=str(agent_dir))
        with fake_home(home):
            await loader.reload()

            loader.extend_resources(
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
