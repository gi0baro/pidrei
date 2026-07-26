"""Mirror of pi agent/test/harness/skills.test.ts."""

import os

import pytest

from pidrei_agent.harness.env.local import LocalExecutionEnv
from pidrei_agent.harness.skills import SourcedSkill, SourcedSkillDiagnostic, load_skills, load_sourced_skills
from pidrei_agent.harness.types import Skill, get_or_throw
from tests.session_helpers import create_temp_dir


@pytest.mark.tonio
async def test_loads_skill_md_files_through_the_execution_environment():
    root = create_temp_dir()
    env = LocalExecutionEnv(cwd=root)
    get_or_throw(await env.create_dir(".agents/skills/example", recursive=True))
    get_or_throw(
        await env.write_file(
            ".agents/skills/example/SKILL.md",
            "---\nname: example\ndescription: Example skill\ndisable-model-invocation: true\n---\nUse this skill.\n",
        )
    )

    result = await load_skills(env, ".agents/skills")

    assert result.diagnostics == []
    assert result.skills == [
        Skill(
            name="example",
            description="Example skill",
            content="Use this skill.",
            file_path=os.path.join(root, ".agents/skills/example/SKILL.md"),
            disable_model_invocation=True,
        )
    ]


@pytest.mark.tonio
async def test_loads_skills_through_symlinked_directories():
    root = create_temp_dir()
    env = LocalExecutionEnv(cwd=root)
    get_or_throw(await env.create_dir("actual/example", recursive=True))
    get_or_throw(
        await env.write_file(
            "actual/example/SKILL.md", "---\nname: example\ndescription: Example skill\n---\nUse this skill."
        )
    )
    os.symlink(os.path.join(root, "actual"), os.path.join(root, "skills-link"))

    result = await load_skills(env, "skills-link")

    assert [skill.name for skill in result.skills] == ["example"]
    assert result.skills[0].file_path == os.path.join(root, "skills-link/example/SKILL.md")


@pytest.mark.tonio
async def test_preserves_source_info_for_sourced_skills():
    root = create_temp_dir()
    env = LocalExecutionEnv(cwd=root)
    get_or_throw(await env.create_dir("user/example", recursive=True))
    get_or_throw(
        await env.write_file(
            "user/example/SKILL.md", "---\nname: example\ndescription: Example skill\n---\nUse this skill."
        )
    )

    result = await load_sourced_skills(env, [{"path": "user", "source": {"type": "user"}}])

    assert result.diagnostics == []
    assert result.skills == [
        SourcedSkill(
            skill=Skill(
                name="example",
                description="Example skill",
                content="Use this skill.",
                file_path=os.path.join(root, "user/example/SKILL.md"),
                disable_model_invocation=False,
            ),
            source={"type": "user"},
        )
    ]


@pytest.mark.tonio
async def test_attaches_source_info_to_diagnostics():
    root = create_temp_dir()
    env = LocalExecutionEnv(cwd=root)
    get_or_throw(await env.create_dir("user/broken", recursive=True))
    get_or_throw(await env.write_file("user/broken/SKILL.md", "---\nname: broken\n---\nMissing description."))

    result = await load_sourced_skills(env, [{"path": "user", "source": {"type": "user"}}])

    assert result.skills == []
    assert result.diagnostics == [
        SourcedSkillDiagnostic(
            code="invalid_metadata",
            message="description is required",
            path=os.path.join(root, "user/broken/SKILL.md"),
            source={"type": "user"},
        )
    ]


@pytest.mark.tonio
async def test_loads_direct_markdown_children_only_from_the_root_directory():
    root = create_temp_dir()
    env = LocalExecutionEnv(cwd=root)
    get_or_throw(await env.create_dir("skills/nested", recursive=True))
    get_or_throw(await env.write_file("skills/root.md", "---\ndescription: Root skill\n---\nRoot content"))
    get_or_throw(await env.write_file("skills/nested/ignored.md", "---\ndescription: Ignored\n---\nIgnored content"))

    result = await load_skills(env, "skills")

    assert [skill.name for skill in result.skills] == ["skills"]
    assert result.skills[0].content == "Root content"
