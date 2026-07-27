"""Mirror of pi coding-agent test/skills.test.ts.

pi's static fixture tree is generated on the fly here. The invalid-YAML test
asserts js-yaml's "at line" message in pi; PyYAML's message differs, so only
the line-bearing warning is asserted.
"""

import os

import pytest

from pidrei.core.skills import Skill, format_skills_for_prompt, load_skills, load_skills_from_dir
from pidrei.core.source_info import create_synthetic_source_info


def write_skill_file(path, frontmatter_lines, body="Content"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    content = "---\n" + "\n".join(frontmatter_lines) + "\n---\n" + body
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


@pytest.fixture
def fixtures_dir(tmp_dir):
    root = tmp_dir / "skills"
    write_skill_file(str(root / "valid-skill" / "SKILL.md"), ["description: A valid skill for testing purposes."])
    write_skill_file(str(root / "name-mismatch" / "SKILL.md"), ["name: different-name", "description: Different name."])
    write_skill_file(
        str(root / "invalid-name-chars" / "SKILL.md"), ["name: Invalid_Name!", "description: Bad name chars."]
    )
    write_skill_file(str(root / "long-name" / "SKILL.md"), [f"name: {'a' * 70}", "description: Long name."])
    write_skill_file(str(root / "missing-description" / "SKILL.md"), ["name: missing-description"])
    write_skill_file(
        str(root / "unknown-field" / "SKILL.md"),
        ["description: Has unknown field.", "unknown-field: whatever"],
    )
    write_skill_file(
        str(root / "nested" / "sub" / "child-skill" / "SKILL.md"),
        ["name: child-skill", "description: Nested child."],
    )
    write_skill_file(
        str(root / "root-skill-preferred" / "SKILL.md"),
        ["name: root-skill-preferred", "description: Root skill should win."],
    )
    write_skill_file(
        str(root / "root-skill-preferred" / "nested" / "SKILL.md"),
        ["name: nested-loser", "description: Should not load."],
    )
    (root / "no-frontmatter").mkdir(parents=True)
    (root / "no-frontmatter" / "SKILL.md").write_text("Just body, no frontmatter.", encoding="utf-8")
    (root / "invalid-yaml").mkdir(parents=True)
    (root / "invalid-yaml" / "SKILL.md").write_text("---\nfoo: [bar\n---\nBody", encoding="utf-8")
    write_skill_file(
        str(root / "multiline-description" / "SKILL.md"),
        ["description: |", "  This is a multiline description.", "  It has two lines."],
    )
    write_skill_file(
        str(root / "consecutive-hyphens" / "SKILL.md"),
        ["name: bad--name", "description: Consecutive hyphens."],
    )
    write_skill_file(
        str(root / "disable-model-invocation" / "SKILL.md"),
        ["description: Hidden from prompt.", "disable-model-invocation: true"],
    )
    return root


def create_test_skill(
    *, name, description, file_path, base_dir, disable_model_invocation=False, source="test"
) -> Skill:
    return Skill(
        name=name,
        description=description,
        file_path=file_path,
        base_dir=base_dir,
        source_info=create_synthetic_source_info(file_path, source=source),
        disable_model_invocation=disable_model_invocation,
    )


class TestLoadSkillsFromDir:
    @pytest.mark.tonio
    async def test_loads_a_valid_skill(self, fixtures_dir):
        result = await load_skills_from_dir(dir=str(fixtures_dir / "valid-skill"), source="test")
        assert len(result.skills) == 1
        assert result.skills[0].name == "valid-skill"
        assert result.skills[0].description == "A valid skill for testing purposes."
        assert result.skills[0].source_info.source == "test"
        assert result.diagnostics == []

    @pytest.mark.tonio
    async def test_allows_names_that_dont_match_parent_directory(self, fixtures_dir):
        result = await load_skills_from_dir(dir=str(fixtures_dir / "name-mismatch"), source="test")
        assert len(result.skills) == 1
        assert result.skills[0].name == "different-name"
        assert not any("does not match parent directory" in d.message for d in result.diagnostics)

    @pytest.mark.tonio
    async def test_warns_when_name_contains_invalid_characters(self, fixtures_dir):
        result = await load_skills_from_dir(dir=str(fixtures_dir / "invalid-name-chars"), source="test")
        assert len(result.skills) == 1
        assert any("invalid characters" in d.message for d in result.diagnostics)

    @pytest.mark.tonio
    async def test_warns_when_name_exceeds_64_characters(self, fixtures_dir):
        result = await load_skills_from_dir(dir=str(fixtures_dir / "long-name"), source="test")
        assert len(result.skills) == 1
        assert any("exceeds 64 characters" in d.message for d in result.diagnostics)

    @pytest.mark.tonio
    async def test_warns_and_skips_skill_when_description_is_missing(self, fixtures_dir):
        result = await load_skills_from_dir(dir=str(fixtures_dir / "missing-description"), source="test")
        assert result.skills == []
        assert any("description is required" in d.message for d in result.diagnostics)

    @pytest.mark.tonio
    async def test_ignores_unknown_frontmatter_fields(self, fixtures_dir):
        result = await load_skills_from_dir(dir=str(fixtures_dir / "unknown-field"), source="test")
        assert len(result.skills) == 1
        assert result.diagnostics == []

    @pytest.mark.tonio
    async def test_loads_nested_skills_recursively(self, fixtures_dir):
        result = await load_skills_from_dir(dir=str(fixtures_dir / "nested"), source="test")
        assert len(result.skills) == 1
        assert result.skills[0].name == "child-skill"
        assert result.diagnostics == []

    @pytest.mark.tonio
    async def test_prefers_a_directorys_root_skill_md_over_nested_skill_md_files(self, fixtures_dir):
        result = await load_skills_from_dir(dir=str(fixtures_dir / "root-skill-preferred"), source="test")
        assert len(result.skills) == 1
        assert result.skills[0].name == "root-skill-preferred"
        assert result.skills[0].description == "Root skill should win."
        assert result.diagnostics == []

    @pytest.mark.tonio
    async def test_skips_files_without_frontmatter(self, fixtures_dir):
        result = await load_skills_from_dir(dir=str(fixtures_dir / "no-frontmatter"), source="test")
        assert result.skills == []
        assert any("description is required" in d.message for d in result.diagnostics)

    @pytest.mark.tonio
    async def test_warns_and_skips_skill_when_yaml_frontmatter_is_invalid(self, fixtures_dir):
        result = await load_skills_from_dir(dir=str(fixtures_dir / "invalid-yaml"), source="test")
        assert result.skills == []
        assert any("line" in d.message for d in result.diagnostics)

    @pytest.mark.tonio
    async def test_preserves_multiline_descriptions_from_yaml(self, fixtures_dir):
        result = await load_skills_from_dir(dir=str(fixtures_dir / "multiline-description"), source="test")
        assert len(result.skills) == 1
        assert "\n" in result.skills[0].description
        assert "This is a multiline description." in result.skills[0].description
        assert result.diagnostics == []

    @pytest.mark.tonio
    async def test_warns_when_name_contains_consecutive_hyphens(self, fixtures_dir):
        result = await load_skills_from_dir(dir=str(fixtures_dir / "consecutive-hyphens"), source="test")
        assert len(result.skills) == 1
        assert any("consecutive hyphens" in d.message for d in result.diagnostics)

    @pytest.mark.tonio
    async def test_loads_all_skills_from_fixture_directory(self, fixtures_dir):
        result = await load_skills_from_dir(dir=str(fixtures_dir), source="test")
        assert len(result.skills) >= 6

    @pytest.mark.tonio
    async def test_returns_empty_for_non_existent_directory(self):
        result = await load_skills_from_dir(dir="/non/existent/path", source="test")
        assert result.skills == []
        assert result.diagnostics == []

    @pytest.mark.tonio
    async def test_uses_parent_directory_name_when_name_not_in_frontmatter(self, fixtures_dir):
        result = await load_skills_from_dir(dir=str(fixtures_dir / "valid-skill"), source="test")
        assert len(result.skills) == 1
        assert result.skills[0].name == "valid-skill"

    @pytest.mark.tonio
    async def test_parses_disable_model_invocation_frontmatter_field(self, fixtures_dir):
        result = await load_skills_from_dir(dir=str(fixtures_dir / "disable-model-invocation"), source="test")
        assert len(result.skills) == 1
        assert result.skills[0].name == "disable-model-invocation"
        assert result.skills[0].disable_model_invocation is True
        assert not any("unknown frontmatter field" in d.message for d in result.diagnostics)

    @pytest.mark.tonio
    async def test_defaults_disable_model_invocation_to_false_when_not_specified(self, fixtures_dir):
        result = await load_skills_from_dir(dir=str(fixtures_dir / "valid-skill"), source="test")
        assert len(result.skills) == 1
        assert result.skills[0].disable_model_invocation is False


class TestFormatSkillsForPrompt:
    def test_returns_empty_string_for_no_skills(self):
        assert format_skills_for_prompt([]) == ""

    def test_formats_skills_as_xml(self):
        skills = [
            create_test_skill(
                name="test-skill",
                description="A test skill.",
                file_path="/path/to/skill/SKILL.md",
                base_dir="/path/to/skill",
            )
        ]

        result = format_skills_for_prompt(skills)

        assert "<available_skills>" in result
        assert "</available_skills>" in result
        assert "<skill>" in result
        assert "<name>test-skill</name>" in result
        assert "<description>A test skill.</description>" in result
        assert "<location>/path/to/skill/SKILL.md</location>" in result

    def test_includes_intro_text_before_xml(self):
        skills = [
            create_test_skill(
                name="test-skill",
                description="A test skill.",
                file_path="/path/to/skill/SKILL.md",
                base_dir="/path/to/skill",
            )
        ]

        result = format_skills_for_prompt(skills)
        intro_text = result[: result.index("<available_skills>")]

        assert "The following skills provide specialized instructions" in intro_text
        assert "Use the read tool to load a skill's file" in intro_text

    def test_escapes_xml_special_characters(self):
        skills = [
            create_test_skill(
                name="test-skill",
                description='A skill with <special> & "characters".',
                file_path="/path/to/skill/SKILL.md",
                base_dir="/path/to/skill",
            )
        ]

        result = format_skills_for_prompt(skills)

        assert "&lt;special&gt;" in result
        assert "&amp;" in result
        assert "&quot;characters&quot;" in result

    def test_formats_multiple_skills(self):
        skills = [
            create_test_skill(
                name="skill-one", description="First skill.", file_path="/path/one/SKILL.md", base_dir="/path/one"
            ),
            create_test_skill(
                name="skill-two", description="Second skill.", file_path="/path/two/SKILL.md", base_dir="/path/two"
            ),
        ]

        result = format_skills_for_prompt(skills)

        assert "<name>skill-one</name>" in result
        assert "<name>skill-two</name>" in result
        assert result.count("<skill>") == 2

    def test_excludes_skills_with_disable_model_invocation_from_prompt(self):
        skills = [
            create_test_skill(
                name="visible-skill",
                description="A visible skill.",
                file_path="/path/visible/SKILL.md",
                base_dir="/path/visible",
            ),
            create_test_skill(
                name="hidden-skill",
                description="A hidden skill.",
                file_path="/path/hidden/SKILL.md",
                base_dir="/path/hidden",
                disable_model_invocation=True,
            ),
        ]

        result = format_skills_for_prompt(skills)

        assert "<name>visible-skill</name>" in result
        assert "<name>hidden-skill</name>" not in result
        assert result.count("<skill>") == 1

    def test_returns_empty_string_when_all_skills_have_disable_model_invocation(self):
        skills = [
            create_test_skill(
                name="hidden-skill",
                description="A hidden skill.",
                file_path="/path/hidden/SKILL.md",
                base_dir="/path/hidden",
                disable_model_invocation=True,
            )
        ]

        assert format_skills_for_prompt(skills) == ""


class TestLoadSkillsWithOptions:
    @pytest.mark.tonio
    async def test_loads_from_explicit_skill_paths(self, fixtures_dir, tmp_dir):
        empty_agent_dir = tmp_dir / "empty-agent"
        empty_cwd = tmp_dir / "empty-cwd"
        empty_agent_dir.mkdir()
        empty_cwd.mkdir()

        result = await load_skills(
            agent_dir=str(empty_agent_dir),
            cwd=str(empty_cwd),
            skill_paths=[str(fixtures_dir / "valid-skill")],
            include_defaults=True,
        )
        assert len(result.skills) == 1
        assert result.skills[0].source_info.scope == "temporary"
        assert result.diagnostics == []

    @pytest.mark.tonio
    async def test_warns_when_skill_path_does_not_exist(self, tmp_dir):
        result = await load_skills(
            agent_dir=str(tmp_dir / "empty-agent"),
            cwd=str(tmp_dir / "empty-cwd"),
            skill_paths=["/non/existent/path"],
            include_defaults=True,
        )
        assert result.skills == []
        assert any("does not exist" in d.message for d in result.diagnostics)


class TestCollisionHandling:
    @pytest.mark.tonio
    async def test_detects_name_collisions_via_load_skills(self, tmp_dir):
        first_dir = tmp_dir / "first" / "calendar"
        second_dir = tmp_dir / "second" / "calendar"
        write_skill_file(str(first_dir / "SKILL.md"), ["name: calendar", "description: First calendar."])
        write_skill_file(str(second_dir / "SKILL.md"), ["name: calendar", "description: Second calendar."])

        result = await load_skills(
            agent_dir=str(tmp_dir / "empty-agent"),
            cwd=str(tmp_dir / "empty-cwd"),
            skill_paths=[str(tmp_dir / "first"), str(tmp_dir / "second")],
            include_defaults=False,
        )

        assert len(result.skills) == 1
        assert result.skills[0].description == "First calendar."
        collisions = [d for d in result.diagnostics if d.type == "collision"]
        assert len(collisions) == 1
        assert collisions[0].collision.resource_type == "skill"
        assert collisions[0].collision.winner_path == result.skills[0].file_path
