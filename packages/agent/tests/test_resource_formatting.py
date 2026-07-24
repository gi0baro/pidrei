"""Mirror of pi agent/test/harness/resource-formatting.test.ts."""

import pytest

from pidrei_agent.harness.prompt_templates import PromptTemplate, format_prompt_template_invocation
from pidrei_agent.harness.skills import format_skill_invocation
from pidrei_agent.harness.types import Skill


@pytest.mark.tonio
async def test_formats_skill_invocations_with_additional_instructions():
    skill = Skill(
        name="inspect",
        description="Inspect things",
        content="Use inspection tools.",
        file_path="/project/.pi/skills/inspect/SKILL.md",
    )

    assert format_skill_invocation(skill, "Check errors.") == (
        '<skill name="inspect" location="/project/.pi/skills/inspect/SKILL.md">\n'
        "References are relative to /project/.pi/skills/inspect.\n\n"
        "Use inspection tools.\n</skill>\n\nCheck errors."
    )


@pytest.mark.tonio
async def test_formats_prompt_template_invocations_with_positional_arguments():
    assert (
        format_prompt_template_invocation(
            PromptTemplate(name="review", content="Review $1 with $ARGUMENTS"), ["a.ts", "care"]
        )
        == "Review a.ts with a.ts care"
    )
