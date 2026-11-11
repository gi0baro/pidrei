"""Mirror of pi coding-agent test/collapsible-message-components.test.ts."""

import pytest

from pidrei.core.agent_session import ParsedSkillBlock
from pidrei.core.messages import BranchSummaryMessage, CompactionSummaryMessage
from pidrei.modes.interactive.components.branch_summary_message import BranchSummaryMessageComponent
from pidrei.modes.interactive.components.compaction_summary_message import CompactionSummaryMessageComponent
from pidrei.modes.interactive.components.skill_invocation_message import SkillInvocationMessageComponent
from pidrei.modes.interactive.theme import init_theme
from pidrei.utils.ansi import strip_ansi
from pidrei_tui import TuiMouseEvent


WIDTH = 80


@pytest.fixture(autouse=True)
async def _theme():
    await init_theme("dark")


def _render_text(component) -> str:
    return strip_ansi("\n".join(component.render(WIDTH)))


async def _click_row(component, marker: str) -> None:
    lines = component.render(WIDTH)
    row = next((index for index, line in enumerate(lines) if marker in strip_ansi(line)), -1)
    assert row >= 0
    event = TuiMouseEvent(
        type="click",
        button="left",
        x=2,
        y=row,
        screen_x=2,
        screen_y=row,
        width=WIDTH,
        height=len(lines),
        click_count=1,
    )
    result = component.handle_mouse(event)
    assert result is not None and result.handled is True


@pytest.mark.tonio
async def test_toggles_a_compaction_summary_when_clicked():
    component = CompactionSummaryMessageComponent(
        CompactionSummaryMessage(summary="compaction details", tokens_before=1234, timestamp=0)
    )

    assert "compaction details" not in _render_text(component)
    await _click_row(component, "[compaction]")
    assert "compaction details" in _render_text(component)
    await _click_row(component, "[compaction]")
    assert "compaction details" not in _render_text(component)


@pytest.mark.tonio
async def test_toggles_a_branch_summary_when_clicked():
    component = BranchSummaryMessageComponent(
        BranchSummaryMessage(summary="branch details", from_id="entry-1", timestamp=0)
    )

    assert "branch details" not in _render_text(component)
    await _click_row(component, "[branch]")
    assert "branch details" in _render_text(component)
    await _click_row(component, "[branch]")
    assert "branch details" not in _render_text(component)


@pytest.mark.tonio
async def test_toggles_a_skill_invocation_when_clicked():
    component = SkillInvocationMessageComponent(
        ParsedSkillBlock(
            name="example-skill", location="/tmp/example-skill.md", content="skill details", user_message=None
        )
    )

    assert "skill details" not in _render_text(component)
    await _click_row(component, "[skill]")
    assert "skill details" in _render_text(component)
    await _click_row(component, "[skill]")
    assert "skill details" not in _render_text(component)


# Markdown transformers


@pytest.mark.parametrize(
    ("message_type", "header", "create"),
    [
        pytest.param(
            "summary",
            "Compacted from 1,234 tokens",
            lambda transformers: CompactionSummaryMessageComponent(
                CompactionSummaryMessage(summary="details", tokens_before=1234, timestamp=0),
                None,
                1,
                transformers,
                "entry-1",
            ),
            id="compaction summary",
        ),
        pytest.param(
            "summary",
            "Branch Summary",
            lambda transformers: BranchSummaryMessageComponent(
                BranchSummaryMessage(summary="details", from_id="entry-0", timestamp=0),
                None,
                1,
                transformers,
                "entry-1",
            ),
            id="branch summary",
        ),
        pytest.param(
            "custom",
            "example-skill",
            lambda transformers: SkillInvocationMessageComponent(
                ParsedSkillBlock(
                    name="example-skill", location="/tmp/example-skill.md", content="details", user_message=None
                ),
                None,
                1,
                transformers,
                "entry-1",
            ),
            id="skill",
        ),
    ],
)
def test_runs_on_expanded_content_without_the_header(message_type, header, create):
    seen = []

    def transformer(markdown, context):
        seen.append({"markdown": markdown, "context": context})
        return markdown.replace("details", "transformed")

    component = create([transformer])
    component.set_expanded(True)

    rendered = _render_text(component)
    assert header in rendered
    assert "transformed" in rendered
    assert "details" not in rendered
    # pi's toMatchObject: a subset check.
    assert seen[0]["markdown"] == "details"
    assert {"messageType": message_type, "messageId": "entry-1"}.items() <= seen[0]["context"].items()
