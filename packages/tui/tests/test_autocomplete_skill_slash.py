"""Mirror of pi tui test/autocomplete-skill-slash.test.ts."""

import os

import pytest

from pidrei_tui.autocomplete import CombinedAutocompleteProvider
from pidrei_tui.components.cancellable_loader import CancelToken


COMMANDS = [
    {"name": "skill:deep-research", "description": "Multi-agent deep research"},
    {"name": "skill:research-idea", "description": "Refine a raw idea into a falsifiable seed"},
    {"name": "skill:to-sidecar", "description": "Route work to a sidecar"},
    {"name": "skill:brainstorm", "description": "Generate ideas"},
    {"name": "model", "description": "Select the active model"},
]


async def suggestions_for(prefix: str) -> list[str]:
    provider = CombinedAutocompleteProvider(COMMANDS, os.getcwd())
    line = f"/{prefix}"
    result = await provider.get_suggestions([line], 0, len(line), {"signal": CancelToken()})
    assert result, f'expected suggestions for "/{prefix}"'
    return [item["value"] for item in result["items"]]


@pytest.mark.tonio
async def test_ranks_skill_research_idea_first_for_query_idea():
    items = await suggestions_for("idea")
    assert items[0] == "skill:research-idea"
    assert items.index("skill:deep-research") > items.index("skill:research-idea")


@pytest.mark.tonio
async def test_keeps_ordinary_slash_commands_matching():
    items = await suggestions_for("mod")
    assert "model" in items


@pytest.mark.tonio
async def test_completes_commands_after_leading_whitespace_and_preserves_it():
    provider = CombinedAutocompleteProvider([{"name": "model"}], os.getcwd())
    for line, expected in [(" /", " /model "), ("  /mod", "  /model "), ("\t/mod", "\t/model ")]:
        result = await provider.get_suggestions([line], 0, len(line), {"signal": CancelToken()})
        assert result
        assert result["prefix"] == line.lstrip()
        assert [item["value"] for item in result["items"]] == ["model"]
        applied = provider.apply_completion([line], 0, len(line), result["items"][0], result["prefix"])
        assert applied["lines"][0] == expected
        assert applied["cursorCol"] == len(expected)


@pytest.mark.tonio
async def test_completes_command_arguments_after_leading_whitespace():
    prefixes: list[str] = []

    async def get_argument_completions(prefix: str) -> list[dict]:
        prefixes.append(prefix)
        return [{"value": "sonnet", "label": "sonnet"}]

    provider = CombinedAutocompleteProvider(
        [{"name": "model", "getArgumentCompletions": get_argument_completions}], os.getcwd()
    )
    line = "  /model son"
    result = await provider.get_suggestions([line], 0, len(line), {"signal": CancelToken()})
    assert result
    assert prefixes == ["son"]
    assert result["prefix"] == "son"
    applied = provider.apply_completion([line], 0, len(line), result["items"][0], result["prefix"])
    assert applied["lines"][0] == "  /model sonnet"


@pytest.mark.tonio
async def test_keeps_explicit_skill_queries_working():
    items = await suggestions_for("skill:side")
    assert "skill:to-sidecar" in items


# Regression test for #9944.
@pytest.mark.tonio
async def test_lists_skills_while_typing_the_skill_prefix():
    items = await suggestions_for("skill")
    assert [item for item in items if item.startswith("skill:")] == [
        command["name"] for command in COMMANDS if command["name"].startswith("skill:")
    ]


@pytest.mark.tonio
async def test_keeps_fuzzy_skill_prefix_shorthand_working():
    items = await suggestions_for("skbra")
    assert "skill:brainstorm" in items
