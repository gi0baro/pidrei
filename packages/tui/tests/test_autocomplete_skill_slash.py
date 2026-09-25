"""Mirror of pi tui test/autocomplete-skill-slash.test.ts."""

import os

import pytest

from pidrei_tui.autocomplete import CombinedAutocompleteProvider
from pidrei_tui.components.cancellable_loader import CancelToken


COMMANDS = [
    {"name": "skill:deep-research", "description": "Multi-agent deep research"},
    {"name": "skill:research-idea", "description": "Refine a raw idea into a falsifiable seed"},
    {"name": "skill:to-sidecar", "description": "Route work to a sidecar"},
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
    assert "skill:deep-research" not in items


@pytest.mark.tonio
async def test_keeps_ordinary_slash_commands_matching():
    items = await suggestions_for("mod")
    assert "model" in items


@pytest.mark.tonio
async def test_keeps_explicit_skill_queries_working():
    items = await suggestions_for("skill:side")
    assert "skill:to-sidecar" in items
