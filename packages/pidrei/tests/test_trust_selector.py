"""Mirror of pi coding-agent test/trust-selector.test.ts."""

import pytest

from pidrei.core.keybindings import KeybindingsManager
from pidrei.core.trust_manager import ProjectTrustStoreEntry, ProjectTrustUpdate
from pidrei.modes.interactive.components import TrustSelectorComponent
from pidrei.modes.interactive.theme import init_theme_sync
from pidrei.utils.ansi import strip_ansi
from pidrei_tui import set_keybindings


async def _ignore_selection(selection: dict) -> None:
    pass


@pytest.fixture(autouse=True)
def _setup():
    init_theme_sync("dark")
    set_keybindings(KeybindingsManager())


class TestTrustSelectorComponent:
    def test_marks_the_saved_trusted_decision(self):
        selector = TrustSelectorComponent(
            {
                "cwd": "/project",
                "savedDecision": ProjectTrustStoreEntry(path="/project", decision=True),
                "projectTrusted": True,
                "onSelect": _ignore_selection,
                "onCancel": lambda: None,
            }
        )

        output = strip_ansi("\n".join(selector.render(120)))

        assert "Saved decision: trusted (/project)" in output
        assert "Current session: trusted" in output
        assert "Trust ✓" in output
        assert "Do not trust ✓" not in output

    @pytest.mark.tonio
    async def test_selects_a_trust_decision(self):
        selections = []

        async def on_select(selection: dict) -> None:
            selections.append(selection)

        selector = TrustSelectorComponent(
            {
                "cwd": "/project",
                "savedDecision": None,
                "projectTrusted": False,
                "onSelect": on_select,
                "onCancel": lambda: None,
            }
        )

        await selector.handle_input("\n")

        assert selections == [{"trusted": True, "updates": [ProjectTrustUpdate(path="/project", decision=True)]}]

    def test_labels_saved_ancestor_decisions_as_inherited(self):
        selector = TrustSelectorComponent(
            {
                "cwd": "/parent/project/nested",
                "savedDecision": ProjectTrustStoreEntry(path="/parent", decision=True),
                "projectTrusted": True,
                "onSelect": _ignore_selection,
                "onCancel": lambda: None,
            }
        )

        output = strip_ansi("\n".join(selector.render(120)))

        assert "Saved decision: trusted (inherited from /parent)" in output

    @pytest.mark.tonio
    async def test_adds_a_trust_parent_option(self):
        selections = []

        async def on_select(selection: dict) -> None:
            selections.append(selection)

        selector = TrustSelectorComponent(
            {
                "cwd": "/parent/project",
                "savedDecision": ProjectTrustStoreEntry(path="/parent", decision=True),
                "projectTrusted": True,
                "onSelect": on_select,
                "onCancel": lambda: None,
            }
        )

        output = strip_ansi("\n".join(selector.render(120)))
        assert "Saved decision: trusted (inherited from /parent)" in output
        assert "Trust parent folder (/parent) ✓" in output

        await selector.handle_input("\n")

        assert selections == [
            {
                "trusted": True,
                "updates": [
                    ProjectTrustUpdate(path="/parent", decision=True),
                    ProjectTrustUpdate(path="/parent/project", decision=None),
                ],
            }
        ]
