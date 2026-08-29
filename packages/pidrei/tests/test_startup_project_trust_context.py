"""Startup project-trust context wiring (pi: src/cli/project-trust.ts).

pi has no dedicated test for `createProjectTrustContext`, so this is not a
mirror. It exists because the interactive branches shipped as
`TODO(Phase 4)` stubs returning "no answer": opening pidrei in an untrusted
folder holding `.pidrei/` resources silently resolved to *untrusted* instead
of prompting, and no test entered that branch (same shape as Phase 4.5's
defect six).

The startup TUI helpers are replaced with recorders — this asserts the
delegation and the option shapes, not the selector's rendering (that is
covered by the startup-ui and trust-selector mirrors).
"""

import contextlib

import pytest

from pidrei.cli import project_trust as project_trust_cli
from pidrei.cli.project_trust import CreateProjectTrustContextOptions, create_project_trust_context


@contextlib.contextmanager
def _recording_startup_ui(selector_result=None, input_result=None):
    """Swap the startup TUI helpers for recorders (hand-rolled; predates tonio 0.9.14 yield-fixture support)."""
    calls: list[tuple] = []

    async def fake_selector(settings_manager, title, options):
        calls.append(("select", settings_manager, title, options))
        return selector_result

    async def fake_input(settings_manager, title, placeholder=None):
        calls.append(("input", settings_manager, title, placeholder))
        return input_result

    original_selector = project_trust_cli.show_startup_selector
    original_input = project_trust_cli.show_startup_input
    project_trust_cli.show_startup_selector = fake_selector
    project_trust_cli.show_startup_input = fake_input
    try:
        yield calls
    finally:
        project_trust_cli.show_startup_selector = original_selector
        project_trust_cli.show_startup_input = original_input


def _context(mode: str = "interactive", has_ui: bool = True):
    return create_project_trust_context(
        CreateProjectTrustContextOptions(
            cwd="/project",
            mode=mode,
            settings_manager="settings-manager-sentinel",
            has_ui=has_ui,
        )
    )


class TestStartupProjectTrustContext:
    def test_maps_interactive_mode_to_tui(self):
        assert _context().mode == "tui"
        assert _context(mode="print").mode == "print"
        assert _context(mode="rpc").mode == "rpc"

    @pytest.mark.tonio
    async def test_select_delegates_to_the_startup_selector(self):
        with _recording_startup_ui(selector_result="Trust") as calls:
            selected = await _context().ui.select("Trust project folder?", ["Trust", "Do not trust"])

        assert selected == "Trust"
        assert calls == [
            (
                "select",
                "settings-manager-sentinel",
                "Trust project folder?",
                [{"label": "Trust", "value": "Trust"}, {"label": "Do not trust", "value": "Do not trust"}],
            )
        ]

    @pytest.mark.tonio
    async def test_select_returns_none_when_cancelled(self):
        with _recording_startup_ui(selector_result=None):
            assert await _context().ui.select("Trust?", ["Trust"]) is None

    @pytest.mark.tonio
    async def test_confirm_joins_title_and_message_and_offers_yes_no(self):
        with _recording_startup_ui(selector_result=True) as calls:
            confirmed = await _context().ui.confirm("Title", "Message")

        assert confirmed is True
        assert calls[0][2] == "Title\nMessage"
        assert calls[0][3] == [{"label": "Yes", "value": True}, {"label": "No", "value": False}]

    @pytest.mark.tonio
    async def test_confirm_keeps_an_explicit_no(self):
        # pi's `?? false`: a selected `false` must not be confused with cancel.
        with _recording_startup_ui(selector_result=False):
            assert await _context().ui.confirm("Title", "Message") is False

    @pytest.mark.tonio
    async def test_confirm_defaults_to_false_on_cancel(self):
        with _recording_startup_ui(selector_result=None):
            assert await _context().ui.confirm("Title", "Message") is False

    @pytest.mark.tonio
    async def test_input_delegates_with_the_placeholder(self):
        with _recording_startup_ui(input_result="typed") as calls:
            value = await _context().ui.input("Title", "placeholder")

        assert value == "typed"
        assert calls == [("input", "settings-manager-sentinel", "Title", "placeholder")]

    @pytest.mark.tonio
    async def test_no_ui_resolves_to_no_answer(self):
        context = _context(has_ui=False)
        with _recording_startup_ui(selector_result="Trust", input_result="typed") as calls:
            assert await context.ui.select("Trust?", ["Trust"]) is None
            assert await context.ui.confirm("Title", "Message") is False
            assert await context.ui.input("Title") is None

        assert calls == []

    @pytest.mark.tonio
    async def test_non_interactive_modes_resolve_to_no_answer(self):
        for mode in ("print", "json", "rpc"):
            context = _context(mode=mode)
            with _recording_startup_ui(selector_result="Trust", input_result="typed") as calls:
                assert await context.ui.select("Trust?", ["Trust"]) is None
                assert await context.ui.confirm("Title", "Message") is False
                assert await context.ui.input("Title") is None

            assert calls == [], mode
