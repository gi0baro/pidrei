"""Mirror of pi coding-agent test/startup-ui-program-status.test.ts.

pi's `vi.mock` swaps the startup TUI's `ProcessTerminal` for a fake that records program status
reports; here `startup_ui.ProcessTerminal` is swapped for a `VirtualTerminal` that records them and
sets an Event when the TUI starts it. For a hermetic run the tmux hyperlink probe and the
terminal color query are swapped for no-ops: the fake terminal answers neither.

Not mirrored: "reports first-time setup as a permission request" (pidrei has no first-time setup:
since pi 1.6.1 it only asks the analytics opt-in, and pidrei has no telemetry).
"""

import os
import sys
from pathlib import Path

import pytest
import tonio.colored as tonio

from pidrei.cli import startup_ui
from pidrei.cli.project_trust import CreateProjectTrustContextOptions, create_project_trust_context
from pidrei.config import ENV_AGENT_DIR
from pidrei.core.project_trust import ResolveProjectTrustedOptions, resolve_project_trusted
from pidrei.core.settings_manager import SettingsManager
from pidrei.core.trust_manager import ProjectTrustStore
from pidrei_tui import ProgramStatus


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tui" / "tests"))
from virtual_terminal import VirtualTerminal


class _RecordingTerminal(VirtualTerminal):
    """Records program status reports and lets the test type into the startup dialog."""

    def __init__(self) -> None:
        super().__init__(80, 24)
        self.reports: list[ProgramStatus] = []
        self.started = tonio.Event()
        self.stopped = False

    async def start(self, on_input, on_resize, on_reply=None, on_error=None) -> None:
        await super().start(on_input, on_resize, on_reply, on_error)
        self.started.set()

    async def stop(self) -> None:
        await super().stop()
        self.stopped = True

    def set_program_status(self, status: ProgramStatus) -> None:
        self.reports.append(status)


async def _no_probe(_probe) -> None:
    return None


@pytest.fixture
def terminal(tmp_path, monkeypatch):
    """The startup TUI's terminal, with the agent dir under `tmp_path`."""
    monkeypatch.setenv(ENV_AGENT_DIR, str(tmp_path / "agent"))
    recording = _RecordingTerminal()
    monkeypatch.setattr(startup_ui, "ProcessTerminal", lambda: recording)
    monkeypatch.setattr(startup_ui, "prime_capabilities", _no_probe)
    monkeypatch.setattr(startup_ui, "request_terminal_colors", lambda _ui: None)
    return recording


async def _started(terminal: _RecordingTerminal) -> _RecordingTerminal:
    """Wait until the startup TUI has started its terminal."""
    await terminal.started.wait(5)
    assert terminal.started.is_set(), "the startup TUI never started"
    return terminal


class _Answer:
    """Runs a dialog in the test's scope and keeps what it returns."""

    def __init__(self, scope, dialog) -> None:
        self.value = None
        self._done = tonio.Event()
        scope.spawn(self._run(dialog))

    async def _run(self, dialog) -> None:
        self.value = await dialog
        self._done.set()

    async def get(self):
        await self._done.wait(5)
        assert self._done.is_set(), "the dialog never returned"
        return self.value


def _blocked(kind: str, message: str) -> ProgramStatus:
    return ProgramStatus(state="blocked", app="pidrei", kind=kind, message=message)


def _project_trust_context(cwd: str):
    return create_project_trust_context(
        CreateProjectTrustContextOptions(
            cwd=cwd, mode="interactive", settings_manager=SettingsManager.in_memory(), has_ui=True
        )
    )


# #12439: startup dialogs run before interactive mode and its status reporter exist.
class TestStartupDialogProgramStatus:
    @pytest.mark.tonio
    async def test_reports_a_selector_as_blocked_on_the_first_line_of_its_title_until_it_is_answered(self, terminal):
        async with tonio.scope(cancel_on_exc=True) as scope:
            result = _Answer(
                scope,
                startup_ui.show_startup_selector(
                    SettingsManager.in_memory(),
                    "Trust project folder?\n/work/project\n\nDetails",
                    [{"label": "Trust", "value": True}],
                ),
            )
            await _started(terminal)
            assert terminal.reports == [_blocked("question", "Trust project folder?")]

            await terminal.send_input("\r")
            assert await result.get() is True
            assert terminal.stopped is True

    @pytest.mark.tonio
    async def test_reports_a_project_trust_confirm_as_a_permission_request(self, terminal):
        context = _project_trust_context("/work/project")
        async with tonio.scope(cancel_on_exc=True) as scope:
            result = _Answer(scope, context.ui.confirm("Load project extensions?", "They run with your permissions."))
            await _started(terminal)
            assert terminal.reports == [_blocked("permission", "Load project extensions?")]

            await terminal.send_input("\r")
            assert await result.get() is True

    # The spec defines permission as approval to do something; trusting a folder lets pidrei run its extensions.
    @pytest.mark.tonio
    async def test_reports_the_project_trust_prompt_as_a_permission_request(self, terminal, tmp_path):
        agent_dir = os.environ[ENV_AGENT_DIR]
        project_dir = tmp_path / "agent" / "project"
        (project_dir / ".pidrei").mkdir(parents=True)
        (project_dir / ".pidrei" / "settings.json").write_text("{}")
        async with tonio.scope(cancel_on_exc=True) as scope:
            trusted = _Answer(
                scope,
                resolve_project_trusted(
                    ResolveProjectTrustedOptions(
                        cwd=str(project_dir),
                        trust_store=ProjectTrustStore(agent_dir),
                        default_project_trust="ask",
                        project_trust_context=_project_trust_context(str(project_dir)),
                    )
                ),
            )
            await _started(terminal)
            assert terminal.reports == [_blocked("permission", "Trust project folder?")]

            await terminal.send_input("\r")
            assert await trusted.get() is True

    @pytest.mark.tonio
    async def test_reports_the_kind_an_extension_passes_to_a_project_trust_select(self, terminal):
        context = _project_trust_context("/work/project")
        async with tonio.scope(cancel_on_exc=True) as scope:
            result = _Answer(
                scope, context.ui.select("Run project hooks?", ["Run", "Skip"], {"blockedKind": "permission"})
            )
            await _started(terminal)
            assert terminal.reports == [_blocked("permission", "Run project hooks?")]

            await terminal.send_input("\r")
            assert await result.get() == "Run"

    @pytest.mark.tonio
    async def test_reports_an_input_as_blocked_on_its_title(self, terminal):
        async with tonio.scope(cancel_on_exc=True) as scope:
            result = _Answer(scope, startup_ui.show_startup_input(SettingsManager.in_memory(), "Project name"))
            await _started(terminal)
            assert terminal.reports == [_blocked("question", "Project name")]

            await terminal.send_input("demo")
            await terminal.send_input("\r")
            assert await result.get() == "demo"
