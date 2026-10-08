"""Mirror of pi coding-agent test/program-status-reporter.test.ts (pi #10607)."""

import dataclasses
from types import SimpleNamespace

from pidrei.core.agent_session import (
    AgentSettledEvent,
    CompactionEndEvent,
    CompactionStartEvent,
    SessionInfoChangedEvent,
)
from pidrei.modes.interactive.program_status_reporter import BlockedStatus, ProgramStatusReporter
from pidrei_tui import ProgramStatus

from .agent_session_helpers import create_assistant_message


def setup(session_name: str | None = None):
    reports: list[ProgramStatus] = []
    terminal = SimpleNamespace(set_program_status=reports.append)
    session = SimpleNamespace(name=session_name)
    reporter = ProgramStatusReporter(lambda: terminal, lambda: session.name)

    def send(*events) -> None:
        for event in events:
            reporter.handle_event(event)

    def last() -> dict:
        status = dataclasses.asdict(reports[-1])
        del status["app"]
        return {key: value for key, value in status.items() if value is not None}

    return reporter, reports, session, send, last


def assistant_end(stop_reason: str, error_message: str | None = None):
    message = create_assistant_message("secret assistant output", stop_reason=stop_reason, error_message=error_message)
    return SimpleNamespace(type="message_end", message=message)


AGENT_START = SimpleNamespace(type="agent_start")
SETTLED = AgentSettledEvent(aborted=False)


def compaction_end(reason: str, *, aborted: bool = False, error_message: str | None = None) -> CompactionEndEvent:
    return CompactionEndEvent(
        reason=reason, result=None, aborted=aborted, will_retry=False, error_message=error_message
    )


def test_reports_idle_working_during_a_run_and_done_once_it_settles():
    reporter, reports, _session, send, last = setup("Fix login")
    reporter.report()
    assert reports[-1] == ProgramStatus(state="idle", app="pidrei")

    send(AGENT_START)
    assert last() == {"state": "working", "message": "Fix login"}

    send(assistant_end("toolUse"), assistant_end("stop"))
    assert last() == {"state": "working", "message": "Fix login"}

    send(SETTLED)
    assert last() == {"state": "done", "message": "Fix login"}
    assert "secret assistant output" not in repr(reports)


def test_reports_only_the_outcome_of_the_run_retried_errors_final_errors_and_aborts():
    _reporter, _reports, _session, send, last = setup()
    send(AGENT_START, assistant_end("error", "overloaded"), assistant_end("stop"), SETTLED)
    assert last() == {"state": "done"}

    send(AGENT_START, assistant_end("error", "Invalid API key\n{details}"), SETTLED)
    assert last() == {"state": "error", "message": "Invalid API key"}

    send(AGENT_START, assistant_end("aborted"), AgentSettledEvent(aborted=True))
    assert last() == {"state": "idle"}

    # Aborted after a successful response, for example in an agent_before_settle hook or a retry delay.
    send(AGENT_START, assistant_end("stop"), AgentSettledEvent(aborted=True))
    assert last() == {"state": "idle"}


def test_reports_a_failed_recovery_compaction_as_the_runs_error_unless_a_later_response_succeeds():
    _reporter, _reports, _session, send, last = setup()
    send(
        AGENT_START,
        assistant_end("length"),
        CompactionStartEvent(reason="overflow"),
        compaction_end("overflow", error_message="Compaction failed\nstack"),
        SETTLED,
    )
    assert last() == {"state": "error", "message": "Compaction failed"}

    send(
        AGENT_START,
        CompactionStartEvent(reason="threshold"),
        compaction_end("threshold", error_message="Compaction failed"),
        assistant_end("stop"),
        SETTLED,
    )
    assert last() == {"state": "done"}


def test_reports_compaction_inside_a_run_and_the_result_of_a_manual_compaction():
    _reporter, _reports, _session, send, last = setup("Session")
    send(AGENT_START, CompactionStartEvent(reason="threshold"))
    assert last() == {"state": "working", "message": "Compacting context"}
    send(compaction_end("threshold"))
    assert last() == {"state": "working", "message": "Session"}
    send(assistant_end("stop"), SETTLED)

    send(CompactionStartEvent(reason="manual"), compaction_end("manual"))
    assert last() == {"state": "done", "message": "Session"}
    send(CompactionStartEvent(reason="manual"), compaction_end("manual", error_message="No model"))
    assert last() == {"state": "error", "message": "No model"}
    send(CompactionStartEvent(reason="manual"), compaction_end("manual", aborted=True))
    assert last() == {"state": "idle"}


def test_reports_the_most_recent_open_dialog_and_the_underlying_state_once_all_close():
    reporter, _reports, _session, send, last = setup()
    send(AGENT_START)
    reporter.set_blocked("extension-selector", BlockedStatus("permission", "Allow bash?"))
    reporter.set_blocked("login", BlockedStatus("auth", "Log in to Anthropic"))
    assert last() == {"state": "blocked", "kind": "auth", "message": "Log in to Anthropic"}

    # The run settles while the selector is still open.
    reporter.set_blocked("login", None)
    send(assistant_end("stop"), SETTLED)
    assert last() == {"state": "blocked", "kind": "permission", "message": "Allow bash?"}

    # Reopening a source replaces its dialog instead of stacking a second one.
    reporter.set_blocked("extension-selector", BlockedStatus("question", "Pick one"))
    assert last() == {"state": "blocked", "kind": "question", "message": "Pick one"}
    reporter.set_blocked("extension-selector", None)
    assert last() == {"state": "done"}


def test_sends_each_status_once_and_follows_session_name_changes():
    reporter, reports, session, send, last = setup("Old")
    send(AGENT_START, SimpleNamespace(type="turn_start"), assistant_end("toolUse"))
    reporter.report()
    assert len(reports) == 1

    session.name = "New"
    send(SessionInfoChangedEvent(name="New"))
    assert last() == {"state": "working", "message": "New"}


def test_returns_to_idle_when_the_session_is_replaced():
    reporter, _reports, _session, send, last = setup()
    send(AGENT_START, assistant_end("stop"), SETTLED)
    reporter.reset()
    assert last() == {"state": "idle"}
