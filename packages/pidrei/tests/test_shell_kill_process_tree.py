"""Mirror of pi coding-agent test/shell-kill-process-tree.test.ts.

Regression coverage for #11252: stop-before-kill is macOS-only and best-effort.
The signals are recorded through the module's own `os` and `sys` names, so no
real process is signalled.
"""

import signal
from types import SimpleNamespace

import pytest

from pidrei.utils import shell


class _Signals:
    def __init__(self, fail=lambda pid, sig: False):
        self.calls: list[tuple[int, signal.Signals]] = []
        self._fail = fail

    def killpg(self, pid: int, sig: signal.Signals) -> None:
        self._record(-pid, sig)

    def kill(self, pid: int, sig: signal.Signals) -> None:
        self._record(pid, sig)

    def _record(self, pid: int, sig: signal.Signals) -> None:
        self.calls.append((pid, sig))
        if self._fail(pid, sig):
            raise ProcessLookupError("failed")


@pytest.fixture
def signals(monkeypatch):
    def install(fail=lambda pid, sig: False, platform="darwin"):
        recorder = _Signals(fail)
        monkeypatch.setattr(shell, "os", SimpleNamespace(killpg=recorder.killpg, kill=recorder.kill))
        monkeypatch.setattr(shell, "sys", SimpleNamespace(platform=platform))
        return recorder

    return install


def test_stops_the_macos_process_group_before_killing_it_without_resuming_it(signals):
    recorder = signals()
    shell.kill_process_tree(1234)
    assert recorder.calls == [(-1234, signal.SIGSTOP), (-1234, signal.SIGKILL)]


def test_still_kills_the_group_when_stopping_fails(signals):
    recorder = signals(fail=lambda pid, sig: sig == signal.SIGSTOP)
    shell.kill_process_tree(1234)
    assert recorder.calls == [(-1234, signal.SIGSTOP), (-1234, signal.SIGKILL)]


def test_falls_back_to_killing_the_shell_when_the_group_kill_fails(signals):
    recorder = signals(fail=lambda pid, sig: pid < 0 and sig == signal.SIGKILL)
    shell.kill_process_tree(1234)
    assert recorder.calls == [(-1234, signal.SIGSTOP), (-1234, signal.SIGKILL), (1234, signal.SIGKILL)]


def test_tolerates_failure_of_both_group_signals_and_the_fallback(signals):
    recorder = signals(fail=lambda pid, sig: True)
    shell.kill_process_tree(1234)
    assert recorder.calls == [(-1234, signal.SIGSTOP), (-1234, signal.SIGKILL), (1234, signal.SIGKILL)]


def test_keeps_linux_group_termination_unchanged(signals):
    recorder = signals(platform="linux")
    shell.kill_process_tree(1234)
    assert recorder.calls == [(-1234, signal.SIGKILL)]
