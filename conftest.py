"""Repo-root pytest config.

Wires the opt-in blocking-fs detector (`scripts/blocking_fs_detector.py`) and
the always-on network guard. Per-package fixtures live in
`packages/*/tests/conftest.py`.

Network guard: tests are hermetic, so a test that reaches past loopback is a
bug, and one that reaches a live server can hang until CI kills the job (a
missing client stub sent a real Codex request that a server accepted and
never answered). An audit hook sees every DNS lookup, connect and sendto,
TonIO's sockets included (they are stdlib sockets underneath), and refuses
non-loopback targets with `OSError(ENETUNREACH)`, so the code under test fails
fast the way it would offline. The autouse `_network_guard` fixture then fails
the test by name, even if the code under test swallowed the error. Loopback
and unix sockets stay allowed: the OAuth callback server and the websocket
loopback tests use them.

Clock and timer guard: every package reads time and arms timers through
`pidrei_utils`, whose seams tests swap (`fake_timers()`, `virtual_clock()`,
`manual_ui_timers()`). The whole suite shares one runtime and one copy of
those modules, so a test that swaps one and never restores it would hand
every later test a frozen clock or a timer queue nothing advances. The
autouse `_clock_seam_guard` restores them after every test and warns, naming
the polluting test.
"""

import errno
import ipaddress
import pathlib
import sys
import threading
import warnings

import pytest

from pidrei_utils import clock, timers


sys.path.insert(0, str(pathlib.Path(__file__).parent / "scripts"))

# The seams tests may swap, captured at collection time before any test can
# touch them.
_CLOCK_SEAMS = (
    (clock, "now_ms", clock.now_ms),
    (clock, "monotonic", clock.monotonic),
    (clock, "sleep_ms", clock.sleep_ms),
    (timers, "_start", timers._start),
)


@pytest.fixture(autouse=True)
def _clock_seam_guard():
    yield
    for module, name, original in _CLOCK_SEAMS:
        if getattr(module, name) is not original:
            setattr(module, name, original)
            warnings.warn(f"test left {module.__name__}.{name} swapped; restored", stacklevel=1)


_LOOPBACK_NAMES = frozenset({"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"})
_network_attempts_lock = threading.Lock()
_network_attempts: list[str] = []


def _is_local_host(host) -> bool:
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if not isinstance(host, str):
        return False
    host = host.split("%", 1)[0]  # scoped IPv6 (fe80::1%lo)
    if host == "" or host.lower() in _LOOPBACK_NAMES:
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    if address.is_loopback or address.is_unspecified:
        return True
    mapped = getattr(address, "ipv4_mapped", None)
    return mapped is not None and mapped.is_loopback


def _network_audit_hook(event: str, args: tuple) -> None:
    if event == "socket.getaddrinfo":
        host, port = args[0], args[1]
    elif event in ("socket.connect", "socket.sendto"):
        address = args[1]
        if not isinstance(address, tuple):  # unix socket path
            return
        host, port = address[0], address[1]
    else:
        return
    if _is_local_host(host):
        return
    target = f"{event} {host}:{port}"
    with _network_attempts_lock:
        _network_attempts.append(target)
    raise OSError(errno.ENETUNREACH, f"network access is blocked in tests ({target})")


@pytest.fixture(autouse=True)
def _network_guard(request):
    yield
    with _network_attempts_lock:
        attempts = list(_network_attempts)
        _network_attempts.clear()
    if attempts:
        pytest.fail(f"{request.node.nodeid} tried to use the network: {', '.join(attempts)}", pytrace=False)


def pytest_configure(config):
    import os

    # Audit hooks cannot be removed; the test process installs it once.
    sys.addaudithook(_network_audit_hook)

    if os.environ.get("PIDREI_FS_DETECT") != "1":
        return
    import blocking_fs_detector

    blocking_fs_detector.install()
    config._blocking_fs_detector = blocking_fs_detector


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    detector = getattr(config, "_blocking_fs_detector", None)
    if detector is None:
        return
    terminalreporter.write_sep("=", "blocking filesystem calls")
    terminalreporter.write_line(detector.format_report())
