import os


def pytest_configure(config):
    # pi's model-registry tests used to really fetch pi.dev catalogs; since
    # 0.83.0 pi also runs its suite with PI_OFFLINE=1 (opt-out per test via
    # allowNetwork()). pidrei tests stay hermetic the same way: the PI_OFFLINE
    # equivalent disables the remote-catalog network path for every runtime
    # built in this suite; tests that exercise network code paths against
    # local mocks pop the variable themselves.
    os.environ["PIDREI_OFFLINE"] = "1"
