import os
import shutil
import tempfile
from pathlib import Path

import pytest


def pytest_configure(config):
    # pi's model-registry tests used to really fetch pi.dev catalogs; since
    # 0.83.0 pi also runs its suite with PI_OFFLINE=1 (opt-out per test via
    # allowNetwork()). pidrei tests stay hermetic the same way: the PI_OFFLINE
    # equivalent disables the remote-catalog network path for every runtime
    # built in this suite; tests that exercise network code paths against
    # local mocks pop the variable themselves.
    os.environ["PIDREI_OFFLINE"] = "1"


@pytest.fixture
def tmp_dir(request):
    """Plain-return temp dir for @pytest.mark.tonio tests.

    pytest's `tmp_path` is a yield fixture, which the tonio plugin cannot wrap
    (Rust abort); this fixture cleans up via addfinalizer instead.
    """
    path = Path(tempfile.mkdtemp(prefix="pidrei-test-"))
    request.addfinalizer(lambda: shutil.rmtree(path, ignore_errors=True))
    return path
