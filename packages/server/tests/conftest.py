import os
import shutil
import tempfile
from pathlib import Path

import pytest


def pytest_configure(config):
    # Hermetic runs: no remote-catalog network and no accidental radius
    # activation from the developer's environment.
    os.environ["PIDREI_OFFLINE"] = "1"
    os.environ.pop("RADIUS_API_KEY", None)


@pytest.fixture
def tmp_dir(request):
    """Plain-return temp dir for @pytest.mark.tonio tests.

    pytest's `tmp_path` is a yield fixture, which the tonio plugin cannot wrap
    (Rust abort); this fixture cleans up via addfinalizer instead.
    """
    path = Path(tempfile.mkdtemp(prefix="pidrei-server-test-"))
    request.addfinalizer(lambda: shutil.rmtree(path, ignore_errors=True))
    return path
