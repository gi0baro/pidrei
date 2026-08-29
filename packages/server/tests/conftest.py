import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def sock_dir():
    """Short temp dir for AF_UNIX socket paths.

    Deliberately not pytest's `tmp_path`: its deep per-test directories
    (`$TMPDIR/pytest-of-<user>/pytest-N/<testname>N/...`) overflow `sun_path`
    (~104 bytes on macOS) on the GHA runners, where `$TMPDIR` itself is
    already ~50 bytes. A bare `mkdtemp` stays well under the limit.
    """
    # An exotic $TMPDIR can be long enough to overflow on its own; AF_UNIX
    # does not care about $TMPDIR conventions, so fall back to /tmp.
    base = None if len(tempfile.gettempdir()) <= 60 else "/tmp"
    path = Path(tempfile.mkdtemp(prefix="pidrei-sock-", dir=base))
    yield path
    shutil.rmtree(path, ignore_errors=True)
