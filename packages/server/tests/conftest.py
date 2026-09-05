import shutil
import tempfile
from pathlib import Path

import pytest


# `sun_path` is 104 bytes on macOS (108 on Linux), terminating NUL included.
SUN_PATH_LIMIT = 103
# The longest socket path a test builds under `sock_dir`: a server-id socket
# (`<uuid4>.sock`, 41 bytes) inside a one-level subdirectory.
LONGEST_SOCKET_SUFFIX = len("/srv-id/00000000-0000-4000-8000-000000000001.sock")
_MKDTEMP_LEN = len("/pidrei-sock-") + 8  # prefix + mkdtemp's random name


@pytest.fixture
def sock_dir():
    """Short temp dir for AF_UNIX socket paths.

    Deliberately not pytest's `tmp_path`: its deep per-test directories
    (`$TMPDIR/pytest-of-<user>/pytest-N/<testname>N/...`) overflow `sun_path`
    on the GHA runners, where the macOS `$TMPDIR`
    (`/var/folders/xx/<30 chars>/T`) is already ~50 bytes. A bare `mkdtemp`
    under a short enough base stays under the limit.
    """
    # The base is chosen against the longest path the tests will append, not a
    # fixed threshold: macOS's 49-byte $TMPDIR left room for `server.sock` but
    # not for a server-id socket. AF_UNIX does not care about $TMPDIR
    # conventions, so fall back to /tmp when the budget does not fit.
    base = tempfile.gettempdir()
    if len(base) + _MKDTEMP_LEN + LONGEST_SOCKET_SUFFIX > SUN_PATH_LIMIT:
        base = "/tmp"
    path = Path(tempfile.mkdtemp(prefix="pidrei-sock-", dir=base))
    assert len(str(path)) + LONGEST_SOCKET_SUFFIX <= SUN_PATH_LIMIT, (
        f"sock_dir {path} leaves no room for a server-id socket; widen the fallback"
    )
    yield path
    shutil.rmtree(path, ignore_errors=True)
