import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_dir(request):
    """Plain-return temp dir for @pytest.mark.tonio tests.

    pytest's `tmp_path` is a yield fixture, which the tonio plugin cannot wrap
    (Rust abort); this fixture cleans up via addfinalizer instead.
    """
    path = Path(tempfile.mkdtemp(prefix="pidrei-tui-test-"))
    request.addfinalizer(lambda: shutil.rmtree(path, ignore_errors=True))
    return path
