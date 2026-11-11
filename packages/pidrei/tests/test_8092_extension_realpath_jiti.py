"""Mirror of pi's suite/regressions/8092-extension-realpath-jiti.test.ts.

Re-specified in Python terms. pi's case is a pnpm store layout: the package
directory under `node_modules` is a symlink into the store, and the
extension's declared dependency is a bare specifier that Node finds only by
walking up from the real location. Python has no bare-specifier resolution of
that kind, so the layout has no counterpart. What the fix changes here is the
same thing in Python terms: the entry is imported from its real path, so an
entry *file* reached through a symlink resolves `from . import x` next to the
real file, not next to the link. (A symlinked *directory* already worked: its
siblings are reachable through the link too.)
"""

import os

import pytest

from pidrei.core.extensions.loader import load_extensions


EXTENSION_SOURCE = """\
from . import helper


async def extension(pi):
    if helper.value != 42:
        raise RuntimeError(f"unexpected helper value: {helper.value}")
"""


@pytest.mark.tonio
async def test_loads_an_extension_whose_sibling_import_is_only_reachable_from_the_real_file(tmp_path):
    real_dir = tmp_path / "store" / "ext"
    real_dir.mkdir(parents=True)
    (real_dir / "helper.py").write_text("value = 42\n", encoding="utf-8")
    (real_dir / "index.py").write_text(EXTENSION_SOURCE, encoding="utf-8")

    linked_dir = tmp_path / "extensions"
    linked_dir.mkdir()
    entry = linked_dir / "index.py"
    entry.symlink_to(real_dir / "index.py")
    assert os.path.exists(entry)
    assert not os.path.exists(linked_dir / "helper.py")

    result = await load_extensions([str(entry)], str(tmp_path))

    assert result.errors == []
    assert len(result.extensions) == 1
