"""Import an example extension the way the loader would.

pi's example suites `import extension from "../examples/extensions/x.ts"`.
The mirrors go through the loader's own module machinery instead, so the
examples are exercised exactly as a user's `-e <path>` would load them, not as
ordinary imports.

The directory is resolved through `config.get_examples_path()` — the same call
the system prompt uses to tell the agent where the examples are — so a move
that breaks the prompt breaks this suite too.
"""

import os

from pidrei.config import get_examples_path
from pidrei.core.extensions.loader import _import_module


EXAMPLES_DIR = os.path.join(get_examples_path(), "extensions")


def load_example(name: str):
    """`name` is either a module (`foo.py`) or a package directory, the two
    forms discovery accepts."""
    package_init = os.path.join(EXAMPLES_DIR, name, "__init__.py")
    if os.path.exists(package_init):
        return _import_module(package_init)
    return _import_module(os.path.join(EXAMPLES_DIR, f"{name}.py"))
