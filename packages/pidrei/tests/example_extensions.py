"""Import an example extension the way the loader would.

pi's example suites `import extension from "../examples/extensions/x.ts"`.
`packages/pidrei/examples` is not an installed package, so the mirrors go
through the loader's own module machinery instead — which also means the
examples are exercised exactly as a user's `-e ./examples/...` would load
them, not as ordinary imports.
"""

import os

from pidrei.core.extensions.loader import _import_module


EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "extensions")


def load_example(name: str):
    """`name` is either a module (`foo.py`) or a package directory, the two
    forms discovery accepts."""
    package_init = os.path.join(EXAMPLES_DIR, name, "__init__.py")
    if os.path.exists(package_init):
        return _import_module(package_init)
    return _import_module(os.path.join(EXAMPLES_DIR, f"{name}.py"))
