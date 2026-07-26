"""Mirror of pi coding-agent src/extensions/index.ts: extensions that ship
with pidrei itself and are loaded as inline factories before any on-disk one.

pi's only entry here is its llama.cpp integration (`src/extensions/llama`,
~1,400 lines: a managed local server, GGUF downloads from HuggingFace, and its
own TUI), registered hidden. It is deliberately **not** ported: it is not
parity work but a product question — whether pidrei ships and maintains a
local-inference stack — and product questions belong to Phase 7. Nothing else
depends on it; `pi --no-extensions` already runs without it, and a user who
wants it can point a `packages` entry at an extension that does the same job.

The registry itself is real, so a bundled extension can be added without
touching `main.py`.
"""

from typing import Any


def builtin_extensions() -> list[Any]:
    """Inline extensions bundled with pidrei.

    Entries are either a bare factory or an object with `name`, `factory` and
    optional `hidden` (pi's InlineExtension); the resource loader names them
    `<inline:name>` in the startup Extensions list.
    """
    return []


__all__ = ["builtin_extensions"]
