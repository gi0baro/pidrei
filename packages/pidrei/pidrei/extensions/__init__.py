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

`main.py` imports this module at top level, and through it the codemode
extension, whose import primes the codemode sandbox runtime at program start.
"""

from typing import Any

from ..core.extensions.types import InlineExtension
from .codemode import extension as codemode_extension
from .mcp import extension as mcp_extension
from .tool_search import extension as tool_search_extension


def builtin_extensions() -> list[Any]:
    """Inline extensions bundled with pidrei.

    Entries are either a bare factory or an object with `name`, `factory` and
    optional `hidden` (pi's InlineExtension); the resource loader names them
    `<inline:name>` in the startup Extensions list.
    """
    return [
        # Replaceable: an extension that registers `codemode`, `tool_search`,
        # or `/mcp` (such as a third-party MCP extension) takes over instead
        # of running alongside the built-in one.
        InlineExtension(name="codemode", factory=codemode_extension, replaceable=True, builtin=True),
        InlineExtension(name="tool-search", factory=tool_search_extension, replaceable=True, builtin=True),
        InlineExtension(name="mcp", factory=mcp_extension, replaceable=True, builtin=True),
    ]


__all__ = ["builtin_extensions"]
