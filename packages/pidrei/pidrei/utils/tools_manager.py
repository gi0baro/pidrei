"""Managed-binary lookup for fd/ripgrep (from pi coding-agent src/utils/tools-manager.ts).

Lookup only: the managed bin dir first, then PATH. **pidrei does not download
tools** (decided 2026-07-27). pi fetches fd and ripgrep from GitHub releases
when they are missing — resolving the release, picking a platform asset,
downloading, extracting and verifying it. That is the one runtime-network
surface the Phase 7 purge otherwise removed, and both tools are a package
manager away on every platform pidrei runs on.

The consequence is real and deliberate: the `find` and `grep` tools do not work
without them. `MISSING_TOOL_HINT` is what the model is told, and it says so
plainly instead of pi's "could not be downloaded", which here would describe an
attempt that never happens.
"""

import os
import shutil
from collections.abc import Awaitable

import tonio.colored as tonio

from ..config import get_bin_dir


_SYSTEM_BINARY_NAMES = {
    "fd": ("fd", "fdfind"),
    "rg": ("rg",),
}


# Sync by design: `ensure_tool` hands this to `spawn_blocking`, so the PATH
# walk never runs on a runtime worker.
def get_tool_path(tool: str) -> str | None:
    names = _SYSTEM_BINARY_NAMES.get(tool)
    if names is None:
        return None

    # Check our bin directory first
    local_path = os.path.join(get_bin_dir(), tool)
    if os.path.exists(local_path):
        return local_path

    # Check system PATH
    for name in names:
        if shutil.which(name):
            return name
    return None


#: Install hints for the tools `find` and `grep` shell out to.
_INSTALL_HINTS = {
    "fd": "fd (https://github.com/sharkdp/fd)",
    "rg": "ripgrep (https://github.com/BurntSushi/ripgrep)",
}


def missing_tool_message(tool: str) -> str:
    """What the model sees when the binary is absent."""
    name = _INSTALL_HINTS.get(tool, tool)
    return f"{name} is not installed. Install it and make sure it is on PATH, or use the bash tool instead."


def ensure_tool(tool: str, on_status=None) -> Awaitable[str | None]:
    """Resolve a tool path. Never downloads — see the module docstring.

    ``on_status`` mirrors pi's download-progress callback (6f707eb3); with the
    download machinery unported there is nothing to report, so it is accepted
    for call-site parity and never invoked.
    """
    return tonio.spawn_blocking(get_tool_path, tool)
