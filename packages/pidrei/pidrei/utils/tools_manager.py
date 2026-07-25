"""Managed-binary lookup for fd/ripgrep (from pi coding-agent src/utils/tools-manager.ts).

Only the lookup half is ported: the managed bin dir is checked first, then
PATH. pi's GitHub-release download machinery is Phase 5; a missing tool
resolves to None and the calling tool reports it as unavailable.
"""

import os
import shutil

from ..config import get_bin_dir


_SYSTEM_BINARY_NAMES = {
    "fd": ("fd", "fdfind"),
    "rg": ("rg",),
}


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


async def ensure_tool(tool: str, silent: bool = False) -> str | None:
    """Resolve a tool path; downloading missing tools is Phase 5."""
    return get_tool_path(tool)
