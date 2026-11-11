"""Mirror of pi mcp src/naming.ts.

Names of MCP tools and of their servers' namespaces as models and codemode
scripts see them. Applications that name tools the same way rank and group
them the same way in tool search and codemode.
"""

import hashlib
import re
from collections.abc import Callable


# Provider tool names are limited to 64 characters of `[A-Za-z0-9_-]`.
_MAX_TOOL_NAME_LENGTH = 64


def create_mcp_tool_name(server: str, tool: str, is_taken: Callable[[str], bool] = lambda _name: False) -> str:
    """`mcp__<server>__<tool>`, sanitized and shortened with a hash suffix
    when too long. Like Codex, everything but `[A-Za-z0-9_]` becomes `_`, so
    the name is also the identifier codemode scripts call it by. `is_taken`
    reports names used by a different MCP tool: sanitizing can map two tools
    to one name (`a-b` and `a_b`), which then get the hash suffix."""
    name = re.sub(r"[^A-Za-z0-9_]", "_", f"mcp__{server}__{tool}")
    if len(name) <= _MAX_TOOL_NAME_LENGTH and not is_taken(name):
        return name
    digest = hashlib.sha256(f"{server}\0{tool}".encode()).hexdigest()[:8]
    return f"{name[: _MAX_TOOL_NAME_LENGTH - len(digest) - 1]}_{digest}"


def mcp_namespace(server: str) -> str:
    """Namespace of a server's tools: `mcp__<server>` with `-` replaced by
    `_`, like the tool names. Server names that differ only in `-` and `_`
    share a namespace, so a configuration must not have both."""
    return f"mcp__{server.replace('-', '_')}"
