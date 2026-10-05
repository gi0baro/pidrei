"""Mirror of pi coding-agent src/extensions/mcp/config.ts: MCP server
configuration.

Servers are read from `mcp.json` in the agent directory and, for trusted
projects, from `<project>/.pidrei/mcp.json`. Both use the `mcpServers` shape
shared by other MCP clients, so existing configurations can be copied over.
Project entries replace global entries with the same name.

A project entry without `command`, `url`, or `type` overrides only `enabled`,
`exposure`, and `toolExposure` of the global server with the same name, for
example to turn it off in one project:
`{ "mcpServers": { "internal-tools": { "enabled": false } } }`. The rest of the
global entry is kept, including credentials the project could not set itself.

```json
{
  "mcpServers": {
    "filesystem": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."] },
    "docs": { "url": "https://example.com/mcp", "headers": { "Authorization": "Bearer ${DOCS_TOKEN}" } },
    "sentry": { "url": "https://mcp.sentry.dev/mcp" }
  }
}
```

HTTP servers without an `Authorization` header use OAuth when they answer 401
(sign in with `/mcp`). `"auth": { "provider": "<provider>" }` sends the token
of a `/login` provider instead. Project files cannot use it, so a repository
cannot pick where the credential goes.

The top-level `autoEnableCodemode` (default true) activates the codemode tool
when a server with `codemode` exposure connects. A project value overrides
the global one.

pi reads and writes the files synchronously; here they go through
`tonio.colored.fs`. An edit (read, change, write) runs under a `FileLock` on
the file: pi's synchronous edit cannot interleave with another in the same
process, an awaited one can, and the lock also keeps edits from other
processes apart.
"""

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from tonio.colored import fs

from ...config import CONFIG_DIR_NAME
from ...core.mcp_servers import (
    McpExposure,
    McpServerConfig,
    get_mcp_tool_exposure,
    mcp_namespace,
    validate_mcp_server_config,
)
from ...utils.lockfile import FileLock


__all__ = [
    "LoadedMcpConfig",
    "McpExposure",
    "McpServerConfig",
    "McpServerConfigPatch",
    "McpServerEntry",
    "add_mcp_server_config",
    "get_mcp_tool_exposure",
    "load_mcp_config",
    "locale_order",
    "remove_mcp_server_config",
    "update_mcp_server_config",
]


type McpServerScope = Literal["global", "project", "extension"]


@dataclass(frozen=True, slots=True)
class McpServerEntry:
    name: str
    config: McpServerConfig
    # Config file that defined the entry, or the path of the extension that registered it.
    source: str
    # The global or the project `mcp.json`, or `extension` for servers
    # registered with `pi.register_mcp_server()`. Changes to extension servers
    # are not saved.
    scope: McpServerScope | None = None
    # Project `mcp.json` with an override of this global server's `enabled`, `exposure`, or `toolExposure`.
    override: str | None = None


@dataclass(frozen=True, slots=True)
class LoadedMcpConfig:
    servers: list[McpServerEntry]
    errors: list[str] = field(default_factory=list)
    # Activate the codemode tool when `codemode` servers connect. Default: true.
    auto_enable_codemode: bool | None = None
    # The project `mcp.json` when the project is trusted, where `/mcp` saves project overrides.
    project_config: str | None = None


_OVERRIDE_KEYS = ("enabled", "exposure", "toolExposure")


def _is_override(value: dict[str, Any]) -> bool:
    """Whether an entry overrides a server defined elsewhere instead of defining one."""
    return "command" not in value and "url" not in value and "type" not in value


def locale_order(text: str) -> tuple[str, str]:
    """Sort key approximating pi's `a.localeCompare(b)` for server names: case-insensitive first."""
    return (text.lower(), text)


def _is_record(value: Any) -> bool:
    return isinstance(value, dict)


@dataclass(slots=True)
class _McpConfigState:
    servers: dict[str, McpServerEntry] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    auto_enable_codemode: bool | None = None


async def _read_config_file(path: str, scope: Literal["global", "project"], state: _McpConfigState) -> None:
    servers, errors = state.servers, state.errors
    try:
        text = await fs.Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return
    except OSError as error:
        errors.append(f"{path}: {error}")
        return
    try:
        parsed = json.loads(text)
    except ValueError as error:
        errors.append(f"{path}: {error}")
        return
    if not _is_record(parsed) or ("mcpServers" in parsed and not _is_record(parsed["mcpServers"])):
        errors.append(f'{path}: expected an object with an "mcpServers" object')
        return
    if isinstance(parsed.get("autoEnableCodemode"), bool):
        state.auto_enable_codemode = parsed["autoEnableCodemode"]
    elif "autoEnableCodemode" in parsed:
        errors.append(f"{path}: autoEnableCodemode must be a boolean")
    for name, value in (parsed.get("mcpServers") or {}).items():
        if scope == "project" and _is_record(value) and _is_override(value):
            base = servers.get(name)
            extra = [key for key in value if key not in _OVERRIDE_KEYS]
            if base is None:
                errors.append(f'{path}: server "{name}" needs "command" or "url", or a global server to override')
            elif extra:
                errors.append(f'{path}: server "{name}": an override can only set {", ".join(_OVERRIDE_KEYS)}')
            else:
                merged = validate_mcp_server_config(name, {**base.config, **value})
                if isinstance(merged, str):
                    errors.append(f"{path}: {merged}")
                else:
                    servers[name] = replace(base, config=merged, override=path)
            continue
        config = validate_mcp_server_config(name, value)
        if isinstance(config, str):
            errors.append(f"{path}: {config}")
            continue
        # Names that differ only in `-` and `_` would share a namespace.
        clash = next(
            (other for other in servers if other != name and mcp_namespace(other) == mcp_namespace(name)),
            None,
        )
        if clash is not None:
            errors.append(f'{path}: server "{name}" conflicts with "{clash}"')
            continue
        if scope == "project" and "url" in config and config.get("auth"):
            errors.append(f'{path}: server "{name}": auth is only allowed in the global mcp.json')
            continue
        servers[name] = McpServerEntry(name=name, config=config, source=path, scope=scope)


async def load_mcp_config(*, agent_dir: str, cwd: str, project_trusted: bool) -> LoadedMcpConfig:
    """Load global and (when trusted) project MCP configuration. Disabled
    servers are included with `enabled: false`, so they can be enabled again."""
    state = _McpConfigState()
    await _read_config_file(os.path.join(agent_dir, "mcp.json"), "global", state)
    project_config = os.path.join(cwd, CONFIG_DIR_NAME, "mcp.json") if project_trusted else None
    if project_config is not None:
        await _read_config_file(project_config, "project", state)
    return LoadedMcpConfig(
        servers=list(state.servers.values()),
        errors=state.errors,
        auto_enable_codemode=state.auto_enable_codemode,
        project_config=project_config,
    )


@dataclass(frozen=True, slots=True)
class McpServerConfigPatch:
    """Settings `/mcp` changes; None leaves one unchanged. `enabled: True` and
    `exposure: "codemode"` are the defaults and remove the key."""

    enabled: bool | None = None
    exposure: McpExposure | None = None

    def as_config(self) -> dict[str, Any]:
        """The patch as config keys, for merging into an entry's config."""
        return {
            **({} if self.enabled is None else {"enabled": self.enabled}),
            **({} if self.exposure is None else {"exposure": self.exposure}),
        }


async def update_mcp_server_config(
    path: str, name: str, patch: McpServerConfigPatch, *, override: bool = False
) -> None:
    """Change one server's settings in the `mcp.json` that defines or overrides
    it. With `override`, a missing entry is added as an override. Overrides
    keep default values, since they replace the global server's. Other content
    is kept; the file is rewritten with its indentation."""

    def edit(servers: dict[str, Any] | None, parsed: dict[str, Any]) -> bool:
        server = servers.get(name) if servers is not None else None
        # pi's `server === undefined`: only a missing entry, not a JSON null.
        if override and (servers is None or name not in servers):
            server = {}
            parsed["mcpServers"] = {**(servers or {}), name: server}
        if not _is_record(server):
            raise Exception(f'{path} does not define MCP server "{name}"')
        keep_defaults = _is_override(server)
        if patch.enabled is not None:
            if patch.enabled and not keep_defaults:
                server.pop("enabled", None)
            else:
                server["enabled"] = patch.enabled
        if patch.exposure is not None:
            if patch.exposure == "codemode" and not keep_defaults:
                server.pop("exposure", None)
            else:
                server["exposure"] = patch.exposure
        return True

    await _edit_mcp_servers(path, edit)


async def add_mcp_server_config(path: str, name: str, config: McpServerConfig) -> bool:
    """Add a server to an `mcp.json`, creating the file when missing. An
    existing entry with the same name is replaced. Returns True when an entry
    was replaced."""
    replaced = False

    def edit(servers: dict[str, Any] | None, parsed: dict[str, Any]) -> bool:
        nonlocal replaced
        target = servers if servers is not None else {}
        replaced = name in target
        target[name] = config
        parsed["mcpServers"] = target
        return True

    await _edit_mcp_servers(path, edit)
    return replaced


async def remove_mcp_server_config(path: str, name: str) -> bool:
    """Remove a server from an `mcp.json`. Returns False when the file does not define it."""
    if not await fs.Path(path).exists():
        return False
    removed = False

    def edit(servers: dict[str, Any] | None, _parsed: dict[str, Any]) -> bool:
        nonlocal removed
        if servers is None or name not in servers:
            return False
        del servers[name]
        removed = True
        return True

    await _edit_mcp_servers(path, edit)
    return removed


_INDENT = re.compile(r"^([ \t]+)\S", re.MULTILINE)


async def _edit_mcp_servers(path: str, edit: Callable[[dict[str, Any] | None, dict[str, Any]], bool]) -> None:
    """Read an `mcp.json` (an empty config when missing), let `edit` change
    its `mcpServers`, and write it back with its indentation when `edit`
    returns True. Other content is kept."""
    file = fs.Path(path)
    # The lock lives next to the file, so its directory comes first.
    await file.parent.mkdir(parents=True, exist_ok=True)
    async with FileLock(path):
        try:
            text: str | None = await file.read_text(encoding="utf-8")
        except FileNotFoundError:
            text = None
        parsed = {} if text is None else json.loads(text)
        if not _is_record(parsed) or ("mcpServers" in parsed and not _is_record(parsed["mcpServers"])):
            raise Exception(f'{path}: expected an object with an "mcpServers" object')
        servers = parsed["mcpServers"] if _is_record(parsed.get("mcpServers")) else None
        if not edit(servers, parsed):
            return
        match = _INDENT.search(text) if text else None
        indent = match.group(1) if match else "  "
        await file.write_text(f"{json.dumps(parsed, indent=indent, ensure_ascii=False)}\n", encoding="utf-8")
