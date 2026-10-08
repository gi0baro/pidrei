"""Mirror of pi coding-agent src/extensions/mcp/cli.ts: `pidrei mcp`: add,
remove, and check MCP servers and sign in to them outside a session. Agents run
it through bash to configure servers, verify an `mcp.json` they wrote, and
start an OAuth sign-in; the user only approves access in the browser. Running
sessions pick up new credentials on their next turn.

The redirect URL pasted in a terminal is read from stdin through `FdReader`,
as a scope child (`run_cancellable`) that the sign-in's callback or the
`--timeout` cancels, where pi uses readline with an abort signal.
"""

import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import tonio.colored as tonio
from tonio.colored import fs

from pidrei_utils.cancel import CancelToken, run_cancellable
from pidrei_utils.timers import Timeout

from ...config import APP_NAME, CONFIG_DIR_NAME
from ...core.mcp_servers import validate_mcp_server_config
from ...core.output_guard import drain_output, write_stderr, write_stdout
from ...core.trust_manager import ProjectTrustStore
from ...utils.colors import bold, dim
from ...utils.fd_io import FdReader
from ...utils.open_browser import open_browser
from .config import (
    LoadedMcpConfig,
    McpServerEntry,
    add_mcp_server_config,
    get_mcp_tool_exposure,
    load_mcp_config,
    remove_mcp_server_config,
)
from .oauth import McpSignInCancelledError, sign_in_mcp_server
from .runtime import McpOAuthCredentialStore, McpServerConnection, McpServerLog, create_default_transport


HELP = f"""{bold("Usage:")}
  {APP_NAME} mcp add <server> [options] -- <command> [args...]
  {APP_NAME} mcp add <server> [options] --url <url>
  {APP_NAME} mcp remove <server> [-l]
  {APP_NAME} mcp list [--json]
  {APP_NAME} mcp login <server> [--timeout <seconds>]
  {APP_NAME} mcp logout <server>

Configure and check MCP servers and sign in to OAuth servers without starting a session.
Reads ~/{CONFIG_DIR_NAME}/agent/mcp.json and, in trusted projects, {CONFIG_DIR_NAME}/mcp.json.

Commands:
  add <server>            Add or replace a server in mcp.json
  remove <server>         Remove a server from mcp.json
  list                    Show state, tools, and errors (exits 1 on failure)
  login <server>          Sign in through the browser
  logout <server>         Delete the stored OAuth credentials

Options for add and remove:
  -l, --local             Use {CONFIG_DIR_NAME}/mcp.json in the current project instead of the global file

Options for add:
  --url <url>             Streamable HTTP server URL (instead of a command)
  --env <KEY=VALUE>       Environment variable for a stdio server (repeatable)
  --cwd <dir>             Working directory for a stdio server
  --header <KEY=VALUE>    HTTP header (repeatable)
  --bearer-token-env-var <NAME>
                          Send "Authorization: Bearer ${{NAME}}"
  --oauth-client-id <id>  Pre-registered OAuth client id
  --oauth-client-secret <secret>
                          OAuth client secret (may be ${{NAME}} or !command)
  --oauth-callback-port <port>
                          Fixed OAuth callback port
  --oauth-client-name <name>
                          Client name sent when registering with the OAuth server
  --exposure <mode>       codemode (default), deferred, direct, or hidden
  --description <text>    What the server offers, shown in the system prompt

Other options:
  --json                  Print the list as JSON
  --timeout <seconds>     How long login waits for the browser (default: 300)"""

HELP_HINT = dim(f'Use "{APP_NAME} mcp --help" for usage.')

_DEFAULT_LOGIN_TIMEOUT_SECONDS = 300

type Output = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class McpCommandOptions:
    cwd: str
    agent_dir: str
    # Defaults to `mcp-auth.json` in the agent directory.
    credentials: McpOAuthCredentialStore | None = None
    # Defaults to the platform browser.
    open_url: Callable[[str], None] | None = None
    # Default to the process's stdout and stderr.
    log: Output | None = None
    error: Output | None = None


def _describe_transport(entry: McpServerEntry) -> str:
    config = entry.config
    return config["url"] if "url" in config else " ".join([config["command"], *(config.get("args") or [])])


def _create_connection(
    entry: McpServerEntry, options: McpCommandOptions, credentials: McpOAuthCredentialStore
) -> McpServerConnection:
    return McpServerConnection(
        entry=entry,
        cwd=options.cwd,
        create_transport=create_default_transport,
        credentials=credentials,
        log=McpServerLog(os.path.join(options.agent_dir, "mcp.log")),
        on_tools=lambda _connection: None,
    )


# Short spellings of options. `-l`/`--local` match `pidrei install`.
_OPTION_ALIASES = {"-l": "--local"}

type _OptionKind = Literal["flag", "value", "list"]


@dataclass(slots=True)
class _ParsedOptions:
    positional: list[str]
    values: dict[str, str | Literal[True]]
    # Values of `list` options, in order.
    lists: dict[str, list[str]]

    def value(self, option: str) -> str | None:
        found = self.values.get(option)
        return found if isinstance(found, str) else None


def _parse_options(
    args: list[str], known: dict[str, _OptionKind], error: Output, max_positionals: float = float("inf")
) -> _ParsedOptions | None:
    """Parse `--name value` options; returns None and reports unknown ones.
    `--` ends the options, as does reaching `max_positionals` positional
    arguments: the remaining arguments are positional, so a command's own
    options (`add <server> <command> --flag`) are passed through."""
    parsed = _ParsedOptions(positional=[], values={}, lists={})
    index = 0
    while index < len(args):
        arg = _OPTION_ALIASES.get(args[index], args[index])
        if arg == "--" or len(parsed.positional) >= max_positionals:
            parsed.positional.extend(args[index + 1 if arg == "--" else index :])
            break
        if not arg.startswith("--"):
            parsed.positional.append(arg)
            index += 1
            continue
        name = arg[2:]
        kind = known.get(name)
        if kind is None:
            error(f"Unknown option {arg}.\n{HELP_HINT}")
            return None
        if kind == "flag":
            parsed.values[name] = True
            index += 1
            continue
        if index + 1 >= len(args):
            error(f"{arg} needs a value.")
            return None
        value = args[index + 1]
        index += 2
        if kind == "list":
            parsed.lists.setdefault(name, []).append(value)
        else:
            parsed.values[name] = value
    return parsed


def _write_line(write: Callable[[str], None]) -> Output:
    return lambda line: write(f"{line}\n")


async def run_mcp_command(args: list[str], options: McpCommandOptions) -> int:
    """Run `pidrei mcp <args>` and return the exit code."""
    log = options.log or _write_line(write_stdout)
    error = options.error or _write_line(write_stderr)
    command, rest = (args[0], args[1:]) if args else (None, [])
    if command is None or command == "help" or "--help" in args or "-h" in args:
        log(HELP)
        return 0

    project_config = os.path.join(options.cwd, CONFIG_DIR_NAME, "mcp.json")
    if command == "add":
        return await _add(rest, project_config, options, log, error)
    if command == "remove":
        return await _remove(rest, project_config, options, log, error)
    project_trusted = await ProjectTrustStore(options.agent_dir).get(options.cwd) is True
    loaded = await load_mcp_config(agent_dir=options.agent_dir, cwd=options.cwd, project_trusted=project_trusted)
    untrusted_note = (
        f"{project_config} is ignored because the project is not trusted. Start {APP_NAME} in the project to trust it."
        if not project_trusted and await fs.Path(project_config).exists()
        else None
    )
    credentials = options.credentials if options.credentials is not None else McpOAuthCredentialStore()

    match command:
        case "list":
            parsed = _parse_options(rest, {"json": "flag"}, error)
            if parsed is None:
                return 1
            if parsed.positional:
                error(f"Usage: {APP_NAME} mcp list [--json]\n{HELP_HINT}")
                return 1
            return await _list(loaded, "json" in parsed.values, untrusted_note, options, credentials, log)
        case "login" | "logout":
            parsed = _parse_options(rest, {"timeout": "value"} if command == "login" else {}, error)
            if parsed is None:
                return 1
            if len(parsed.positional) != 1:
                error(f"Usage: {APP_NAME} mcp {command} <server>\n{HELP_HINT}")
                return 1
            name = parsed.positional[0]
            entry = next((server for server in loaded.servers if server.name == name), None)
            if entry is None:
                configured = ", ".join(server.name for server in loaded.servers) or "none"
                note = f" {untrusted_note}" if untrusted_note else ""
                error(f'No MCP server named "{name}".{note} Configured: {configured}.')
                return 1
            connection = _create_connection(entry, options, credentials)
            url = connection.oauth_url
            if not url:
                error(f'MCP server "{name}" does not use OAuth. Only HTTP servers without an Authorization header do.')
                return 1
            if command == "logout":
                removed = await credentials.remove(name, url)
                log(
                    f'Signed out of MCP server "{name}".'
                    if removed
                    else f'No stored credentials for MCP server "{name}".'
                )
                return 0
            timeout = _seconds(parsed.value("timeout"))
            if timeout is None:
                error("--timeout must be a positive number of seconds.")
                return 1
            try:
                return await _login(entry, connection, url, timeout * 1000, options, credentials, log, error)
            finally:
                await connection.close()
        case _:
            error(f'Unknown mcp command "{command}".\n{HELP_HINT}')
            return 1


async def handle_mcp_command(args: list[str], *, cwd: str, agent_dir: str) -> bool | int:
    """False when `args` is not the mcp command; otherwise its exit code."""
    if not args or args[0] != "mcp":
        return False
    return await run_mcp_command(args[1:], McpCommandOptions(cwd=cwd, agent_dir=agent_dir))


def _seconds(value: str | None) -> float | None:
    """`Number(value ?? 300)`, positive and finite."""
    if value is None:
        return _DEFAULT_LOGIN_TIMEOUT_SECONDS
    try:
        seconds = float(value.strip() or "0")
    except ValueError:
        return None
    return seconds if 0 < seconds < float("inf") else None


def _parse_pairs(option: str, pairs: list[str] | None, error: Output) -> dict[str, str] | None:
    """Parse `KEY=VALUE` pairs of a repeatable option into a dict."""
    record: dict[str, str] = {}
    for pair in pairs or []:
        separator = pair.find("=")
        if separator <= 0:
            error(f'--{option} expects KEY=VALUE, got "{pair}".')
            return None
        record[pair[:separator]] = pair[separator + 1 :]
    return record


_HTTP_ONLY = (
    "header",
    "bearer-token-env-var",
    "oauth-client-id",
    "oauth-client-secret",
    "oauth-callback-port",
    "oauth-client-name",
)
_STDIO_ONLY = ("env", "cwd")


async def _add(args: list[str], project_config: str, options: McpCommandOptions, log: Output, error: Output) -> int:
    usage = f"Usage: {APP_NAME} mcp add <server> [options] (--url <url> | -- <command> [args...])\n{HELP_HINT}"
    parsed = _parse_options(
        args,
        {
            "local": "flag",
            "url": "value",
            "env": "list",
            "cwd": "value",
            "header": "list",
            "bearer-token-env-var": "value",
            "oauth-client-id": "value",
            "oauth-client-secret": "value",
            "oauth-callback-port": "value",
            "oauth-client-name": "value",
            "exposure": "value",
            "description": "value",
        },
        error,
        2,
    )
    if parsed is None:
        return 1
    name, command = (parsed.positional[0], parsed.positional[1:]) if parsed.positional else (None, [])
    url = parsed.values.get("url")
    if not name or (url is None) == (len(command) == 0):
        error(usage)
        return 1
    misplaced = next(
        (
            option
            for option in (_HTTP_ONLY if url is None else _STDIO_ONLY)
            if option in parsed.values or option in parsed.lists
        ),
        None,
    )
    if misplaced is not None:
        target = "HTTP servers (--url)" if url is None else "stdio servers"
        error(f"--{misplaced} only applies to {target}.")
        return 1

    config: dict[str, Any]
    if isinstance(url, str):
        headers = _parse_pairs("header", parsed.lists.get("header"), error)
        if headers is None:
            return 1
        bearer = parsed.value("bearer-token-env-var")
        if bearer is not None:
            headers["Authorization"] = f"Bearer ${{{bearer}}}"
        port = parsed.value("oauth-callback-port")
        oauth: dict[str, Any] = {}
        if parsed.value("oauth-client-id") is not None:
            oauth["clientId"] = parsed.value("oauth-client-id")
        if parsed.value("oauth-client-secret") is not None:
            oauth["clientSecret"] = parsed.value("oauth-client-secret")
        if port is not None:
            oauth["callbackPort"] = _js_number(port)
        if parsed.value("oauth-client-name") is not None:
            oauth["clientName"] = parsed.value("oauth-client-name")
        config = {"url": url, **({"headers": headers} if headers else {}), **({"oauth": oauth} if oauth else {})}
    else:
        env = _parse_pairs("env", parsed.lists.get("env"), error)
        if env is None:
            return 1
        executable, *command_args = command
        config = {
            "command": executable,
            **({"args": command_args} if command_args else {}),
            **({"env": env} if env else {}),
            **({"cwd": parsed.value("cwd")} if parsed.value("cwd") is not None else {}),
        }
    if parsed.value("exposure") is not None:
        config["exposure"] = parsed.value("exposure")
    if parsed.value("description") is not None:
        config["description"] = parsed.value("description")
    validated = validate_mcp_server_config(name, config)
    if isinstance(validated, str):
        error(validated)
        return 1

    project = "local" in parsed.values
    path = project_config if project else os.path.join(options.agent_dir, "mcp.json")
    scope = "project" if project else "global"
    try:
        replaced = await add_mcp_server_config(path, name, validated)
    except Exception as add_error:
        error(f"Could not update {path}: {add_error}")
        return 1
    log(f'{"Replaced" if replaced else "Added"} {scope} MCP server "{name}" in {path}.')
    if project and await ProjectTrustStore(options.agent_dir).get(options.cwd) is not True:
        log(f"The project is not trusted, so {path} is ignored until you start {APP_NAME} in the project and trust it.")
    # HTTP servers without an Authorization header may use OAuth.
    may_need_sign_in = "url" in validated and not any(
        header.lower() == "authorization" for header in validated.get("headers") or {}
    )
    sign_in = f". If it requires sign-in: {APP_NAME} mcp login {name}" if may_need_sign_in else ""
    log(f"Check it with: {APP_NAME} mcp list{sign_in}")
    return 0


def _js_number(value: str) -> int | float | str:
    """`Number(value)`: an int or float, left as the string when it is not a
    number (validation then rejects it, as pi's NaN is rejected)."""
    try:
        number = float(value.strip() or "0")
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


async def _remove(args: list[str], project_config: str, options: McpCommandOptions, log: Output, error: Output) -> int:
    parsed = _parse_options(args, {"local": "flag"}, error)
    if parsed is None:
        return 1
    if len(parsed.positional) != 1:
        error(f"Usage: {APP_NAME} mcp remove <server> [-l]\n{HELP_HINT}")
        return 1
    name = parsed.positional[0]
    project = "local" in parsed.values
    path = project_config if project else os.path.join(options.agent_dir, "mcp.json")
    scope = "project" if project else "global"
    try:
        removed = await remove_mcp_server_config(path, name)
    except Exception as remove_error:
        error(f"Could not update {path}: {remove_error}")
        return 1
    if removed:
        log(f'Removed {scope} MCP server "{name}" from {path}.')
        return 0
    loaded = await load_mcp_config(agent_dir=options.agent_dir, cwd=options.cwd, project_trusted=True)
    other = next((server for server in loaded.servers if server.name == name and server.scope != scope), None)
    hint = ""
    if other is not None:
        flag = "; use --local" if other.scope == "project" else "; omit --local"
        hint = f" It is defined in {other.source}{flag}."
    error(f'No {scope} MCP server named "{name}" in {path}.{hint}')
    return 1


async def _report(
    entry: McpServerEntry, options: McpCommandOptions, credentials: McpOAuthCredentialStore
) -> dict[str, Any]:
    exposure = entry.config.get("exposure") or "codemode"
    report: dict[str, Any] = {
        "name": entry.name,
        "scope": entry.scope or "global",
        "source": entry.source,
        # Project `mcp.json` that overrides `enabled`, `exposure`, or `toolExposure` of this global server.
        **({"override": entry.override} if entry.override else {}),
        "enabled": entry.config.get("enabled") is not False,
        "exposure": exposure,
        "transport": _describe_transport(entry),
        "state": "disabled",
        "tools": [],
    }
    if not report["enabled"]:
        return report
    connection = _create_connection(entry, options, credentials)
    try:
        await connection.get_client()
    except Exception:
        # The connection records the state and error.
        pass
    # One snapshot: the server can still change the connection (a dropped transport, a list change).
    view = connection.view
    report["state"] = view.state
    report["tools"] = [tool["name"] for tool in view.tools]
    overrides = {
        tool["name"]: tool_exposure
        for tool in view.tools
        if (tool_exposure := get_mcp_tool_exposure(entry.config, tool["name"])) != exposure
    }
    if overrides:
        report["toolExposure"] = overrides
    if view.has_resources:
        report["resources"] = len(view.resources)
        report["resourceTemplates"] = len(view.resource_templates)
    if view.state != "connected" and view.error:
        report["error"] = view.error
    await connection.close()
    return report


async def _list(
    loaded: LoadedMcpConfig,
    as_json: bool,
    untrusted_note: str | None,
    options: McpCommandOptions,
    credentials: McpOAuthCredentialStore,
    log: Output,
) -> int:
    handles = [tonio.spawn(_report(entry, options, credentials)) for entry in loaded.servers]
    reports = [await handle for handle in handles]
    failed = bool(loaded.errors) or any(report["enabled"] and report["state"] != "connected" for report in reports)

    if as_json:
        payload = {"servers": reports, "errors": loaded.errors, **({"note": untrusted_note} if untrusted_note else {})}
        log(json.dumps(payload, indent=2, ensure_ascii=False))
        return 1 if failed else 0
    if not reports and not loaded.errors:
        log(
            "No MCP servers configured. Add them to "
            f"{os.path.join(options.agent_dir, 'mcp.json')} or {CONFIG_DIR_NAME}/mcp.json."
        )
    for report in reports:
        tools = report["tools"]
        if report["state"] == "connected":
            state = f"connected, {len(tools)} tool{'' if len(tools) == 1 else 's'}"
        elif report["state"] == "needs-auth":
            state = "needs sign-in"
        else:
            state = report["state"]
        log(f"{report['name']}: {state} ({report['exposure']}, {report['scope']})")
        log(f"  {report['transport']}")
        if report.get("override"):
            log(f"  project override: {report['override']}")
        if report["state"] == "needs-auth":
            log(f"  sign in with: {APP_NAME} mcp login {report['name']}")
        if tools:
            overrides = report.get("toolExposure") or {}
            listed = [f"{tool} [{overrides[tool]}]" if tool in overrides else tool for tool in tools]
            log(f"  tools: {', '.join(listed)}")
        if "resources" in report:
            log(f"  resources: {report['resources']}, URI templates: {report.get('resourceTemplates', 0)}")
        if report.get("error"):
            log("  " + "\n  ".join(report["error"].split("\n")))
    for config_error in loaded.errors:
        log(f"config error: {config_error}")
    if untrusted_note:
        log(untrusted_note)
    return 1 if failed else 0


async def _login(
    entry: McpServerEntry,
    connection: McpServerConnection,
    url: str,
    timeout_ms: float,
    options: McpCommandOptions,
    credentials: McpOAuthCredentialStore,
    log: Output,
    error: Output,
) -> int:
    name = entry.name
    # Connecting first answers whether a sign-in is needed and records the server's challenge.
    try:
        await connection.get_client()
        log(f'Already signed in to MCP server "{name}" ({len(connection.tools)} tools).')
        return 0
    except Exception:
        view = connection.view
        if view.state != "needs-auth":
            error(f'MCP server "{name}" failed to connect: {view.error or "unknown error"}')
            return 1

    open_url = options.open_url or open_browser
    interactive = sys.stdin.isatty() and options.open_url is None

    class Prompt:
        def show_authorization_url(self, authorization_url: str) -> None:
            log(f'Sign in to MCP server "{name}" in your browser:\n{authorization_url}')
            open_url(authorization_url)

        def prompt_for_redirect_url(self, cancel: CancelToken):
            return _wait_for_redirect_url(cancel, interactive)

    # `--timeout` bounds the whole sign-in (pi: `signal: AbortSignal.timeout(timeoutMs)`).
    cancel = CancelToken()
    timer = Timeout(timeout_ms, cancel.cancel)
    try:
        await sign_in_mcp_server(
            server_url=url,
            store=credentials.for_server(name, url),
            settings=await connection.oauth_settings(),
            challenge=connection.challenge,
            prompt=Prompt(),
            cancel=cancel,
        )
    except McpSignInCancelledError:
        error(
            f'Sign-in to MCP server "{name}" was cancelled or not completed within {round(timeout_ms / 1000)} seconds.'
        )
        return 1
    except Exception as sign_in_error:
        error(f'Sign-in to MCP server "{name}" failed: {sign_in_error}')
        return 1
    finally:
        timer.cancel()
    connection.challenge = None
    try:
        await connection.reconnect()
    except Exception as connect_error:
        error(f"Signed in, but {connect_error}")
        return 1
    log(f'Signed in to MCP server "{name}" ({len(connection.tools)} tools).')
    return 0


async def _read_line() -> str:
    """One line from stdin, read on readiness (see `main._prompt_confirm`)."""
    reader = FdReader(sys.stdin.fileno())
    buffer = b""
    try:
        while b"\n" not in buffer:
            chunk = await reader.read()
            if not chunk:
                break
            buffer += chunk
    finally:
        reader.close()
    lines = buffer.decode("utf-8", "replace").splitlines()
    return lines[0] if lines else ""


async def _wait_for_redirect_url(cancel: CancelToken, interactive: bool) -> str | None:
    """The pasted redirect URL in a terminal; otherwise only the browser
    callback can finish the sign-in. None when `cancel` fires: the callback
    arrived, or the sign-in timed out."""
    if not interactive:
        await cancel.event.wait()
        return None
    write_stderr("If the browser cannot reach this machine, paste the URL it was redirected to: ")
    await drain_output()
    try:
        return await run_cancellable(_read_line(), cancel)
    except Exception:
        return None
