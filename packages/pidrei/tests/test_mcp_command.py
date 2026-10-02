"""Mirror of pi's mcp-command.test.ts.

The stdio server is the mcp package's Python fixture (one `echo` tool), where
pi's is a node script. The missing command's error is Python's spawn error,
where pi's is node's `spawn ... ENOENT`.
"""

import json
import os
import sys
from pathlib import Path

import pytest

from pidrei.extensions.mcp.cli import McpCommandOptions, run_mcp_command


FIXTURE = str(Path(__file__).resolve().parents[2] / "mcp" / "tests" / "fixtures" / "stdio_server.py")

SERVERS = {
    "fixture": {"command": sys.executable, "args": [FIXTURE]},
    "broken": {"command": "pidrei-test-missing-mcp-server"},
    "parked": {"command": sys.executable, "args": [FIXTURE], "enabled": False},
    "bad": {"args": ["no command"]},
}


async def run(tmp_path, args: list[str], servers: dict | None, agent_dir: str | None = None):
    if agent_dir is None:
        agent_dir = str(tmp_path / f"agent-{len(list(tmp_path.iterdir()))}")
        os.mkdir(agent_dir)
    if servers is not None:
        Path(agent_dir, "mcp.json").write_text(json.dumps({"mcpServers": servers}))
    output: list[str] = []
    exit_code = await run_mcp_command(
        args, McpCommandOptions(cwd=agent_dir, agent_dir=agent_dir, log=output.append, error=output.append)
    )
    return exit_code, "\n".join(output), agent_dir


def read_config(path: str) -> dict:
    return json.loads(Path(path).read_text())


@pytest.mark.tonio
async def test_lists_servers_with_their_state_tools_and_errors_and_fails_while_anything_is_wrong(tmp_path):
    exit_code, output, _ = await run(tmp_path, ["list"], SERVERS)
    assert exit_code == 1
    assert "fixture: connected, 1 tool (codemode, global)\n" in output
    assert "  tools: echo" in output
    assert (
        "broken: failed (codemode, global)\n  pidrei-test-missing-mcp-server\n"
        "  [Errno 2] No such file or directory: 'pidrei-test-missing-mcp-server'"
    ) in output
    assert "parked: disabled (codemode, global)" in output
    assert "config error: " in output
    assert 'server "bad" needs either "command"' in output

    ok_code, _, _ = await run(tmp_path, ["list"], {"fixture": SERVERS["fixture"]})
    assert ok_code == 0


@pytest.mark.tonio
async def test_prints_json_for_scripts(tmp_path):
    exit_code, output, _ = await run(
        tmp_path, ["list", "--json"], {"fixture": SERVERS["fixture"], "parked": SERVERS["parked"]}
    )
    assert exit_code == 0
    parsed = json.loads(output)
    assert [{"name": s["name"], "state": s["state"], "tools": s["tools"]} for s in parsed["servers"]] == [
        {"name": "fixture", "state": "connected", "tools": ["echo"]},
        {"name": "parked", "state": "disabled", "tools": []},
    ]


@pytest.mark.tonio
async def test_rejects_unknown_servers_and_servers_without_oauth_for_login_and_logout(tmp_path):
    assert (await run(tmp_path, ["login", "nope"], SERVERS))[:2] == (
        1,
        'No MCP server named "nope". Configured: fixture, broken, parked.',
    )
    assert (await run(tmp_path, ["logout", "fixture"], SERVERS))[:2] == (
        1,
        'MCP server "fixture" does not use OAuth. Only HTTP servers without an Authorization header do.',
    )
    assert (await run(tmp_path, ["frobnicate"], SERVERS))[0] == 1


@pytest.mark.tonio
async def test_adds_stdio_servers_and_passes_options_after_the_command_through(tmp_path):
    exit_code, output, agent_dir = await run(
        tmp_path, ["add", "--env", "A=1", "--env", "B=x=y", "files", "--", "npx", "-y", "server", "--root", "."], None
    )
    assert exit_code == 0
    assert 'Added global MCP server "files"' in output
    assert read_config(os.path.join(agent_dir, "mcp.json")) == {
        "mcpServers": {
            "files": {"command": "npx", "args": ["-y", "server", "--root", "."], "env": {"A": "1", "B": "x=y"}}
        }
    }

    # Without `--`, options after the command belong to the command too.
    _, replaced, _ = await run(tmp_path, ["add", "files", "node", "server.js", "--port", "1"], None, agent_dir)
    assert 'Replaced global MCP server "files"' in replaced
    assert read_config(os.path.join(agent_dir, "mcp.json")) == {
        "mcpServers": {"files": {"command": "node", "args": ["server.js", "--port", "1"]}}
    }


@pytest.mark.tonio
async def test_adds_http_servers_and_keeps_other_content_of_the_file(tmp_path):
    exit_code, output, agent_dir = await run(
        tmp_path,
        [
            "add",
            "docs",
            "--url",
            "https://example.com/mcp",
            "--bearer-token-env-var",
            "DOCS_TOKEN",
            "--header",
            "X-Team=core",
            "--exposure",
            "direct",
            "--description",
            "Product docs",
        ],
        {"fixture": SERVERS["fixture"]},
    )
    assert exit_code == 0
    assert "mcp login" not in output
    assert read_config(os.path.join(agent_dir, "mcp.json")) == {
        "mcpServers": {
            "fixture": SERVERS["fixture"],
            "docs": {
                "url": "https://example.com/mcp",
                "headers": {"X-Team": "core", "Authorization": "Bearer ${DOCS_TOKEN}"},
                "exposure": "direct",
                "description": "Product docs",
            },
        }
    }

    _, oauth_output, _ = await run(
        tmp_path,
        [
            "add",
            "sentry",
            "--url",
            "https://mcp.sentry.dev/mcp",
            "--oauth-client-id",
            "pi",
            "--oauth-client-name",
            "Claude Code",
        ],
        None,
        agent_dir,
    )
    assert "If it requires sign-in: pidrei mcp login sentry" in oauth_output
    sentry = read_config(os.path.join(agent_dir, "mcp.json"))["mcpServers"]["sentry"]
    assert sentry["url"] == "https://mcp.sentry.dev/mcp"
    assert sentry["oauth"] == {"clientId": "pi", "clientName": "Claude Code"}


@pytest.mark.tonio
@pytest.mark.parametrize(
    "args",
    [
        ["add", "x"],
        ["add", "x", "--url", "https://example.com", "--", "cmd"],
        ["add", "bad name", "--", "cmd"],
        ["add", "x", "--url", "ftp://example.com"],
        ["add", "x", "--env", "A=1", "--url", "https://example.com"],
        ["add", "x", "--header", "A=1", "--", "cmd"],
        ["add", "x", "--env", "NOVALUE", "--", "cmd"],
        ["add", "x", "--exposure", "loud", "--", "cmd"],
    ],
)
async def test_rejects_invalid_add_invocations_without_writing(tmp_path, args):
    exit_code, _, agent_dir = await run(tmp_path, args, None)
    assert exit_code == 1, " ".join(args)
    assert not os.path.exists(os.path.join(agent_dir, "mcp.json"))


@pytest.mark.tonio
async def test_adds_and_removes_project_servers(tmp_path):
    _, added, agent_dir = await run(tmp_path, ["add", "-l", "local", "--", "node", "server.js"], None)
    assert "The project is not trusted" in added
    project_config = os.path.join(agent_dir, ".pidrei", "mcp.json")
    assert read_config(project_config) == {"mcpServers": {"local": {"command": "node", "args": ["server.js"]}}}

    wrong_code, wrong_scope, _ = await run(tmp_path, ["remove", "local"], None, agent_dir)
    assert wrong_code == 1
    assert f"It is defined in {project_config}; use --local." in wrong_scope

    removed_code, removed, _ = await run(tmp_path, ["remove", "local", "--local"], None, agent_dir)
    assert removed_code == 0
    assert 'Removed project MCP server "local"' in removed
    assert read_config(project_config) == {"mcpServers": {}}


@pytest.mark.tonio
async def test_removes_global_servers(tmp_path):
    exit_code, _, agent_dir = await run(tmp_path, ["remove", "broken"], SERVERS)
    assert exit_code == 0
    assert list(read_config(os.path.join(agent_dir, "mcp.json"))["mcpServers"]) == ["fixture", "parked", "bad"]
    missing_code, missing, _ = await run(tmp_path, ["remove", "broken"], None, agent_dir)
    assert missing_code == 1
    assert 'No global MCP server named "broken"' in missing
