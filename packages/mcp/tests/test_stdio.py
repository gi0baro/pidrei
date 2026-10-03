"""Mirror of pi mcp test/stdio.test.ts. The Node fixtures are Python scripts
run with this interpreter."""

import sys
from pathlib import Path

import pytest
import tonio.colored as tonio

from pidrei_mcp import McpClient, StdioTransport


_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_WAIT_S = 5.0


def _stderr_watch(needle: str):
    """An `on_stderr` collecting the chunks, and an Event set once they
    contain `needle`."""
    chunks: list[str] = []
    seen = tonio.Event()

    async def on_stderr(chunk: str) -> None:
        chunks.append(chunk)
        if needle in "".join(chunks):
            seen.set()

    return chunks, seen, on_stderr


@pytest.mark.tonio
async def test_connects_to_a_newline_delimited_mcp_server_and_captures_stderr():
    chunks, ready, on_stderr = _stderr_watch("stdio fixture ready")
    transport = StdioTransport(sys.executable, args=[str(_FIXTURES / "stdio_server.py")], on_stderr=on_stderr)
    client = McpClient(name="stdio-test", version="1.0.0")
    await client.connect(transport)
    assert await client.list_tools() == [{"name": "echo", "inputSchema": {"type": "object"}}]
    assert await client.call_tool("echo", {"text": "hello"}) == {"content": [{"type": "text", "text": "hello"}]}
    assert isinstance(transport.pid, int)
    await ready.wait(_WAIT_S)
    assert "stdio fixture ready" in "".join(chunks)
    assert "stdio fixture ready" in transport.stderr
    await client.close()
    assert client.connection_state == "closed"


@pytest.mark.tonio
async def test_kills_a_server_that_ignores_shutdown_including_its_children():
    _chunks, spawned, on_stderr = _stderr_watch("grandchild ")
    transport = StdioTransport(
        sys.executable, args=[str(_FIXTURES / "stubborn_server.py")], close_timeout_ms=100, on_stderr=on_stderr
    )
    client = McpClient(name="stdio-test", version="1.0.0")
    await client.connect(transport)
    await spawned.wait(_WAIT_S)
    assert spawned.is_set()

    # The close waits for the server's stderr to reach EOF, and the
    # grandchild held it: completing means both are gone.
    _result, completed = await tonio.time.timeout(client.close(), _WAIT_S)
    assert completed
