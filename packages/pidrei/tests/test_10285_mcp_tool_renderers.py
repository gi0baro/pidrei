"""Mirror of pi's suite/regressions/10285-mcp-tool-renderers.test.ts."""

import base64
import json
import os
import re
import shutil
import tempfile

import pytest

from pidrei.core.session_manager import SessionManager
from pidrei.extensions.mcp import create_mcp_extension
from pidrei.extensions.mcp.config import LoadedMcpConfig
from pidrei.modes.interactive.theme import init_theme, theme
from pidrei.utils.ansi import strip_ansi
from pidrei_ai.types import AssistantMessage, ToolCall, Usage, UsageCost, UserMessage

from .harness import create_harness


async def _load_config(_ctx) -> LoadedMcpConfig:
    return LoadedMcpConfig(servers=[], errors=[])


mcp_extension = create_mcp_extension(load_config=_load_config)


# Regression #10285: a resumed session renders MCP tool calls before their server connected, if it
# ever does. They render with the MCP renderers anyway, instead of the expanded fallback.
@pytest.mark.tonio
async def test_renders_calls_to_mcp_tools_that_are_not_registered():
    await init_theme("dark")
    harness = await create_harness(extension_factories=[mcp_extension])
    try:

        def resolve(tool_name: str):
            return harness.session.extension_runner.resolve_tool_renderers(
                tool_name, lambda: harness.session.get_tool_definition(tool_name)
            )

        call = resolve("mcp__my_docs__search").render_call({"query": "pi"}, theme, {"expanded": False})
        assert 'my_docs/search query="pi"' in strip_ansi("\n".join(call.render(100)))
        assert resolve("not_mcp") is None
        # Registered tools keep their own renderers.
        assert resolve("read").render_call is harness.session.get_tool_definition("read").render_call
    finally:
        harness.cleanup()


@pytest.mark.tonio
async def test_renders_them_in_html_exports_too():
    await init_theme("dark")
    directory = tempfile.mkdtemp(prefix="pidrei-10285-")
    session_manager = await SessionManager(directory, os.path.join(directory, "sessions"))
    harness = await create_harness(extension_factories=[mcp_extension], session_manager=session_manager)
    try:
        await session_manager.append_message(UserMessage(content="search", timestamp=1))
        await session_manager.append_message(
            AssistantMessage(
                content=[ToolCall(id="call-1", name="mcp__my_docs__search", arguments={"query": "pi"})],
                api="anthropic-messages",
                provider="anthropic",
                model="test",
                usage=Usage(input=0, output=0, cache_read=0, cache_write=0, total_tokens=0, cost=UsageCost()),
                stop_reason="toolUse",
                timestamp=2,
            )
        )

        output = await harness.session.export_to_html(os.path.join(directory, "export.html"))
        with open(output, encoding="utf-8") as file:
            html = file.read()
        match = re.search(r'<script id="session-data" type="application/json">([^<]*)</script>', html)
        session = json.loads(base64.b64decode(match[1] if match else "").decode("utf-8"))
        rendered = (session.get("renderedTools") or {}).get("call-1") or {}
        assert "my_docs/search" in strip_ansi(rendered.get("callHtml") or "")
    finally:
        harness.cleanup()
        shutil.rmtree(directory, ignore_errors=True)
