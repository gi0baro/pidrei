"""Mirror of pi coding-agent test/read-line-range-display.test.ts.

Pi runs the exported viewer script under Node; the test environment has no
JavaScript runtime, so the HTML half asserts on the exported document: the
session data the viewer renders and the viewer's use of its range helper.
"""

import base64
import json
import re
from types import SimpleNamespace

import pytest
from tonio.colored import fs

from pidrei.core.export_html import export_from_file
from pidrei.core.session_manager import CURRENT_SESSION_VERSION, SessionManager
from pidrei.core.tools.read import create_read_tool_definition
from pidrei.core.tools.render_utils import format_read_line_range
from pidrei.modes.interactive.components import ToolExecutionComponent, TreeSelectorComponent
from pidrei.modes.interactive.theme import init_theme
from pidrei.utils.ansi import strip_ansi


CASES = [
    (25, 13, ":25-37"),
    ("25", "13", ":25-37"),
    (None, "13", ":1-13"),
    ("25", None, ":25"),
    (None, None, ""),
    ("abc", 13, ""),
    (25, "", ""),
    (25, "Infinity", ""),
]


@pytest.fixture(autouse=True)
async def _theme():
    await init_theme("dark")


# #9887
@pytest.mark.parametrize(("offset", "limit", "line_range"), CASES)
def test_formats_raw_read_arguments(offset, limit, line_range):
    assert format_read_line_range(offset, limit) == line_range


@pytest.mark.tonio
async def test_shows_string_ranges_in_the_tui_session_tree_and_html_export(tmp_path):
    args = {"path": "notes.txt", "offset": "25", "limit": "13"}
    timestamp = "1970-01-01T00:00:00.000Z"
    assistant = {
        "type": "message",
        "id": "assistant",
        "parentId": None,
        "timestamp": timestamp,
        "message": {
            "role": "assistant",
            "content": [{"type": "toolCall", "id": "call", "name": "read", "arguments": args}],
            "api": "anthropic-messages",
            "provider": "anthropic",
            "model": "test-model",
            "usage": {
                "input": 0,
                "output": 0,
                "cacheRead": 0,
                "cacheWrite": 0,
                "totalTokens": 0,
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
            },
            "stopReason": "toolUse",
            "timestamp": 0,
        },
    }
    result = {
        "type": "message",
        "id": "result",
        "parentId": "assistant",
        "timestamp": timestamp,
        "message": {
            "role": "toolResult",
            "toolCallId": "call",
            "toolName": "read",
            "content": [],
            "isError": False,
            "timestamp": 0,
        },
    }
    header = {
        "type": "session",
        "version": CURRENT_SESSION_VERSION,
        "id": "range-test",
        "timestamp": timestamp,
        "cwd": str(tmp_path),
    }
    session_path = str(tmp_path / "session.jsonl")
    await fs.Path(session_path).write_text(
        "".join(f"{json.dumps(entry)}\n" for entry in (header, assistant, result)), encoding="utf-8"
    )

    tool = ToolExecutionComponent(
        "read",
        "call",
        args,
        {},
        create_read_tool_definition(str(tmp_path)),
        SimpleNamespace(request_render=lambda: None),
        str(tmp_path),
    )
    assert "read notes.txt:25-37" in strip_ansi("\n".join(tool.render(120)))

    session_manager = await SessionManager(session_file=session_path)
    tree = TreeSelectorComponent(session_manager.get_tree(), "result", 40, lambda _entry_id: None, lambda: None)
    assert "[read: notes.txt:25-37]" in "\n".join(strip_ansi(line) for line in tree.get_tree_list().render(120))

    html = await fs.Path(await export_from_file(session_path, {"outputPath": str(tmp_path / "out.html")})).read_text(
        encoding="utf-8"
    )
    session_data = re.search(r'<script id="session-data" type="application/json">([^<]+)</script>', html)
    assert session_data is not None
    entries = json.loads(base64.b64decode(session_data.group(1)))["entries"]
    assert entries[0]["message"]["content"][0]["arguments"] == args
    viewer = re.findall(r"<script>([\s\S]*?)</script>", html)[-1]
    assert "function formatReadLineRange(offset, limit) {" in viewer
    assert "return `[read: ${path}${formatReadLineRange(args.offset, args.limit)}]`;" in viewer
    assert "const readLineRange = formatReadLineRange(args.offset, args.limit);" in viewer
    assert '<span class="line-numbers">${escapeHtml(readLineRange)}</span>' in viewer
