"""Partial mirror of pi coding-agent test/export-html-deep-tree.test.ts.

https://github.com/earendil-works/pi/issues/11225: a session deeper than the call stack. pi runs the
exported viewer script in node:vm and walks the tree with it; the tests have no JS runtime, so this
exports a 20,000-entry chain (the Python half: loading, wire conversion, embedding) and checks the
viewer walks the tree without recursion.
"""

import base64
import json
import re
from datetime import UTC, datetime

import pytest
from tonio.colored import fs

from pidrei.core.export_html import export_from_file
from pidrei.core.session_manager import CURRENT_SESSION_VERSION
from pidrei.modes.interactive.theme import init_theme


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@pytest.mark.tonio
async def test_builds_and_flattens_a_single_child_chain_deeper_than_the_call_stack(tmp_path):
    depth = 20_000
    await init_theme("dark")
    header = {
        "type": "session",
        "version": CURRENT_SESSION_VERSION,
        "id": "deep-test",
        "timestamp": _iso(0),
        "cwd": str(tmp_path),
    }
    lines = [json.dumps(header)]
    for i in range(depth):
        entry = {
            "type": "message",
            "id": f"e{i}",
            "parentId": None if i == 0 else f"e{i - 1}",
            "timestamp": _iso(i * 1000),
            "message": {"role": "user", "content": "x", "timestamp": i * 1000},
        }
        lines.append(json.dumps(entry))
    session_path = str(tmp_path / "session.jsonl")
    await fs.Path(session_path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    html = await fs.Path(await export_from_file(session_path, {"outputPath": str(tmp_path / "out.html")})).read_text(
        encoding="utf-8"
    )

    session_data = re.search(r'<script id="session-data" type="application/json">([^<]+)</script>', html)
    assert session_data is not None
    data = json.loads(base64.b64decode(session_data[1]))
    assert [entry["id"] for entry in data["entries"]] == [f"e{i}" for i in range(depth)]
    assert data["leafId"] == f"e{depth - 1}"

    viewer = re.findall(r"<script>([\s\S]*?)</script>", html).pop()
    for walk in ("sortChildren", "mapNodes", "markActive"):
        assert f"function {walk}(" not in viewer
    assert "path.unshift(" not in viewer
