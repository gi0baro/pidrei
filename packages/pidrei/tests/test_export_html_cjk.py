"""Partial mirror of pi coding-agent test/export-html-cjk.test.ts.

https://github.com/earendil-works/pi/issues/10154: CJK emphasis in the HTML export. pi runs the exported
viewer script in node:vm and parses markdown with it; the tests have no JS runtime, so this checks the
export embeds marked-cjk-friendly ahead of the viewer and the viewer hooks its emStrong tokenizer.
"""

import json
import re
from datetime import UTC, datetime

import pytest
from tonio.colored import fs

from pidrei.core.export_html import export_from_file
from pidrei.core.session_manager import CURRENT_SESSION_VERSION
from pidrei.modes.interactive.theme import init_theme


@pytest.mark.tonio
async def test_embeds_marked_cjk_friendly_and_hooks_its_em_strong_tokenizer(tmp_path):
    await init_theme("dark")
    header = {
        "type": "session",
        "version": CURRENT_SESSION_VERSION,
        "id": "cjk-test",
        "timestamp": datetime.fromtimestamp(0, UTC).isoformat().replace("+00:00", "Z"),
        "cwd": str(tmp_path),
    }
    session_path = str(tmp_path / "session.jsonl")
    await fs.Path(session_path).write_text(f"{json.dumps(header)}\n", encoding="utf-8")

    html = await fs.Path(await export_from_file(session_path, {"outputPath": str(tmp_path / "out.html")})).read_text(
        encoding="utf-8"
    )

    scripts = re.findall(r"<script>([\s\S]*?)</script>", html)
    viewer = scripts.pop()
    plugin = next(index for index, script in enumerate(scripts) if "globalThis.markedCjkFriendly=" in script)
    assert any("marked v18.0.11" in script for script in scripts[:plugin])
    assert "emStrong: markedCjkFriendly().tokenizer.emStrong" in viewer
