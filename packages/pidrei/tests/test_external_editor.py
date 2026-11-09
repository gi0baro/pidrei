"""Mirror of pi coding-agent test/external-editor.test.ts.

pi's fake editor is `fixtures/fake-external-editor.mjs` run by node; here it
is the Python script below, run by this interpreter.
"""

import json
import os
import sys
import tempfile

import pytest
from tonio.colored import fs

from pidrei.modes.interactive.external_editor import edit_in_external_editor


_FAKE_EDITOR = """
import json, os, signal, stat, sys

capture_path = sys.argv[1]
file_path = sys.argv[-1]
directory = os.path.dirname(file_path)
with open(file_path, encoding="utf-8") as handle:
    content = handle.read()
with open(capture_path, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "filePath": file_path,
            "content": content,
            "entries": sorted(os.listdir(directory)),
            "directoryMode": stat.S_IMODE(os.stat(directory).st_mode),
        },
        handle,
    )
if "--fail" in sys.argv:
    sys.exit(1)
if "--signal" in sys.argv:
    os.kill(os.getpid(), signal.SIGTERM)
with open(file_path, "w", encoding="utf-8") as handle:
    handle.write("" if "--empty" in sys.argv else "edited\\n")
"""


async def _run_external_editor(tmp_path, fixture_flag: str | None = None) -> tuple[dict, dict]:
    editor_path = tmp_path / "fake_external_editor.py"
    capture_path = tmp_path / "capture.json"
    await fs.Path(editor_path).write_text(_FAKE_EDITOR, encoding="utf-8")
    command = f"{sys.executable} {editor_path} {capture_path}" + (f" {fixture_flag}" if fixture_flag else "")
    result = await edit_in_external_editor({"command": command, "content": "original"})
    capture = json.loads(await fs.Path(capture_path).read_text(encoding="utf-8"))
    return result, capture


@pytest.mark.tonio
async def test_edits_a_prompt_inside_a_private_temporary_directory(tmp_path):
    result, capture = await _run_external_editor(tmp_path)
    directory = os.path.dirname(capture["filePath"])

    assert result == {"status": "complete", "content": "edited"}
    assert os.path.dirname(directory) == tempfile.gettempdir()
    assert os.path.basename(directory).startswith("pidrei-editor-")
    assert os.path.basename(capture["filePath"]) == "prompt.md"
    assert capture["entries"] == ["prompt.md"]
    assert capture["content"] == "original"
    assert capture["directoryMode"] & 0o077 == 0
    assert not await fs.Path(directory).exists()


@pytest.mark.tonio
async def test_keeps_the_original_content_when_the_editor_exits_unsuccessfully(tmp_path):
    result, capture = await _run_external_editor(tmp_path, "--fail")

    assert result == {"status": "failed", "message": "External editor exited with code 1; prompt unchanged"}
    assert not await fs.Path(os.path.dirname(capture["filePath"])).exists()


@pytest.mark.tonio
async def test_names_the_signal_that_ended_the_editor(tmp_path):
    # PiDrei: pi's message names the signal (Node's `close` signal); here it comes from the negative return code.
    result, _capture = await _run_external_editor(tmp_path, "--signal")

    assert result == {"status": "failed", "message": "External editor exited with SIGTERM; prompt unchanged"}


@pytest.mark.tonio
async def test_returns_empty_content_when_the_editor_clears_the_prompt(tmp_path):
    result, _capture = await _run_external_editor(tmp_path, "--empty")

    assert result == {"status": "complete", "content": ""}


# #11592: launch failures must be reported, not silently ignored.
@pytest.mark.tonio
async def test_reports_an_editor_that_cannot_be_launched():
    command = "pi-external-editor-does-not-exist"
    assert await edit_in_external_editor({"command": command, "content": "original"}) == {
        "status": "failed",
        "message": f"Failed to launch external editor: {command}",
    }
