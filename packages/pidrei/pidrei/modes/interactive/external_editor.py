"""Mirror of pi coding-agent src/modes/interactive/external-editor.ts."""

import os
import re
import shutil
import sys
import tempfile

import tonio


async def edit_in_external_editor(options: dict) -> dict:
    """Run ``options["command"]`` on ``options["content"]`` in a temp file.

    Returns ``{"status": "complete", "content": ...}`` or
    ``{"status": "failed"}``.
    """
    directory = tempfile.mkdtemp(prefix="pidrei-editor-")
    file_path = os.path.join(directory, "prompt.md")
    try:
        # Blocking file I/O on tiny temp files, exactly like pi's sync
        # write/read around the editor process.
        with open(file_path, "w", encoding="utf-8") as f:  # noqa: ASYNC230
            f.write(options["content"])
        editor, *editor_args = options["command"].split(" ")
        sys.stdout.write(
            f"Launching external editor: {options['command']}\npidrei will resume when the editor exits.\n"
        )
        sys.stdout.flush()

        try:
            process = tonio.open_process([editor, *editor_args, file_path])
            exit_code = await process.wait()
        except Exception:
            exit_code = None

        if exit_code != 0:
            return {"status": "failed"}

        with open(file_path, encoding="utf-8") as f:  # noqa: ASYNC230
            content = f.read()
        return {"status": "complete", "content": re.sub(r"\n$", "", content)}
    finally:
        # Cleanup is best effort.
        shutil.rmtree(directory, ignore_errors=True)
