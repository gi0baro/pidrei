"""Mirror of pi coding-agent src/modes/interactive/external-editor.ts."""

import os
import re
import shutil
import sys
import tempfile

import tonio
from tonio.colored import fs


async def edit_in_external_editor(options: dict) -> dict:
    """Run ``options["command"]`` on ``options["content"]`` in a temp file.

    Returns ``{"status": "complete", "content": ...}`` or
    ``{"status": "failed"}``.
    """
    directory = await tonio.spawn_blocking(tempfile.mkdtemp, prefix="pidrei-editor-")
    file_path = os.path.join(directory, "prompt.md")
    try:
        # pi does these synchronously around the editor process; here they go
        # through `fs` like every other read/write. `tempfile` and
        # `shutil.rmtree` have no `fs` equivalent, so those use the pool
        # directly.
        await fs.Path(file_path).write_text(options["content"], encoding="utf-8")
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

        content = await fs.Path(file_path).read_text(encoding="utf-8")
        return {"status": "complete", "content": re.sub(r"\n$", "", content)}
    finally:
        # Cleanup is best effort.
        await tonio.spawn_blocking(shutil.rmtree, directory, ignore_errors=True)
