"""Mirror of pi coding-agent src/modes/interactive/external-editor.ts."""

import os
import re
import shutil
import signal
import tempfile

import tonio.colored as tonio
from tonio.colored import fs

from ...core.output_guard import drain_output, write_stdout
from ...utils.text import strip_bom


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"signal {signum}"


async def edit_in_external_editor(options: dict) -> dict:
    """Run ``options["command"]`` on ``options["content"]`` in a temp file.

    Returns ``{"status": "complete", "content": ...}`` or
    ``{"status": "failed", "message": ...}``.
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
        write_stdout(f"Launching external editor: {options['command']}\npidrei will resume when the editor exits.\n")
        # On the terminal before the editor takes it over.
        await drain_output()

        # The editor inherits pidrei's cwd so project-relative paths, integrated terminals,
        # and similar editor features keep working.
        try:
            process = await tonio.open_process([editor, *editor_args, file_path])
            exit_code = await process.wait()
        except Exception:
            # The editor could not be launched.
            return {"status": "failed", "message": f"Failed to launch external editor: {options['command']}"}

        if exit_code != 0:
            # A negative return code is the signal that ended the editor.
            exit_text = f"code {exit_code}" if exit_code > 0 else _signal_name(-exit_code)
            return {"status": "failed", "message": f"External editor exited with {exit_text}; prompt unchanged"}

        content = strip_bom(await fs.Path(file_path).read_text(encoding="utf-8"))
        return {"status": "complete", "content": re.sub(r"\n$", "", content)}
    finally:
        # Cleanup is best effort.
        await tonio.spawn_blocking(shutil.rmtree, directory, ignore_errors=True)
