"""Mirror of pi coding-agent src/core/session-export.ts.

pi extracted this out of `AgentSession.exportToJsonl` so its Radius share
path could append presentation entries after the conversation. Radius is
dropped surface here (see FEASIBILITY), so nothing in-tree passes
`create_trailing_entries` yet; the hook is kept so the module stays 1:1 with
upstream and any later export that needs a trailing entry has the seam.

Writing is async: pi's `writeFileSync` would block the runtime.
"""

import os
from collections.abc import Callable, Sequence
from typing import Any

from tonio.colored import fs

from ..utils.paths import resolve_path
from .session_manager import CURRENT_SESSION_VERSION, SessionManager, _dump_json, _entry_to_wire, _iso_now


async def export_session_to_jsonl(
    session_manager: SessionManager,
    output_path: str | None = None,
    create_trailing_entries: Callable[[str | None, str], Sequence[dict[str, Any]]] | None = None,
) -> str:
    """Write the current session branch and optional trailing export-only entries as JSONL."""
    default_name = f"session-{_iso_now().replace(':', '-').replace('.', '-')}.jsonl"
    file_path = resolve_path(output_path if output_path is not None else default_name, os.getcwd())
    directory = os.path.dirname(file_path)
    if directory and not await fs.Path(directory).exists():
        await fs.Path(directory).mkdir(parents=True, exist_ok=True)

    timestamp = _iso_now()
    header = {
        "type": "session",
        "version": CURRENT_SESSION_VERSION,
        "id": session_manager.get_session_id(),
        "timestamp": timestamp,
        "cwd": session_manager.get_cwd(),
    }
    lines = [_dump_json(header)]

    # Re-chain parentIds to form a linear sequence
    parent_id: str | None = None
    for entry in session_manager.get_branch():
        linear = dict(entry)
        linear["parentId"] = parent_id
        lines.append(_dump_json(_entry_to_wire(linear)))
        parent_id = entry["id"]

    if create_trailing_entries is not None:
        for entry in create_trailing_entries(parent_id, timestamp):
            lines.append(_dump_json(_entry_to_wire(dict(entry))))

    await fs.Path(file_path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
    return file_path


__all__ = ["export_session_to_jsonl"]
