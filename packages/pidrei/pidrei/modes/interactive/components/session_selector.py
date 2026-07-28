"""Mirror of pi coding-agent src/modes/interactive/components/session-selector.ts."""

import os
import re
import time
from collections.abc import Awaitable
from datetime import datetime
from typing import Any

import tonio.colored as tonio
from tonio.colored import fs

from pidrei_tui import Container, Input, Spacer, Text, get_keybindings, truncate_to_width, visible_width
from pidrei_tui._timers import Timeout

from ....utils.paths import canonicalize_path as _canonicalize_path
from ....utils.process import run_command
from ..theme import theme
from .dynamic_border import DynamicBorder
from .keybinding_hints import key_hint, key_text
from .session_selector_search import filter_and_sort_sessions, has_session_name


_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")


def _shorten_path(path: str) -> str:
    home = os.path.expanduser("~")
    if not path:
        return path
    if path.startswith(home):
        return f"~{path[len(home) :]}"
    return path


def format_session_date(date: datetime) -> str:
    now = time.time()
    diff_s = now - date.timestamp()
    diff_mins = int(diff_s // 60)
    diff_hours = int(diff_s // 3600)
    diff_days = int(diff_s // 86400)

    if diff_mins < 1:
        return "now"
    if diff_mins < 60:
        return f"{diff_mins}m"
    if diff_hours < 24:
        return f"{diff_hours}h"
    if diff_days < 7:
        return f"{diff_days}d"
    if diff_days < 30:
        return f"{diff_days // 7}w"
    if diff_days < 365:
        return f"{diff_days // 30}mo"
    return f"{diff_days // 365}y"


def _canonicalize(path: str | None) -> str | None:
    if not path:
        return path
    return _canonicalize_path(path)


class SessionSelectorHeader:
    def __init__(self, scope: str, sort_mode: str, name_filter: str, request_render) -> None:
        self._scope = scope
        self._sort_mode = sort_mode
        self._name_filter = name_filter
        self._request_render = request_render
        self._loading = False
        self._load_progress: dict | None = None
        self._show_path = False
        self._confirming_delete_path: str | None = None
        self._status_message: dict | None = None
        self._status_timeout: Timeout | None = None
        self._show_rename_hint = False

    def set_scope(self, scope: str) -> None:
        self._scope = scope

    def set_sort_mode(self, sort_mode: str) -> None:
        self._sort_mode = sort_mode

    def set_name_filter(self, name_filter: str) -> None:
        self._name_filter = name_filter

    def set_loading(self, loading: bool) -> None:
        self._loading = loading
        # Progress is scoped to the current load; clear whenever the loading
        # state is set
        self._load_progress = None

    def set_progress(self, loaded: int, total: int) -> None:
        self._load_progress = {"loaded": loaded, "total": total}

    def set_show_path(self, show_path: bool) -> None:
        self._show_path = show_path

    def set_show_rename_hint(self, show: bool) -> None:
        self._show_rename_hint = show

    def set_confirming_delete_path(self, path: str | None) -> None:
        self._confirming_delete_path = path

    def _clear_status_timeout(self) -> None:
        if self._status_timeout is None:
            return
        self._status_timeout.cancel()
        self._status_timeout = None

    def set_status_message(self, msg: dict | None, auto_hide_ms: float | None = None) -> None:
        """``msg`` is a ``{"type": "info" | "error", "message"}`` record."""
        self._clear_status_timeout()
        self._status_message = msg
        if not msg or not auto_hide_ms:
            return

        def hide() -> None:
            self._status_message = None
            self._status_timeout = None
            self._request_render()

        self._status_timeout = Timeout(auto_hide_ms, hide)

    def invalidate(self) -> None:
        pass

    def render(self, width: int) -> list:
        title = "Resume Session (Current Folder)" if self._scope == "current" else "Resume Session (All)"
        left_text = theme.bold(title)

        if self._sort_mode == "threaded":
            sort_label = "Threaded"
        elif self._sort_mode == "recent":
            sort_label = "Recent"
        else:
            sort_label = "Fuzzy"
        sort_text = theme.fg("muted", "Sort: ") + theme.fg("accent", sort_label)

        name_label = "All" if self._name_filter == "all" else "Named"
        name_text = theme.fg("muted", "Name: ") + theme.fg("accent", name_label)

        if self._loading:
            if self._load_progress:
                progress_text = f"{self._load_progress['loaded']}/{self._load_progress['total']}"
            else:
                progress_text = "..."
            scope_text = theme.fg("muted", "○ Current Folder | ") + theme.fg("accent", f"Loading {progress_text}")
        elif self._scope == "current":
            scope_text = theme.fg("accent", "◉ Current Folder") + theme.fg("muted", " | ○ All")
        else:
            scope_text = theme.fg("muted", "○ Current Folder | ") + theme.fg("accent", "◉ All")

        right_text = truncate_to_width(f"{scope_text}  {name_text}  {sort_text}", width, "")
        available_left = max(0, width - visible_width(right_text) - 1)
        left = truncate_to_width(left_text, available_left, "")
        spacing = max(0, width - visible_width(left) - visible_width(right_text))

        # Build hint lines - changes based on state (all branches truncate to
        # width)
        if self._confirming_delete_path is not None:
            confirm_hint = (
                f"Delete session? {key_hint('tui.select.confirm', 'confirm')} · "
                f"{key_hint('tui.select.cancel', 'cancel')}"
            )
            hint_line1 = theme.fg("error", truncate_to_width(confirm_hint, width, "…"))
            hint_line2 = ""
        elif self._status_message:
            color = "error" if self._status_message["type"] == "error" else "accent"
            hint_line1 = theme.fg(color, truncate_to_width(self._status_message["message"], width, "…"))
            hint_line2 = ""
        else:
            path_state = "(on)" if self._show_path else "(off)"
            sep = theme.fg("muted", " · ")
            hint1 = key_hint("tui.input.tab", "scope") + sep + theme.fg("muted", 're:<pattern> regex · "phrase" exact')
            hint2_parts = [
                key_hint("app.session.toggleSort", "sort"),
                key_hint("app.session.toggleNamedFilter", "named"),
                key_hint("app.session.delete", "delete"),
                key_hint("app.session.togglePath", f"path {path_state}"),
            ]
            if self._show_rename_hint:
                hint2_parts.append(key_hint("app.session.rename", "rename"))
            hint2 = sep.join(hint2_parts)
            hint_line1 = truncate_to_width(hint1, width, "…")
            hint_line2 = truncate_to_width(hint2, width, "…")

        return [f"{left}{' ' * spacing}{right_text}", hint_line1, hint_line2]


def build_canonical_path_map(sessions: list) -> dict[str, str]:
    """Canonical form of every path a session tree needs, resolved once.

    Blocking (one `realpath` per entry), so callers offload it. It is kept
    out of `build_session_tree` because that runs on every keystroke in the
    search box, and resolving symlinks per keypress meant a filesystem storm
    on a runtime worker.
    """
    canonical: dict[str, str] = {}
    for session in sessions:
        for path in (session.path, session.parent_session_path):
            if path and path not in canonical:
                canonical[path] = _canonicalize_path(path)
    return canonical


def build_session_tree(sessions: list, canonical_by_path: dict[str, str] | None = None) -> list:
    """Build a tree from sessions based on parent_session_path.

    Returns root ``{"session", "children", "latestActivity"}`` nodes sorted
    by latest subtree activity (descending).

    `canonical_by_path` comes from `build_canonical_path_map`. Without it the
    paths are resolved inline, which touches the filesystem — only acceptable
    off the runtime.
    """
    lookup = canonical_by_path if canonical_by_path is not None else {}

    def canonical(path: str | None) -> str | None:
        if not path:
            return path
        cached = lookup.get(path)
        return cached if cached is not None else _canonicalize(path)

    by_path: dict = {}

    for session in sessions:
        session_path = canonical(session.path) or session.path
        by_path[session_path] = {
            "session": session,
            "children": [],
            "latestActivity": session.modified.timestamp(),
        }

    roots: list = []

    for session in sessions:
        session_path = canonical(session.path) or session.path
        node = by_path[session_path]
        parent_path = canonical(session.parent_session_path)

        if parent_path and parent_path in by_path:
            by_path[parent_path]["children"].append(node)
        else:
            roots.append(node)

    def update_latest_activity(node: dict) -> float:
        latest_activity = node["session"].modified.timestamp()
        for child in node["children"]:
            latest_activity = max(latest_activity, update_latest_activity(child))
        node["latestActivity"] = latest_activity
        return latest_activity

    for root in roots:
        update_latest_activity(root)

    # Sort children and roots by latest activity in each subtree (descending)
    def sort_nodes(nodes: list) -> None:
        nodes.sort(key=lambda n: -n["latestActivity"])
        for node in nodes:
            sort_nodes(node["children"])

    sort_nodes(roots)

    return roots


def flatten_session_tree(roots: list) -> list:
    """Flatten tree into ``{"session", "depth", "isLast", "ancestorContinues"}`` nodes."""
    result: list = []

    def walk(node: dict, depth: int, ancestor_continues: list, is_last: bool) -> None:
        result.append(
            {"session": node["session"], "depth": depth, "isLast": is_last, "ancestorContinues": ancestor_continues}
        )

        for i, child in enumerate(node["children"]):
            child_is_last = i == len(node["children"]) - 1
            # Only show continuation line for non-root ancestors
            continues = (not is_last) if depth > 0 else False
            walk(child, depth + 1, [*ancestor_continues, continues], child_is_last)

    for i, root in enumerate(roots):
        walk(root, 0, [], i == len(roots) - 1)

    return result


class SessionList:
    """Custom session list component with multi-line items and search."""

    def __init__(
        self,
        sessions: list,
        show_cwd: bool,
        sort_mode: str,
        name_filter: str,
        keybindings,
        current_session_file_path: str | None = None,
    ) -> None:
        self._all_sessions = sessions
        # Populated by `set_sessions`; empty here because a constructor cannot
        # await, and `build_session_tree` falls back to resolving inline.
        self._canonical_by_path: dict[str, str] = {}
        self._filtered_sessions: list = []
        self._selected_index = 0
        self._search_input = Input()
        self._show_cwd = show_cwd
        self._sort_mode = sort_mode
        self._name_filter = name_filter
        self._keybindings = keybindings
        self._show_path = False
        self._confirming_delete_path: str | None = None
        self._current_session_canonical_path = _canonicalize(current_session_file_path)
        self._max_visible = 10  # Max sessions visible (one line each)
        self._focused = False

        self.on_select = None
        self.on_cancel = None
        self.on_exit = lambda: None
        self.on_toggle_scope = None
        self.on_toggle_sort = None
        self.on_toggle_name_filter = None
        self.on_toggle_path = None
        self.on_delete_confirmation_change = None
        self.on_delete_session = None  # async callable
        self.on_rename_session = None
        self.on_error = None

        self._filter_sessions("")

        # Handle Enter in search input - select current item
        def on_submit(_value=None) -> None:
            if 0 <= self._selected_index < len(self._filtered_sessions) and self.on_select is not None:
                self.on_select(self._filtered_sessions[self._selected_index]["session"].path)

        self._search_input.on_submit = on_submit

    # Focusable implementation - propagate to search input for IME cursor
    # positioning
    @property
    def focused(self) -> bool:
        return self._focused

    @focused.setter
    def focused(self, value: bool) -> None:
        self._focused = value
        self._search_input.focused = value

    def get_selected_session_path(self) -> str | None:
        if 0 <= self._selected_index < len(self._filtered_sessions):
            return self._filtered_sessions[self._selected_index]["session"].path
        return None

    def set_sort_mode(self, sort_mode: str) -> None:
        self._sort_mode = sort_mode
        self._filter_sessions(self._search_input.get_value())

    def set_name_filter(self, name_filter: str) -> None:
        self._name_filter = name_filter
        self._filter_sessions(self._search_input.get_value())

    async def set_sessions(self, sessions: list, show_cwd: bool) -> None:
        self._all_sessions = sessions
        self._show_cwd = show_cwd
        # Resolve every session path once here, off the runtime, so the
        # per-keystroke filter below never touches the filesystem.
        self._canonical_by_path = await tonio.spawn_blocking(build_canonical_path_map, sessions)
        self._filter_sessions(self._search_input.get_value())

    def _filter_sessions(self, query: str) -> None:
        trimmed = query.strip()
        if self._name_filter == "all":
            name_filtered = self._all_sessions
        else:
            name_filtered = [session for session in self._all_sessions if has_session_name(session)]

        if self._sort_mode == "threaded" and not trimmed:
            # Threaded mode without search: show tree structure
            roots = build_session_tree(name_filtered, self._canonical_by_path)
            self._filtered_sessions = flatten_session_tree(roots)
        else:
            # Other modes or with search: flat list
            filtered = filter_and_sort_sessions(name_filtered, query, self._sort_mode, "all")
            self._filtered_sessions = [
                {"session": session, "depth": 0, "isLast": True, "ancestorContinues": []} for session in filtered
            ]
        self._selected_index = min(self._selected_index, max(0, len(self._filtered_sessions) - 1))

    def _set_confirming_delete_path(self, path: str | None) -> None:
        self._confirming_delete_path = path
        if self.on_delete_confirmation_change is not None:
            self.on_delete_confirmation_change(path)

    def _start_delete_confirmation_for_selected_session(self) -> None:
        if not (0 <= self._selected_index < len(self._filtered_sessions)):
            return
        selected = self._filtered_sessions[self._selected_index]

        # Prevent deleting current session
        if self._is_current_session_path(selected["session"].path):
            if self.on_error is not None:
                self.on_error("Cannot delete the currently active session")
            return

        self._set_confirming_delete_path(selected["session"].path)

    def _is_current_session_path(self, path: str) -> bool:
        if not self._current_session_canonical_path:
            return False
        return (_canonicalize(path) or path) == self._current_session_canonical_path

    def invalidate(self) -> None:
        pass

    def render(self, width: int) -> list:
        lines: list = []

        # Render search input
        lines.extend(self._search_input.render(width))
        lines.append("")  # Blank line after search

        if not self._filtered_sessions:
            if self._name_filter == "named":
                toggle_key = key_text("app.session.toggleNamedFilter")
                if self._show_cwd:
                    empty_message = f"  No named sessions found. Press {toggle_key} to show all."
                else:
                    empty_message = (
                        f"  No named sessions in current folder. Press {toggle_key} to show all, or Tab to view all."
                    )
            elif self._show_cwd:
                # "All" scope - no sessions anywhere that match filter
                empty_message = "  No sessions found"
            else:
                # "Current folder" scope - hint to try "all"
                empty_message = "  No sessions in current folder. Press Tab to view all."
            lines.append(theme.fg("muted", truncate_to_width(empty_message, width, "…")))
            return lines

        # Calculate visible range with scrolling
        start_index = max(
            0,
            min(self._selected_index - self._max_visible // 2, len(self._filtered_sessions) - self._max_visible),
        )
        end_index = min(start_index + self._max_visible, len(self._filtered_sessions))

        # Render visible sessions (one line each with tree structure)
        for i in range(start_index, end_index):
            node = self._filtered_sessions[i]
            session = node["session"]
            is_selected = i == self._selected_index
            is_confirming_delete = session.path == self._confirming_delete_path
            is_current = self._is_current_session_path(session.path)

            # Build tree prefix
            prefix = self._build_tree_prefix(node)

            # Session display text (name or first message)
            has_name = bool(session.name)
            display_text = session.name if session.name is not None else session.first_message
            normalized_message = _CONTROL_CHARS_RE.sub(" ", display_text).strip()

            # Right side: message count and age
            age = format_session_date(session.modified)
            msg_count = str(session.message_count)
            right_part = f"{msg_count} {age}"
            if self._show_cwd and session.cwd:
                right_part = f"{_shorten_path(session.cwd)} {right_part}"
            if self._show_path:
                right_part = f"{_shorten_path(session.path)} {right_part}"

            # Cursor
            cursor = theme.fg("accent", "› ") if is_selected else "  "

            # Calculate available width for message
            prefix_width = visible_width(prefix)
            right_width = visible_width(right_part) + 2  # +2 for spacing
            available_for_msg = width - 2 - prefix_width - right_width  # -2 for cursor

            truncated_msg = truncate_to_width(normalized_message, max(10, available_for_msg), "…")

            # Style message
            if is_confirming_delete:
                message_color = "error"
            elif is_current:
                message_color = "accent"
            elif has_name:
                message_color = "warning"
            else:
                message_color = None
            styled_msg = theme.fg(message_color, truncated_msg) if message_color else truncated_msg
            if is_selected:
                styled_msg = theme.bold(styled_msg)

            # Build line
            left_part = cursor + theme.fg("dim", prefix) + styled_msg
            left_width = visible_width(left_part)
            spacing = max(1, width - left_width - visible_width(right_part))
            styled_right = theme.fg("error" if is_confirming_delete else "dim", right_part)

            line = left_part + " " * spacing + styled_right
            if is_selected:
                line = theme.bg("selectedBg", line)
            lines.append(truncate_to_width(line, width))

        # Add scroll indicator if needed
        if start_index > 0 or end_index < len(self._filtered_sessions):
            scroll_text = f"  ({self._selected_index + 1}/{len(self._filtered_sessions)})"
            lines.append(theme.fg("muted", truncate_to_width(scroll_text, width, "")))

        return lines

    def _build_tree_prefix(self, node: dict) -> str:
        if node["depth"] == 0:
            return ""

        parts = ["│  " if continues else "   " for continues in node["ancestorContinues"]]
        branch = "└─ " if node["isLast"] else "├─ "
        return "".join(parts) + branch

    async def handle_input(self, key_data: str) -> None:
        kb = get_keybindings()

        # Handle delete confirmation state first - intercept all keys
        if self._confirming_delete_path is not None:
            if kb.matches(key_data, "tui.select.confirm"):
                path_to_delete = self._confirming_delete_path
                self._set_confirming_delete_path(None)
                if self.on_delete_session is not None:
                    tonio.spawn.without_tracking(self.on_delete_session(path_to_delete))
                return
            if kb.matches(key_data, "tui.select.cancel"):
                self._set_confirming_delete_path(None)
                return
            # Ignore all other keys while confirming
            return

        if kb.matches(key_data, "tui.input.tab"):
            if self.on_toggle_scope is not None:
                # Sync or coroutine-returning, like the other selector callbacks.
                result = self.on_toggle_scope()
                if isinstance(result, Awaitable):
                    await result
            return

        if kb.matches(key_data, "app.session.toggleSort"):
            if self.on_toggle_sort is not None:
                self.on_toggle_sort()
            return

        if self._keybindings.matches(key_data, "app.session.toggleNamedFilter"):
            if self.on_toggle_name_filter is not None:
                self.on_toggle_name_filter()
            return

        # Ctrl+P: toggle path display
        if kb.matches(key_data, "app.session.togglePath"):
            self._show_path = not self._show_path
            if self.on_toggle_path is not None:
                self.on_toggle_path(self._show_path)
            return

        # Ctrl+D: initiate delete confirmation (useful on terminals that
        # don't distinguish Ctrl+Backspace from Backspace)
        if kb.matches(key_data, "app.session.delete"):
            self._start_delete_confirmation_for_selected_session()
            return

        # Rename selected session
        if kb.matches(key_data, "app.session.rename"):
            if 0 <= self._selected_index < len(self._filtered_sessions) and self.on_rename_session is not None:
                self.on_rename_session(self._filtered_sessions[self._selected_index]["session"].path)
            return

        # Ctrl+Backspace: non-invasive convenience alias for delete. Only
        # triggers deletion when the query is empty; otherwise it is
        # forwarded to the input
        if kb.matches(key_data, "app.session.deleteNoninvasive"):
            if len(self._search_input.get_value()) > 0:
                await self._search_input.handle_input(key_data)
                self._filter_sessions(self._search_input.get_value())
                return

            self._start_delete_confirmation_for_selected_session()
            return

        # Up arrow
        if kb.matches(key_data, "tui.select.up"):
            self._selected_index = max(0, self._selected_index - 1)
        # Down arrow
        elif kb.matches(key_data, "tui.select.down"):
            self._selected_index = min(len(self._filtered_sessions) - 1, self._selected_index + 1)
        # Page up - jump up by max_visible items
        elif kb.matches(key_data, "tui.select.pageUp"):
            self._selected_index = max(0, self._selected_index - self._max_visible)
        # Page down - jump down by max_visible items
        elif kb.matches(key_data, "tui.select.pageDown"):
            self._selected_index = min(len(self._filtered_sessions) - 1, self._selected_index + self._max_visible)
        # Enter
        elif kb.matches(key_data, "tui.select.confirm"):
            if 0 <= self._selected_index < len(self._filtered_sessions) and self.on_select is not None:
                self.on_select(self._filtered_sessions[self._selected_index]["session"].path)
        # Escape - cancel
        elif kb.matches(key_data, "tui.select.cancel"):
            if self.on_cancel is not None:
                self.on_cancel()
        # Pass everything else to search input
        else:
            await self._search_input.handle_input(key_data)
            self._filter_sessions(self._search_input.get_value())


async def delete_session_file(session_path: str) -> dict:
    """Delete a session file: `trash` CLI first, then unlink fallback.

    Returns ``{"ok", "method", "error"?}``.
    """
    # Try `trash` first (if installed)
    trash_args = ["--", session_path] if session_path.startswith("-") else [session_path]

    try:
        trash_result: Any = await run_command(
            ["trash", *trash_args],  # PATH lookup, like pi's spawnSync
            capture_output=True,
        )
    except OSError as error:
        trash_result = error

    def get_trash_error_hint() -> str | None:
        parts: list = []
        if isinstance(trash_result, OSError):
            parts.append(str(trash_result))
        else:
            stderr = (trash_result.stderr or b"").decode("utf-8", "replace").strip()
            if stderr:
                parts.append(stderr.split("\n")[0])
        if not parts:
            return None
        return f"trash: {' · '.join(parts)[:200]}"

    # If trash reports success, or the file is gone afterwards, treat it as
    # successful
    trash_status = None if isinstance(trash_result, OSError) else trash_result.returncode
    if trash_status == 0 or not await fs.Path(session_path).exists():
        return {"ok": True, "method": "trash"}

    # Fallback to permanent deletion
    try:
        await tonio.spawn_blocking(os.unlink, session_path)
        return {"ok": True, "method": "unlink"}
    except OSError as err:
        trash_error_hint = get_trash_error_hint()
        error = f"{err} ({trash_error_hint})" if trash_error_hint else str(err)
        return {"ok": False, "method": "unlink", "error": error}


class SessionSelectorComponent(Container):
    """Component that renders a session selector."""

    def __init__(
        self,
        current_sessions_loader,
        all_sessions_loader,
        on_select,
        on_cancel,
        on_exit,
        request_render,
        options: dict | None = None,
        current_session_file_path: str | None = None,
    ) -> None:
        super().__init__()
        options = options or {}
        # Construction-time reads are hoisted out of constructors (PLAN: never
        # block the runtime). Both callers pass `keybindings`; the fallback is
        # the already-loaded global rather than a fresh read from disk.
        self._keybindings = options.get("keybindings") or get_keybindings()
        self._current_sessions_loader = current_sessions_loader
        self._all_sessions_loader = all_sessions_loader
        self._request_render = request_render
        self._scope = "current"
        self._sort_mode = "threaded"
        self._name_filter = "all"
        self._current_sessions: list | None = None
        self._all_sessions: list | None = None
        self._current_loading = False
        self._all_loading = False
        self._all_load_seq = 0
        self._mode = "list"
        self._rename_input = Input()
        self._rename_target_path: str | None = None
        self._focused = False

        self._header = SessionSelectorHeader(self._scope, self._sort_mode, self._name_filter, self._request_render)
        rename_session = options.get("renameSession")
        self._rename_session = rename_session
        self._can_rename = bool(rename_session)
        show_rename_hint = options.get("showRenameHint")
        self._header.set_show_rename_hint(show_rename_hint if show_rename_hint is not None else self._can_rename)

        # Create session list (starts empty, will be populated after load)
        self._session_list = SessionList(
            [],
            False,
            self._sort_mode,
            self._name_filter,
            self._keybindings,
            current_session_file_path,
        )

        self._build_base_layout(self._session_list)

        self._rename_input.on_submit = lambda value: tonio.spawn.without_tracking(self._confirm_rename(value))

        # Ensure header status timeouts are cleared when leaving the selector
        def clear_status_message() -> None:
            self._header.set_status_message(None)

        def handle_select(session_path: str) -> None:
            clear_status_message()
            on_select(session_path)

        def handle_cancel() -> None:
            clear_status_message()
            on_cancel()

        def handle_exit() -> None:
            clear_status_message()
            on_exit()

        self._session_list.on_select = handle_select
        self._session_list.on_cancel = handle_cancel
        self._session_list.on_exit = handle_exit
        self._session_list.on_toggle_scope = self._toggle_scope
        self._session_list.on_toggle_sort = self._toggle_sort_mode
        self._session_list.on_toggle_name_filter = self._toggle_name_filter

        def handle_rename(session_path: str) -> None:
            if rename_session is None:
                return
            if self._scope == "current" and self._current_loading:
                return
            if self._scope == "all" and self._all_loading:
                return

            sessions = (self._all_sessions or []) if self._scope == "all" else (self._current_sessions or [])
            session = next((s for s in sessions if s.path == session_path), None)
            self._enter_rename_mode(session_path, session.name if session is not None else None)

        self._session_list.on_rename_session = handle_rename

        # Sync list events to header
        def handle_toggle_path(show_path: bool) -> None:
            self._header.set_show_path(show_path)
            self._request_render()

        def handle_delete_confirmation_change(path: str | None) -> None:
            self._header.set_confirming_delete_path(path)
            self._request_render()

        def handle_error(msg: str) -> None:
            self._header.set_status_message({"type": "error", "message": msg}, 3000)
            self._request_render()

        self._session_list.on_toggle_path = handle_toggle_path
        self._session_list.on_delete_confirmation_change = handle_delete_confirmation_change
        self._session_list.on_error = handle_error

        # Handle session deletion
        async def handle_delete_session(session_path: str) -> None:
            result = await delete_session_file(session_path)

            if result["ok"]:
                if self._current_sessions is not None:
                    self._current_sessions = [s for s in self._current_sessions if s.path != session_path]
                if self._all_sessions is not None:
                    self._all_sessions = [s for s in self._all_sessions if s.path != session_path]

                sessions = (self._all_sessions or []) if self._scope == "all" else (self._current_sessions or [])
                show_cwd = self._scope == "all"
                await self._session_list.set_sessions(sessions, show_cwd)

                msg = "Session moved to trash" if result["method"] == "trash" else "Session deleted"
                self._header.set_status_message({"type": "info", "message": msg}, 2000)
                await self._refresh_sessions_after_mutation()
            else:
                error_message = result.get("error") or "Unknown error"
                self._header.set_status_message(
                    {"type": "error", "message": f"Failed to delete: {error_message}"}, 3000
                )

            self._request_render()

        self._session_list.on_delete_session = handle_delete_session

        # Start loading current sessions immediately
        self._load_current_sessions()

    # Focusable implementation - propagate to session list for IME cursor
    # positioning
    @property
    def focused(self) -> bool:
        return self._focused

    @focused.setter
    def focused(self, value: bool) -> None:
        self._focused = value
        self._session_list.focused = value
        self._rename_input.focused = value
        if value and self._mode == "rename":
            self._rename_input.focused = True

    async def handle_input(self, data: str) -> None:
        if self._mode == "rename":
            kb = get_keybindings()
            if kb.matches(data, "tui.select.cancel"):
                self._exit_rename_mode()
                return
            await self._rename_input.handle_input(data)
            return

        await self._session_list.handle_input(data)

    def _build_base_layout(self, content, options: dict | None = None) -> None:
        options = options or {}
        self.clear()
        self.add_child(Spacer(1))
        self.add_child(DynamicBorder(lambda s: theme.fg("accent", s)))
        self.add_child(Spacer(1))
        show_header = options.get("showHeader")
        if show_header if show_header is not None else True:
            self.add_child(self._header)
            self.add_child(Spacer(1))
        self.add_child(content)
        self.add_child(Spacer(1))
        self.add_child(DynamicBorder(lambda s: theme.fg("accent", s)))

    def _load_current_sessions(self) -> None:
        tonio.spawn.without_tracking(self._load_scope("current", "initial"))

    def _enter_rename_mode(self, session_path: str, current_name: str | None) -> None:
        self._mode = "rename"
        self._rename_target_path = session_path
        self._rename_input.set_value(current_name or "")
        self._rename_input.focused = True

        panel = Container()
        panel.add_child(Text(theme.bold("Rename Session"), 1, 0))
        panel.add_child(Spacer(1))
        panel.add_child(self._rename_input)
        panel.add_child(Spacer(1))
        panel.add_child(
            Text(
                theme.fg(
                    "muted",
                    f"{key_text('tui.select.confirm')} to save · {key_text('tui.select.cancel')} to cancel",
                ),
                1,
                0,
            )
        )

        self._build_base_layout(panel, {"showHeader": False})
        self._request_render()

    def _exit_rename_mode(self) -> None:
        self._mode = "list"
        self._rename_target_path = None

        self._build_base_layout(self._session_list)

        self._request_render()

    async def _confirm_rename(self, value: str) -> None:
        next_name = value.strip()
        if not next_name:
            return
        target = self._rename_target_path
        if not target:
            self._exit_rename_mode()
            return

        rename_session = self._rename_session
        if rename_session is None:
            self._exit_rename_mode()
            return

        try:
            await rename_session(target, next_name)
            await self._refresh_sessions_after_mutation()
        finally:
            self._exit_rename_mode()

    def _load_scope(self, scope: str, reason: str):
        """Start a scope load; returns the awaitable remainder.

        pi's async loadScope runs synchronously up to its first await, so
        the loading flags are observable immediately after the call; this
        sync prologue mirrors that before handing back the coroutine.
        """
        show_cwd = scope == "all"

        # Mark loading
        if scope == "current":
            self._current_loading = True
        else:
            self._all_loading = True

        if scope == "all":
            self._all_load_seq += 1
            seq = self._all_load_seq
        else:
            seq = None
        self._header.set_scope(scope)
        self._header.set_loading(True)
        self._request_render()

        return self._load_scope_rest(scope, reason, show_cwd, seq)

    async def _load_scope_rest(self, scope: str, reason: str, show_cwd: bool, seq: int | None) -> None:
        def on_progress(loaded: int, total: int) -> None:
            if scope != self._scope:
                return
            if seq is not None and seq != self._all_load_seq:
                return
            self._header.set_progress(loaded, total)
            self._request_render()

        try:
            if scope == "current":
                sessions = await self._current_sessions_loader(on_progress)
            else:
                sessions = await self._all_sessions_loader(on_progress)

            if scope == "current":
                self._current_sessions = sessions
                self._current_loading = False
            else:
                self._all_sessions = sessions
                self._all_loading = False

            if scope != self._scope:
                return
            if seq is not None and seq != self._all_load_seq:
                return

            self._header.set_loading(False)
            await self._session_list.set_sessions(sessions, show_cwd)
            self._request_render()
        except Exception as err:
            if scope == "current":
                self._current_loading = False
            else:
                self._all_loading = False

            if scope != self._scope:
                return
            if seq is not None and seq != self._all_load_seq:
                return

            self._header.set_loading(False)
            self._header.set_status_message({"type": "error", "message": f"Failed to load sessions: {err}"}, 4000)

            if reason == "initial":
                await self._session_list.set_sessions([], show_cwd)
            self._request_render()

    def _toggle_sort_mode(self) -> None:
        # Cycle: threaded -> recent -> relevance -> threaded
        if self._sort_mode == "threaded":
            self._sort_mode = "recent"
        elif self._sort_mode == "recent":
            self._sort_mode = "relevance"
        else:
            self._sort_mode = "threaded"
        self._header.set_sort_mode(self._sort_mode)
        self._session_list.set_sort_mode(self._sort_mode)
        self._request_render()

    def _toggle_name_filter(self) -> None:
        self._name_filter = "named" if self._name_filter == "all" else "all"
        self._header.set_name_filter(self._name_filter)
        self._session_list.set_name_filter(self._name_filter)
        self._request_render()

    async def _refresh_sessions_after_mutation(self) -> None:
        await self._load_scope(self._scope, "refresh")

    async def _toggle_scope(self) -> None:
        if self._scope == "current":
            self._scope = "all"
            self._header.set_scope(self._scope)

            if self._all_sessions is not None:
                self._header.set_loading(False)
                await self._session_list.set_sessions(self._all_sessions, True)
                self._request_render()
                return

            if not self._all_loading:
                tonio.spawn.without_tracking(self._load_scope("all", "toggle"))
            return

        self._scope = "current"
        self._header.set_scope(self._scope)
        self._header.set_loading(self._current_loading)
        await self._session_list.set_sessions(self._current_sessions or [], False)
        self._request_render()

    def get_session_list(self) -> SessionList:
        return self._session_list
