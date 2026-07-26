"""Mirror of pi coding-agent src/modes/interactive/components/tree-selector.ts.

Session tree entries are camelCase dicts; ``entry["message"]`` values are
pidrei_ai/pidrei_agent message dataclasses.
"""

import json
import os
import re
from datetime import datetime

from pidrei_tui import (
    Container,
    Input,
    Spacer,
    Text,
    get_keybindings,
    slice_by_column,
    truncate_to_width,
    visible_width,
    wrap_text_with_ansi,
)
from pidrei_tui._timers import Timeout

from ..theme import theme
from .dynamic_border import DynamicBorder
from .keybinding_hints import format_key_text, key_hint


TREE_GUTTER_WIDTH = 2
MIN_VISIBLE_ANCHOR_CONTENT_WIDTH = 4
MAX_VISIBLE_ANCHOR_CONTENT_WIDTH = 20
MIN_ANCHOR_CONTEXT_WIDTH = 2
MAX_ANCHOR_CONTEXT_WIDTH = 12

_NORMALIZE_RE = re.compile(r"[\n\t]")

FILTER_MODES = ["default", "no-tools", "user-only", "labeled-only", "all"]


def _render_horizontal_viewport(rows: list, width: int) -> list:
    """Render tree rows into a horizontally clipped viewport.

    The tree gutter is always kept visible. The row bodies are shifted left
    only when the selected row's anchor (the start of its entry text after
    tree indentation/markers) would otherwise be too far right to see useful
    content. Rows are ``{"gutter", "body", "anchorCol", "bodyWidth",
    "isSelected"}`` records.
    """
    viewport_width = max(0, width - TREE_GUTTER_WIDTH)
    max_body_width = max((row["bodyWidth"] for row in rows), default=0)
    max_horizontal_scroll = max(0, max_body_width - viewport_width)
    selected_row = next((row for row in rows if row["isSelected"]), None)

    # Only pan horizontally when needed to keep enough selected-row content
    # visible after its anchor.
    horizontal_scroll = 0
    if selected_row is not None and max_horizontal_scroll > 0:
        min_visible_anchor_content_width = min(
            MAX_VISIBLE_ANCHOR_CONTENT_WIDTH,
            max(MIN_VISIBLE_ANCHOR_CONTENT_WIDTH, viewport_width // 3),
        )
        if selected_row["anchorCol"] > viewport_width - min_visible_anchor_content_width:
            anchor_context_width = min(
                MAX_ANCHOR_CONTEXT_WIDTH,
                max(MIN_ANCHOR_CONTEXT_WIDTH, viewport_width // 4),
            )
            horizontal_scroll = min(max_horizontal_scroll, selected_row["anchorCol"] - anchor_context_width)

    # Clip only the body; the fixed-width gutter remains visible as
    # navigation context.
    lines = []
    for row in rows:
        if horizontal_scroll > 0:
            line = f"{row['gutter']}{slice_by_column(row['body'], horizontal_scroll, viewport_width, True)}\x1b[0m"
        else:
            line = row["gutter"] + row["body"]
        lines.append(truncate_to_width(line, width, ""))
    return lines


def _extract_full_content(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    result = ""
    for block in content:
        block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if block_type == "text":
            result += block.get("text") if isinstance(block, dict) else getattr(block, "text", "")
    return result


def _has_text_content(content) -> bool:
    if isinstance(content, str):
        return len(content.strip()) > 0
    if isinstance(content, list):
        for c in content:
            c_type = c.get("type") if isinstance(c, dict) else getattr(c, "type", None)
            if c_type == "text":
                text = c.get("text") if isinstance(c, dict) else getattr(c, "text", None)
                if text and text.strip():
                    return True
    return False


class TreeList:
    """Tree list component with selection and ASCII art visualization."""

    def __init__(
        self,
        tree: list,
        current_leaf_id: str | None,
        max_visible_lines: int,
        initial_selected_id: str | None = None,
        initial_filter_mode: str | None = None,
    ) -> None:
        self._current_leaf_id = current_leaf_id
        self._max_visible_lines = max_visible_lines
        self._filter_mode = initial_filter_mode if initial_filter_mode is not None else "default"
        self._search_query = ""
        self._tool_call_map: dict = {}
        self._multiple_roots = len(tree) > 1
        self._show_label_timestamps = False
        self._active_path_ids: set = set()
        self._visible_parent_map: dict = {}
        self._visible_children_map: dict = {}
        self._last_selected_id: str | None = None
        self._folded_nodes: set = set()
        self._selected_index = 0
        self._filtered_nodes: list = []

        self.on_select = None
        self.on_cancel = None
        self.on_copy = None
        self.on_label_edit = None

        self._flat_nodes = self._flatten_tree(tree)
        self._build_active_path()
        self._apply_filter()

        # Start with initial_selected_id if provided, otherwise current leaf
        target_id = initial_selected_id if initial_selected_id is not None else current_leaf_id
        self._selected_index = self._find_nearest_visible_index(target_id)
        if self._filtered_nodes:
            self._last_selected_id = self._filtered_nodes[self._selected_index]["node"].entry["id"]

    def _find_nearest_visible_index(self, entry_id: str | None) -> int:
        """Nearest visible entry index, walking up the parent chain if needed."""
        if not self._filtered_nodes:
            return 0

        # Build a map for parent lookup
        entry_map = {flat_node["node"].entry["id"]: flat_node for flat_node in self._flat_nodes}

        # Build a map of visible entry IDs to their indices in filtered nodes
        visible_id_to_index = {node["node"].entry["id"]: i for i, node in enumerate(self._filtered_nodes)}

        # Walk from entry_id up to root, looking for a visible entry
        current_id = entry_id
        while current_id is not None:
            if current_id in visible_id_to_index:
                return visible_id_to_index[current_id]
            node = entry_map.get(current_id)
            if node is None:
                break
            current_id = node["node"].entry.get("parentId")

        # Fallback: last visible entry
        return len(self._filtered_nodes) - 1

    def _build_active_path(self) -> None:
        """Build the set of entry IDs on the path from root to current leaf."""
        self._active_path_ids.clear()
        if not self._current_leaf_id:
            return

        # Build a map of id -> entry for parent lookup
        entry_map = {flat_node["node"].entry["id"]: flat_node for flat_node in self._flat_nodes}

        # Walk from leaf to root
        current_id = self._current_leaf_id
        while current_id:
            self._active_path_ids.add(current_id)
            node = entry_map.get(current_id)
            if node is None:
                break
            current_id = node["node"].entry.get("parentId")

    def _flatten_tree(self, roots: list) -> list:
        result: list = []
        self._tool_call_map.clear()

        # Indentation rules:
        # - At indent 0: stay at 0 unless parent has >1 children (then +1)
        # - At indent 1: children always go to indent 2 (visual grouping)
        # - At indent 2+: stay flat for single-child chains, +1 only if
        #   parent branches

        # Determine which subtrees contain the active leaf (to sort current
        # branch first). Iterative post-order traversal to avoid recursion
        # limits.
        contains_active: dict = {}
        leaf_id = self._current_leaf_id
        all_nodes: list = []
        pre_order_stack = list(roots)
        while pre_order_stack:
            node = pre_order_stack.pop()
            all_nodes.append(node)
            # Push children in reverse so they're processed left-to-right
            pre_order_stack.extend(reversed(node.children))
        # Process in reverse (post-order): children before parents
        for node in reversed(all_nodes):
            has = leaf_id is not None and node.entry["id"] == leaf_id
            for child in node.children:
                if contains_active.get(id(child)):
                    has = True
            contains_active[id(node)] = has

        # Add roots in reverse order, prioritizing the one containing the
        # active leaf. If multiple roots, treat them as children of a virtual
        # root that branches.
        multiple_roots = len(roots) > 1
        ordered_roots = sorted(roots, key=lambda n: -int(bool(contains_active.get(id(n)))))
        stack: list = []
        for i in range(len(ordered_roots) - 1, -1, -1):
            is_last = i == len(ordered_roots) - 1
            stack.append(
                (
                    ordered_roots[i],
                    1 if multiple_roots else 0,
                    multiple_roots,
                    multiple_roots,
                    is_last,
                    [],
                    multiple_roots,
                )
            )

        while stack:
            node, indent, just_branched, show_connector, is_last, gutters, is_virtual_root_child = stack.pop()

            # Extract tool calls from assistant messages for later lookup
            entry = node.entry
            message = entry.get("message")
            if entry.get("type") == "message" and getattr(message, "role", None) == "assistant":
                content = getattr(message, "content", None)
                if isinstance(content, list):
                    for block in content:
                        block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
                        if block_type == "toolCall":
                            tc_id = block.get("id") if isinstance(block, dict) else block.id
                            tc_name = block.get("name") if isinstance(block, dict) else block.name
                            tc_args = block.get("arguments") if isinstance(block, dict) else block.arguments
                            self._tool_call_map[tc_id] = {"name": tc_name, "arguments": tc_args}

            result.append(
                {
                    "node": node,
                    "indent": indent,
                    "showConnector": show_connector,
                    "isLast": is_last,
                    "gutters": gutters,
                    "isVirtualRootChild": is_virtual_root_child,
                }
            )

            children = node.children
            multiple_children = len(children) > 1

            # Order children so the branch containing the active leaf comes
            # first
            prioritized = [child for child in children if contains_active.get(id(child))]
            rest = [child for child in children if not contains_active.get(id(child))]
            ordered_children = [*prioritized, *rest]

            # Calculate child indent
            if multiple_children:
                # Parent branches: children get +1
                child_indent = indent + 1
            elif just_branched and indent > 0:
                # First generation after a branch: +1 for visual grouping
                child_indent = indent + 1
            else:
                # Single-child chain: stay flat
                child_indent = indent

            # Build gutters for children. If this node showed a connector,
            # add a gutter entry for descendants (only when the connector is
            # actually displayed, i.e. not suppressed for virtual root
            # children).
            connector_displayed = show_connector and not is_virtual_root_child
            current_display_indent = max(0, indent - 1) if self._multiple_roots else indent
            connector_position = max(0, current_display_indent - 1)
            child_gutters = (
                [*gutters, {"position": connector_position, "show": not is_last}] if connector_displayed else gutters
            )

            # Add children in reverse order
            for i in range(len(ordered_children) - 1, -1, -1):
                child_is_last = i == len(ordered_children) - 1
                stack.append(
                    (
                        ordered_children[i],
                        child_indent,
                        multiple_children,
                        multiple_children,
                        child_is_last,
                        child_gutters,
                        False,
                    )
                )

        return result

    def _apply_filter(self) -> None:
        # Update last_selected_id only when we have a valid selection
        # (non-empty list). This preserves the selection when switching
        # through empty filter results
        if self._filtered_nodes and 0 <= self._selected_index < len(self._filtered_nodes):
            self._last_selected_id = self._filtered_nodes[self._selected_index]["node"].entry["id"]

        search_tokens = [t for t in self._search_query.lower().split() if t]

        def passes(flat_node: dict) -> bool:
            entry = flat_node["node"].entry
            is_current_leaf = entry["id"] == self._current_leaf_id
            message = entry.get("message")

            # Skip assistant messages with only tool calls (no text) unless
            # error/aborted. Always show current leaf so active position is
            # visible
            if entry.get("type") == "message" and getattr(message, "role", None) == "assistant" and not is_current_leaf:
                has_text = _has_text_content(getattr(message, "content", None))
                stop_reason = getattr(message, "stop_reason", None)
                is_error_or_aborted = stop_reason and stop_reason not in ("stop", "toolUse")
                # Only hide if no text AND not an error/aborted message
                if not has_text and not is_error_or_aborted:
                    return False

            # Apply filter mode
            # Entry types hidden in default view (settings/bookkeeping)
            is_settings_entry = entry.get("type") in (
                "label",
                "custom",
                "model_change",
                "thinking_level_change",
                "session_info",
            )

            if self._filter_mode == "user-only":
                # Just user messages
                passes_filter = entry.get("type") == "message" and getattr(message, "role", None) == "user"
            elif self._filter_mode == "no-tools":
                # Default minus tool results
                passes_filter = not is_settings_entry and not (
                    entry.get("type") == "message" and getattr(message, "role", None) == "toolResult"
                )
            elif self._filter_mode == "labeled-only":
                # Just labeled entries
                passes_filter = flat_node["node"].label is not None
            elif self._filter_mode == "all":
                # Show everything
                passes_filter = True
            else:
                # Default mode: hide settings/bookkeeping entries
                passes_filter = not is_settings_entry

            if not passes_filter:
                return False

            # Apply search filter
            if search_tokens:
                node_text = self._get_searchable_text(flat_node["node"]).lower()
                return all(token in node_text for token in search_tokens)

            return True

        self._filtered_nodes = [flat_node for flat_node in self._flat_nodes if passes(flat_node)]

        # Filter out descendants of folded nodes.
        if self._folded_nodes:
            skip_set: set = set()
            for flat_node in self._flat_nodes:
                entry_id = flat_node["node"].entry["id"]
                parent_id = flat_node["node"].entry.get("parentId")
                if parent_id is not None and (parent_id in self._folded_nodes or parent_id in skip_set):
                    skip_set.add(entry_id)
            self._filtered_nodes = [
                flat_node for flat_node in self._filtered_nodes if flat_node["node"].entry["id"] not in skip_set
            ]

        # Recalculate visual structure (indent, connectors, gutters) based on
        # visible tree
        self._recalculate_visual_structure()

        # Try to preserve cursor on the same node, or find nearest visible
        # ancestor
        if self._last_selected_id:
            self._selected_index = self._find_nearest_visible_index(self._last_selected_id)
        elif self._selected_index >= len(self._filtered_nodes):
            # Clamp index if out of bounds
            self._selected_index = max(0, len(self._filtered_nodes) - 1)

        # Update last_selected_id to the actual selection (may have changed
        # due to parent walk)
        if self._filtered_nodes:
            self._last_selected_id = self._filtered_nodes[self._selected_index]["node"].entry["id"]

    def _recalculate_visual_structure(self) -> None:
        """Recompute indentation/connectors for the filtered view.

        Filtering can hide intermediate entries; descendants attach to the
        nearest visible ancestor. Keeps indentation semantics aligned with
        _flatten_tree() so single-child chains don't drift right.
        """
        if not self._filtered_nodes:
            return

        visible_ids = {n["node"].entry["id"] for n in self._filtered_nodes}

        # Build entry map for efficient parent lookup (using full tree)
        entry_map = {flat_node["node"].entry["id"]: flat_node for flat_node in self._flat_nodes}

        def find_visible_ancestor(node_id: str) -> str | None:
            flat = entry_map.get(node_id)
            current_id = flat["node"].entry.get("parentId") if flat is not None else None
            while current_id is not None:
                if current_id in visible_ids:
                    return current_id
                flat = entry_map.get(current_id)
                current_id = flat["node"].entry.get("parentId") if flat is not None else None
            return None

        # Build visible tree structure:
        # - visible_parent: node_id → nearest visible ancestor (None = root)
        # - visible_children: parent_id → list of visible children (in order)
        visible_parent: dict = {}
        visible_children: dict = {None: []}

        for flat_node in self._filtered_nodes:
            node_id = flat_node["node"].entry["id"]
            ancestor_id = find_visible_ancestor(node_id)
            visible_parent[node_id] = ancestor_id

            visible_children.setdefault(ancestor_id, []).append(node_id)

        # Update multiple_roots based on visible roots
        visible_root_ids = visible_children[None]
        self._multiple_roots = len(visible_root_ids) > 1

        # Build a map for quick lookup: node_id → flat node
        filtered_node_map = {flat_node["node"].entry["id"]: flat_node for flat_node in self._filtered_nodes}

        # DFS over the visible tree using _flatten_tree() indentation
        # semantics
        stack: list = []

        # Add visible roots in reverse order (to process forward via stack)
        for i in range(len(visible_root_ids) - 1, -1, -1):
            is_last = i == len(visible_root_ids) - 1
            stack.append(
                (
                    visible_root_ids[i],
                    1 if self._multiple_roots else 0,
                    self._multiple_roots,
                    self._multiple_roots,
                    is_last,
                    [],
                    self._multiple_roots,
                )
            )

        while stack:
            node_id, indent, just_branched, show_connector, is_last, gutters, is_virtual_root_child = stack.pop()

            flat_node = filtered_node_map.get(node_id)
            if flat_node is None:
                continue

            # Update this node's visual properties
            flat_node["indent"] = indent
            flat_node["showConnector"] = show_connector
            flat_node["isLast"] = is_last
            flat_node["gutters"] = gutters
            flat_node["isVirtualRootChild"] = is_virtual_root_child

            # Get visible children of this node
            children = visible_children.get(node_id, [])
            multiple_children = len(children) > 1

            # Child indent follows _flatten_tree(): branch points (and first
            # generation after a branch) shift +1
            if multiple_children or just_branched and indent > 0:
                child_indent = indent + 1
            else:
                child_indent = indent

            # Child gutters follow _flatten_tree() connector/gutter rules
            connector_displayed = show_connector and not is_virtual_root_child
            current_display_indent = max(0, indent - 1) if self._multiple_roots else indent
            connector_position = max(0, current_display_indent - 1)
            child_gutters = (
                [*gutters, {"position": connector_position, "show": not is_last}] if connector_displayed else gutters
            )

            # Add children in reverse order (to process forward via stack)
            for i in range(len(children) - 1, -1, -1):
                child_is_last = i == len(children) - 1
                stack.append(
                    (
                        children[i],
                        child_indent,
                        multiple_children,
                        multiple_children,
                        child_is_last,
                        child_gutters,
                        False,
                    )
                )

        # Store visible tree maps for ancestor/descendant lookups in
        # navigation
        self._visible_parent_map = visible_parent
        self._visible_children_map = visible_children

    def _get_searchable_text(self, node) -> str:
        """Get searchable text content from a node."""
        entry = node.entry
        parts: list = []

        if node.label:
            parts.append(node.label)

        entry_type = entry.get("type")
        if entry_type == "message":
            msg = entry["message"]
            parts.append(msg.role)
            content = getattr(msg, "content", None)
            if content:
                parts.append(self._extract_content(content))
            if msg.role == "bashExecution":
                command = getattr(msg, "command", None)
                if command:
                    parts.append(command)
        elif entry_type == "custom_message":
            parts.append(entry.get("customType", ""))
            content = entry.get("content")
            if isinstance(content, str):
                parts.append(content)
            else:
                parts.append(self._extract_content(content))
        elif entry_type == "compaction":
            parts.append("compaction")
        elif entry_type == "branch_summary":
            parts.extend(["branch summary", entry.get("summary", "")])
        elif entry_type == "session_info":
            parts.append("title")
            if entry.get("name"):
                parts.append(entry["name"])
        elif entry_type == "model_change":
            parts.extend(["model", entry.get("modelId", "")])
        elif entry_type == "thinking_level_change":
            parts.extend(["thinking", entry.get("thinkingLevel", "")])
        elif entry_type == "custom":
            parts.extend(["custom", entry.get("customType", "")])
        elif entry_type == "label":
            parts.extend(["label", entry.get("label") or ""])

        return " ".join(parts)

    def invalidate(self) -> None:
        pass

    def get_search_query(self) -> str:
        return self._search_query

    def get_selected_node(self):
        if 0 <= self._selected_index < len(self._filtered_nodes):
            return self._filtered_nodes[self._selected_index]["node"]
        return None

    def copy_selected(self) -> None:
        node = self.get_selected_node()
        if self.on_copy is not None:
            self.on_copy(self._get_entry_copy_text(node) if node is not None else None)

    def update_node_label(self, entry_id: str, label: str | None, label_timestamp: str | None = None) -> None:
        for flat_node in self._flat_nodes:
            if flat_node["node"].entry["id"] == entry_id:
                flat_node["node"].label = label
                if label:
                    flat_node["node"].label_timestamp = (
                        label_timestamp if label_timestamp is not None else datetime.now().astimezone().isoformat()
                    )
                else:
                    flat_node["node"].label_timestamp = None
                break

    def _get_status_labels(self) -> str:
        labels = ""
        if self._filter_mode == "no-tools":
            labels += " [no-tools]"
        elif self._filter_mode == "user-only":
            labels += " [user]"
        elif self._filter_mode == "labeled-only":
            labels += " [labeled]"
        elif self._filter_mode == "all":
            labels += " [all]"
        if self._show_label_timestamps:
            labels += " [+label time]"
        return labels

    def render(self, width: int) -> list:
        lines: list = []

        if not self._filtered_nodes:
            lines.append(truncate_to_width(theme.fg("muted", "  No entries found"), width))
            lines.append(truncate_to_width(theme.fg("muted", f"  (0/0){self._get_status_labels()}"), width))
            return lines

        start_index = max(
            0,
            min(
                self._selected_index - self._max_visible_lines // 2,
                len(self._filtered_nodes) - self._max_visible_lines,
            ),
        )
        end_index = min(start_index + self._max_visible_lines, len(self._filtered_nodes))

        rendered_rows: list = []
        for i in range(start_index, end_index):
            flat_node = self._filtered_nodes[i]
            entry = flat_node["node"].entry
            is_selected = i == self._selected_index

            # Build line: cursor + prefix + path marker + label + content
            cursor = theme.fg("accent", "› ") if is_selected else "  "

            # If multiple roots, shift display (roots at 0, not 1)
            display_indent = max(0, flat_node["indent"] - 1) if self._multiple_roots else flat_node["indent"]

            # Build prefix with gutters at their correct positions. Each
            # gutter has a position (display indent where its connector was
            # shown)
            if flat_node["showConnector"] and not flat_node["isVirtualRootChild"]:
                connector = "└─ " if flat_node["isLast"] else "├─ "
            else:
                connector = ""
            connector_position = display_indent - 1 if connector else -1

            # Build prefix char by char, placing gutters and connector at
            # their positions
            total_chars = display_indent * 3
            prefix_chars: list = []
            is_folded = entry["id"] in self._folded_nodes
            for char_i in range(total_chars):
                level = char_i // 3
                pos_in_level = char_i % 3

                # Check if there's a gutter at this level
                gutter = next((g for g in flat_node["gutters"] if g["position"] == level), None)
                if gutter is not None:
                    if pos_in_level == 0:
                        prefix_chars.append("│" if gutter["show"] else " ")
                    else:
                        prefix_chars.append(" ")
                elif connector and level == connector_position:
                    # Connector at this level, with fold indicator
                    if pos_in_level == 0:
                        prefix_chars.append("└" if flat_node["isLast"] else "├")
                    elif pos_in_level == 1:
                        foldable = self._is_foldable(entry["id"])
                        prefix_chars.append("⊞" if is_folded else ("⊟" if foldable else "─"))
                    else:
                        prefix_chars.append(" ")
                else:
                    prefix_chars.append(" ")
            prefix = "".join(prefix_chars)

            # Fold marker for nodes without connectors (roots)
            shows_fold_in_connector = flat_node["showConnector"] and not flat_node["isVirtualRootChild"]
            fold_marker = theme.fg("accent", "⊞ ") if is_folded and not shows_fold_in_connector else ""

            # Active path marker - shown right before the entry text
            is_on_active_path = entry["id"] in self._active_path_ids
            path_marker = theme.fg("accent", "• ") if is_on_active_path else ""

            label = theme.fg("warning", f"[{flat_node['node'].label}] ") if flat_node["node"].label else ""
            if self._show_label_timestamps and flat_node["node"].label and flat_node["node"].label_timestamp:
                label_timestamp = theme.fg(
                    "muted", f"{self._format_label_timestamp(flat_node['node'].label_timestamp)} "
                )
            else:
                label_timestamp = ""
            content = self._get_entry_display_text(flat_node["node"], is_selected)
            prefix_part = theme.fg("dim", prefix) + fold_marker + path_marker
            anchor_col = visible_width(prefix_part)
            gutter_text = cursor
            body = prefix_part + label + label_timestamp + content
            if is_selected:
                gutter_text = theme.bg("selectedBg", gutter_text)
                body = theme.bg("selectedBg", body)
            rendered_rows.append(
                {
                    "gutter": gutter_text,
                    "body": body,
                    "anchorCol": anchor_col,
                    "bodyWidth": visible_width(body),
                    "isSelected": is_selected,
                }
            )

        lines.extend(_render_horizontal_viewport(rendered_rows, width))
        lines.append(
            truncate_to_width(
                theme.fg(
                    "muted",
                    f"  ({self._selected_index + 1}/{len(self._filtered_nodes)}){self._get_status_labels()}",
                ),
                width,
            )
        )

        return lines

    def _get_entry_display_text(self, node, is_selected: bool) -> str:
        entry = node.entry

        def normalize(s: str) -> str:
            return _NORMALIZE_RE.sub(" ", s).strip()

        entry_type = entry.get("type")
        if entry_type == "message":
            msg = entry["message"]
            role = msg.role
            if role == "user":
                content = normalize(self._extract_content(getattr(msg, "content", None)))
                result = theme.fg("accent", "user: ") + content
            elif role == "assistant":
                text_content = normalize(self._extract_content(getattr(msg, "content", None)))
                if text_content:
                    result = theme.fg("success", "assistant: ") + text_content
                elif getattr(msg, "stop_reason", None) == "aborted":
                    result = theme.fg("success", "assistant: ") + theme.fg("muted", "(aborted)")
                elif getattr(msg, "error_message", None):
                    err_msg = normalize(msg.error_message)[:80]
                    result = theme.fg("success", "assistant: ") + theme.fg("error", err_msg)
                else:
                    result = theme.fg("success", "assistant: ") + theme.fg("muted", "(no content)")
            elif role == "toolResult":
                tool_call_id = getattr(msg, "tool_call_id", None)
                tool_call = self._tool_call_map.get(tool_call_id) if tool_call_id else None
                if tool_call is not None:
                    result = theme.fg("muted", self._format_tool_call(tool_call["name"], tool_call["arguments"]))
                else:
                    result = theme.fg("muted", f"[{getattr(msg, 'tool_name', None) or 'tool'}]")
            elif role == "bashExecution":
                result = theme.fg("dim", f"[bash]: {normalize(getattr(msg, 'command', None) or '')}")
            else:
                result = theme.fg("dim", f"[{role}]")
        elif entry_type == "custom_message":
            raw_content = entry.get("content")
            if isinstance(raw_content, str):
                content = raw_content
            else:
                content = "".join(
                    (c.get("text") if isinstance(c, dict) else getattr(c, "text", ""))
                    for c in raw_content or []
                    if (c.get("type") if isinstance(c, dict) else getattr(c, "type", None)) == "text"
                )
            result = theme.fg("customMessageLabel", f"[{entry.get('customType')}]: ") + normalize(content)
        elif entry_type == "compaction":
            tokens = round(entry.get("tokensBefore", 0) / 1000)
            result = theme.fg("borderAccent", f"[compaction: {tokens}k tokens]")
        elif entry_type == "branch_summary":
            result = theme.fg("warning", "[branch summary]: ") + normalize(entry.get("summary", ""))
        elif entry_type == "model_change":
            result = theme.fg("dim", f"[model: {entry.get('modelId')}]")
        elif entry_type == "thinking_level_change":
            result = theme.fg("dim", f"[thinking: {entry.get('thinkingLevel')}]")
        elif entry_type == "custom":
            result = theme.fg("dim", f"[custom: {entry.get('customType')}]")
        elif entry_type == "label":
            result = theme.fg("dim", f"[label: {entry.get('label') or '(cleared)'}]")
        elif entry_type == "session_info":
            if entry.get("name"):
                result = theme.fg("dim", "[title: ") + theme.fg("dim", entry["name"]) + theme.fg("dim", "]")
            else:
                result = theme.fg("dim", "[title: ") + theme.italic(theme.fg("dim", "empty")) + theme.fg("dim", "]")
        else:
            result = ""

        return theme.bold(result) if is_selected else result

    def _format_label_timestamp(self, timestamp: str) -> str:
        date = datetime.fromisoformat(timestamp)
        if date.tzinfo is not None:
            date = date.astimezone()
        now = datetime.now().astimezone()
        time_text = f"{date.hour:02d}:{date.minute:02d}"

        if date.year == now.year and date.month == now.month and date.day == now.day:
            return time_text

        if date.year == now.year:
            return f"{date.month}/{date.day} {time_text}"

        return f"{str(date.year)[-2:]}/{date.month}/{date.day} {time_text}"

    def _extract_content(self, content) -> str:
        return _extract_full_content(content)[:200]

    def _get_entry_copy_text(self, node) -> str | None:
        entry = node.entry
        text: str | None = None

        entry_type = entry.get("type")
        if entry_type == "message":
            msg = entry["message"]
            if msg.role == "bashExecution":
                text = msg.command
            elif hasattr(msg, "content"):
                text = _extract_full_content(msg.content)
                if not text and msg.role == "assistant":
                    text = getattr(msg, "error_message", None)
        elif entry_type == "custom_message":
            text = _extract_full_content(entry.get("content"))
        elif entry_type in ("compaction", "branch_summary"):
            text = entry.get("summary")

        return text if text and text.strip() else None

    def _format_tool_call(self, name: str, args: dict) -> str:
        def shorten(p: str) -> str:
            home = os.environ.get("HOME") or ""
            if home and p.startswith(home):
                return f"~{p[len(home) :]}"
            return p

        args = args or {}
        if name == "read":
            path = shorten(str(args.get("path") or args.get("file_path") or ""))
            offset = args.get("offset")
            limit = args.get("limit")
            display = path
            if offset is not None or limit is not None:
                start = offset if offset is not None else 1
                end = start + limit - 1 if limit is not None else ""
                display += f":{start}{f'-{end}' if end else ''}"
            return f"[read: {display}]"
        if name == "write":
            path = shorten(str(args.get("path") or args.get("file_path") or ""))
            return f"[write: {path}]"
        if name == "edit":
            path = shorten(str(args.get("path") or args.get("file_path") or ""))
            return f"[edit: {path}]"
        if name == "bash":
            raw_cmd = str(args.get("command") or "")
            cmd = _NORMALIZE_RE.sub(" ", raw_cmd).strip()[:50]
            return f"[bash: {cmd}{'...' if len(raw_cmd) > 50 else ''}]"
        if name == "grep":
            pattern = str(args.get("pattern") or "")
            path = shorten(str(args.get("path") or "."))
            return f"[grep: /{pattern}/ in {path}]"
        if name == "find":
            pattern = str(args.get("pattern") or "")
            path = shorten(str(args.get("path") or "."))
            return f"[find: {pattern} in {path}]"
        if name == "ls":
            path = shorten(str(args.get("path") or "."))
            return f"[ls: {path}]"
        # Custom tool - show name and truncated JSON args
        args_json = json.dumps(args, separators=(",", ":"))
        args_str = args_json[:40]
        return f"[{name}: {args_str}{'...' if len(args_json) > 40 else ''}]"

    def handle_input(self, key_data: str) -> None:
        kb = get_keybindings()
        if kb.matches(key_data, "tui.select.up"):
            self._selected_index = (
                len(self._filtered_nodes) - 1 if self._selected_index == 0 else self._selected_index - 1
            )
        elif kb.matches(key_data, "tui.select.down"):
            self._selected_index = (
                0 if self._selected_index == len(self._filtered_nodes) - 1 else self._selected_index + 1
            )
        elif kb.matches(key_data, "app.tree.foldOrUp"):
            current_id = (
                self._filtered_nodes[self._selected_index]["node"].entry["id"]
                if 0 <= self._selected_index < len(self._filtered_nodes)
                else None
            )
            if current_id and self._is_foldable(current_id) and current_id not in self._folded_nodes:
                self._folded_nodes.add(current_id)
                self._apply_filter()
            else:
                self._selected_index = self._find_branch_segment_start("up")
        elif kb.matches(key_data, "app.tree.unfoldOrDown"):
            current_id = (
                self._filtered_nodes[self._selected_index]["node"].entry["id"]
                if 0 <= self._selected_index < len(self._filtered_nodes)
                else None
            )
            if current_id and current_id in self._folded_nodes:
                self._folded_nodes.discard(current_id)
                self._apply_filter()
            else:
                self._selected_index = self._find_branch_segment_start("down")
        elif kb.matches(key_data, "tui.editor.cursorLeft") or kb.matches(key_data, "tui.select.pageUp"):
            # Page up
            self._selected_index = max(0, self._selected_index - self._max_visible_lines)
        elif kb.matches(key_data, "tui.editor.cursorRight") or kb.matches(key_data, "tui.select.pageDown"):
            # Page down
            self._selected_index = min(len(self._filtered_nodes) - 1, self._selected_index + self._max_visible_lines)
        elif kb.matches(key_data, "tui.select.confirm"):
            if 0 <= self._selected_index < len(self._filtered_nodes) and self.on_select is not None:
                self.on_select(self._filtered_nodes[self._selected_index]["node"].entry["id"])
        elif kb.matches(key_data, "app.message.copy"):
            self.copy_selected()
        elif kb.matches(key_data, "tui.select.cancel"):
            if self._search_query:
                self._search_query = ""
                self._folded_nodes.clear()
                self._apply_filter()
            elif self.on_cancel is not None:
                self.on_cancel()
        elif kb.matches(key_data, "app.tree.filter.default"):
            # Direct filter: default
            self._filter_mode = "default"
            self._folded_nodes.clear()
            self._apply_filter()
        elif kb.matches(key_data, "app.tree.filter.noTools"):
            # Toggle filter: no-tools ↔ default
            self._filter_mode = "default" if self._filter_mode == "no-tools" else "no-tools"
            self._folded_nodes.clear()
            self._apply_filter()
        elif kb.matches(key_data, "app.tree.filter.userOnly"):
            # Toggle filter: user-only ↔ default
            self._filter_mode = "default" if self._filter_mode == "user-only" else "user-only"
            self._folded_nodes.clear()
            self._apply_filter()
        elif kb.matches(key_data, "app.tree.filter.labeledOnly"):
            # Toggle filter: labeled-only ↔ default
            self._filter_mode = "default" if self._filter_mode == "labeled-only" else "labeled-only"
            self._folded_nodes.clear()
            self._apply_filter()
        elif kb.matches(key_data, "app.tree.filter.all"):
            # Toggle filter: all ↔ default
            self._filter_mode = "default" if self._filter_mode == "all" else "all"
            self._folded_nodes.clear()
            self._apply_filter()
        elif kb.matches(key_data, "app.tree.filter.cycleBackward"):
            # Cycle filter backwards
            current_index = FILTER_MODES.index(self._filter_mode)
            self._filter_mode = FILTER_MODES[(current_index - 1 + len(FILTER_MODES)) % len(FILTER_MODES)]
            self._folded_nodes.clear()
            self._apply_filter()
        elif kb.matches(key_data, "app.tree.filter.cycleForward"):
            # Cycle filter forwards: default → no-tools → user-only →
            # labeled-only → all → default
            current_index = FILTER_MODES.index(self._filter_mode)
            self._filter_mode = FILTER_MODES[(current_index + 1) % len(FILTER_MODES)]
            self._folded_nodes.clear()
            self._apply_filter()
        elif kb.matches(key_data, "tui.editor.deleteCharBackward"):
            if len(self._search_query) > 0:
                self._search_query = self._search_query[:-1]
                self._folded_nodes.clear()
                self._apply_filter()
        elif kb.matches(key_data, "app.tree.editLabel"):
            if 0 <= self._selected_index < len(self._filtered_nodes) and self.on_label_edit is not None:
                selected = self._filtered_nodes[self._selected_index]
                self.on_label_edit(selected["node"].entry["id"], selected["node"].label)
        elif kb.matches(key_data, "app.tree.toggleLabelTimestamp"):
            self._show_label_timestamps = not self._show_label_timestamps
        else:
            has_control_chars = any(ord(ch) < 32 or ord(ch) == 0x7F or 0x80 <= ord(ch) <= 0x9F for ch in key_data)
            if not has_control_chars and len(key_data) > 0:
                self._search_query += key_data
                self._folded_nodes.clear()
                self._apply_filter()

    def _is_foldable(self, entry_id: str) -> bool:
        """Whether a node can be folded.

        A node is foldable if it has visible children and is either a root
        (no visible parent) or a segment start (visible parent has multiple
        visible children).
        """
        children = self._visible_children_map.get(entry_id)
        if not children:
            return False
        parent_id = self._visible_parent_map.get(entry_id)
        if parent_id is None:
            return True
        siblings = self._visible_children_map.get(parent_id)
        return siblings is not None and len(siblings) > 1

    def _find_branch_segment_start(self, direction: str) -> int:
        """Index of the next branch segment start in the given direction.

        A segment start is the first child of a branch point. "up" walks the
        visible parent chain; "down" walks visible children (always
        following the first child).
        """
        if not (0 <= self._selected_index < len(self._filtered_nodes)):
            return self._selected_index
        selected_id = self._filtered_nodes[self._selected_index]["node"].entry["id"]

        index_by_entry_id = {node["node"].entry["id"]: i for i, node in enumerate(self._filtered_nodes)}
        current_id = selected_id
        if direction == "down":
            while True:
                children = self._visible_children_map.get(current_id, [])
                if not children:
                    return index_by_entry_id[current_id]
                if len(children) > 1:
                    return index_by_entry_id[children[0]]
                current_id = children[0]

        # direction == "up"
        while True:
            parent_id = self._visible_parent_map.get(current_id)
            if parent_id is None:
                return index_by_entry_id[current_id]
            children = self._visible_children_map.get(parent_id, [])
            if len(children) > 1:
                segment_start = index_by_entry_id[current_id]
                if segment_start < self._selected_index:
                    return segment_start
            current_id = parent_id


class SearchLine:
    """Component that displays the current search query."""

    def __init__(self, tree_list: TreeList) -> None:
        self._tree_list = tree_list

    def invalidate(self) -> None:
        pass

    def render(self, width: int) -> list:
        query = self._tree_list.get_search_query()
        if query:
            return [truncate_to_width(f"  {theme.fg('muted', 'Type to search:')} {theme.fg('accent', query)}", width)]
        return [truncate_to_width(f"  {theme.fg('muted', 'Type to search:')}", width)]

    def handle_input(self, _key_data: str) -> None:
        pass


TREE_HELP_ITEMS = [
    {"keys": ["tui.select.up", "tui.select.down"], "label": "move"},
    {"keys": ["tui.editor.cursorLeft", "tui.editor.cursorRight"], "label": "page"},
    {"keys": ["app.tree.foldOrUp", "app.tree.unfoldOrDown"], "label": "branch"},
    {"keys": ["app.message.copy"], "label": "copy"},
    {"keys": ["app.tree.editLabel"], "label": "label"},
    {"keys": ["app.tree.toggleLabelTimestamp"], "label": "label time"},
    {
        "keys": [
            "app.tree.filter.default",
            "app.tree.filter.noTools",
            "app.tree.filter.userOnly",
            "app.tree.filter.labeledOnly",
            "app.tree.filter.all",
        ],
        "label": "filters",
        "labelFirst": True,
    },
    {"keys": ["app.tree.filter.cycleForward", "app.tree.filter.cycleBackward"], "label": "cycle", "labelFirst": True},
]


def _compact_raw_keys(keys: list) -> str:
    if len(keys) == 1:
        return keys[0]

    parts = []
    for key in keys:
        separator_index = key.rfind("+")
        if separator_index == -1:
            parts.append({"prefix": "", "suffix": key})
        else:
            parts.append({"prefix": key[: separator_index + 1], "suffix": key[separator_index + 1 :]})
    prefix = parts[0]["prefix"]
    if prefix and all(part["prefix"] == prefix for part in parts):
        return prefix + "/".join(part["suffix"] for part in parts)
    return "/".join(keys)


def _format_help_keys(keybindings: list) -> str:
    keys: list = []
    for keybinding in keybindings:
        binding_keys = get_keybindings().get_keys(keybinding)
        if binding_keys:
            keys.append(binding_keys[0])
    if not keys:
        return ""

    text = format_key_text(_compact_raw_keys(keys))
    text = re.sub(r"\bpageUp\b", "pgup", text)
    text = re.sub(r"\bpageDown\b", "pgdn", text)
    text = re.sub(r"\bup\b", "↑", text)
    text = re.sub(r"\bdown\b", "↓", text)
    text = re.sub(r"\bleft\b", "←", text)
    text = re.sub(r"\bright\b", "→", text)
    return text


class TreeHelp:
    """Renders tree help as semantic rows with chunk-aware wrapping."""

    def invalidate(self) -> None:
        pass

    def render(self, width: int) -> list:
        items = []
        for entry in TREE_HELP_ITEMS:
            text = _format_help_keys(entry["keys"])
            if not text:
                items.append(entry["label"])
            elif entry.get("labelFirst"):
                items.append(f"{entry['label']} {text}")
            else:
                items.append(f"{text} {entry['label']}")

        available_width = max(1, width)
        indent = "  "
        separator = " · "
        lines: list = []
        current_line = ""

        for item in items:
            if current_line:
                candidate = f"{current_line}{separator}{item}"
            elif visible_width(f"{indent}{item}") <= available_width:
                candidate = f"{indent}{item}"
            else:
                candidate = item
            if not current_line or visible_width(candidate) <= available_width:
                current_line = candidate
                continue

            lines.extend(wrap_text_with_ansi(current_line.rstrip(), available_width))
            current_line = f"{indent}{item}" if visible_width(f"{indent}{item}") <= available_width else item

        if current_line:
            lines.extend(wrap_text_with_ansi(current_line.rstrip(), available_width))

        return [theme.fg("muted", line) for line in lines]


class LabelInput:
    """Label input component shown when editing a label."""

    def __init__(self, entry_id: str, current_label: str | None) -> None:
        self._entry_id = entry_id
        self._input = Input()
        self._focused = False
        if current_label:
            self._input.set_value(current_label)
        self.on_submit = None
        self.on_cancel = None

    # Focusable implementation - propagate to input for IME cursor
    # positioning
    @property
    def focused(self) -> bool:
        return self._focused

    @focused.setter
    def focused(self, value: bool) -> None:
        self._focused = value
        self._input.focused = value

    def invalidate(self) -> None:
        pass

    def render(self, width: int) -> list:
        lines: list = []
        indent = "  "
        available_width = width - len(indent)
        lines.append(truncate_to_width(f"{indent}{theme.fg('muted', 'Label (empty to remove):')}", width))
        lines.extend(truncate_to_width(f"{indent}{line}", width) for line in self._input.render(available_width))
        lines.append(
            truncate_to_width(
                f"{indent}{key_hint('tui.select.confirm', 'save')}  {key_hint('tui.select.cancel', 'cancel')}",
                width,
            )
        )
        return lines

    def handle_input(self, key_data: str) -> None:
        kb = get_keybindings()
        if kb.matches(key_data, "tui.select.confirm"):
            value = self._input.get_value().strip()
            if self.on_submit is not None:
                self.on_submit(self._entry_id, value or None)
        elif kb.matches(key_data, "tui.select.cancel"):
            if self.on_cancel is not None:
                self.on_cancel()
        else:
            self._input.handle_input(key_data)


class TreeSelectorComponent(Container):
    """Component that renders a session tree selector for navigation."""

    def __init__(
        self,
        tree: list,
        current_leaf_id: str | None,
        terminal_height: int,
        on_select,
        on_cancel,
        on_label_change=None,
        initial_selected_id: str | None = None,
        initial_filter_mode: str | None = None,
    ) -> None:
        super().__init__()

        self._on_label_change_callback = on_label_change
        self._label_input: LabelInput | None = None
        self._focused = False
        max_visible_lines = max(5, terminal_height // 2)

        self._tree_list = TreeList(tree, current_leaf_id, max_visible_lines, initial_selected_id, initial_filter_mode)
        self._tree_list.on_select = on_select
        self._tree_list.on_cancel = on_cancel

        def on_copy(text) -> None:
            if self.on_copy is not None:
                self.on_copy(text)

        self._tree_list.on_copy = on_copy
        self._tree_list.on_label_edit = self._show_label_input
        self.on_copy = None

        self._tree_container = Container()
        self._tree_container.add_child(self._tree_list)

        self._label_input_container = Container()

        self.add_child(Spacer(1))
        self.add_child(DynamicBorder())
        self.add_child(Text(theme.bold("  Session Tree"), 1, 0))
        self.add_child(TreeHelp())
        self.add_child(SearchLine(self._tree_list))
        self.add_child(DynamicBorder())
        self.add_child(Spacer(1))
        self.add_child(self._tree_container)
        self.add_child(self._label_input_container)
        self.add_child(Spacer(1))
        self.add_child(DynamicBorder())

        if not tree:
            Timeout(100, lambda: on_cancel())

    # Focusable implementation - propagate to label input when active for
    # IME cursor positioning
    @property
    def focused(self) -> bool:
        return self._focused

    @focused.setter
    def focused(self, value: bool) -> None:
        self._focused = value
        # Propagate to label input when it's active
        if self._label_input is not None:
            self._label_input.focused = value

    def _show_label_input(self, entry_id: str, current_label: str | None) -> None:
        self._label_input = LabelInput(entry_id, current_label)

        def on_submit(entry_id: str, label: str | None) -> None:
            self._tree_list.update_node_label(entry_id, label)
            if self._on_label_change_callback is not None:
                self._on_label_change_callback(entry_id, label)
            self._hide_label_input()

        self._label_input.on_submit = on_submit
        self._label_input.on_cancel = lambda: self._hide_label_input()

        # Propagate current focused state to the new label input
        self._label_input.focused = self._focused

        self._tree_container.clear()
        self._label_input_container.clear()
        self._label_input_container.add_child(self._label_input)

    def _hide_label_input(self) -> None:
        self._label_input = None
        self._label_input_container.clear()
        self._tree_container.clear()
        self._tree_container.add_child(self._tree_list)

    def handle_input(self, key_data: str) -> None:
        if self._label_input is not None:
            self._label_input.handle_input(key_data)
        else:
            self._tree_list.handle_input(key_data)

    def get_tree_list(self) -> TreeList:
        return self._tree_list
