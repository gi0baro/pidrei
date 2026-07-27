"""Mirror of pi coding-agent src/modes/interactive/components/config-selector.ts.

TUI component for managing package resources (enable/disable). Package
sources are str-or-dict records like pi's PackageSource union.
"""

import os

from pidrei_tui import Container, Input, Spacer, get_keybindings, matches_key, truncate_to_width, visible_width

from ....config import CONFIG_DIR_NAME
from ....utils.paths import canonicalize_path, is_local_path, resolve_path
from ..theme import theme
from .dynamic_border import DynamicBorder
from .keybinding_hints import key_hint, raw_key_hint


RESOURCE_TYPES = ("extensions", "skills", "prompts", "themes")

RESOURCE_TYPE_LABELS = {
    "extensions": "Extensions",
    "skills": "Skills",
    "prompts": "Prompts",
    "themes": "Themes",
}


def _format_base_dir(base_dir: str) -> str:
    home_dir = os.path.expanduser("~")

    if base_dir == home_dir:
        display_path = "~"
    elif base_dir.startswith(home_dir):
        # Replace home prefix with ~
        display_path = f"~{base_dir[len(home_dir) :]}"
    else:
        display_path = base_dir

    return display_path if display_path.endswith("/") else f"{display_path}/"


def _get_group_label(metadata, agent_dir: str) -> str:
    if metadata.origin == "package":
        return f"{metadata.source} ({metadata.scope})"
    # Top-level resources
    if metadata.source == "auto":
        if metadata.base_dir:
            if metadata.scope == "user":
                return f"User ({_format_base_dir(metadata.base_dir)})"
            return f"Project ({_format_base_dir(metadata.base_dir)})"
        return f"User ({_format_base_dir(agent_dir)})" if metadata.scope == "user" else f"Project ({CONFIG_DIR_NAME}/)"
    return "User settings" if metadata.scope == "user" else "Project settings"


def build_groups(resolved, agent_dir: str) -> list:
    """Build display groups; groups/subgroups/items are camelCase dicts."""
    group_map: dict = {}

    def add_to_group(resources: list, resource_type: str) -> None:
        for res in resources:
            path = res.path
            enabled = res.enabled
            metadata = res.metadata
            group_key = f"{metadata.origin}:{metadata.scope}:{metadata.source}:{metadata.base_dir or ''}"

            if group_key not in group_map:
                group_map[group_key] = {
                    "key": group_key,
                    "label": _get_group_label(metadata, agent_dir),
                    "scope": metadata.scope,
                    "origin": metadata.origin,
                    "source": metadata.source,
                    "subgroups": [],
                }

            group = group_map[group_key]
            subgroup_key = f"{group_key}:{resource_type}"

            subgroup = next((sg for sg in group["subgroups"] if sg["type"] == resource_type), None)
            if subgroup is None:
                subgroup = {"type": resource_type, "label": RESOURCE_TYPE_LABELS[resource_type], "items": []}
                group["subgroups"].append(subgroup)

            file_name = os.path.basename(path)
            parent_folder = os.path.basename(os.path.dirname(path))
            if resource_type == "extensions" and parent_folder != "extensions":
                display_name = f"{parent_folder}/{file_name}"
            elif resource_type == "skills" and file_name == "SKILL.md":
                display_name = parent_folder
            else:
                display_name = file_name
            subgroup["items"].append(
                {
                    "path": path,
                    "enabled": enabled,
                    "metadata": metadata,
                    "resourceType": resource_type,
                    "displayName": display_name,
                    "groupKey": group_key,
                    "subgroupKey": subgroup_key,
                }
            )

    add_to_group(resolved.extensions, "extensions")
    add_to_group(resolved.skills, "skills")
    add_to_group(resolved.prompts, "prompts")
    add_to_group(resolved.themes, "themes")

    # Sort groups: packages first, then top-level; user before project
    groups = list(group_map.values())
    groups.sort(
        key=lambda g: (
            0 if g["origin"] == "package" else 1,
            0 if g["scope"] == "user" else 1,
            g["source"].lower(),
            g["source"],
        )
    )

    # Sort subgroups within each group by type order, and items by name
    type_order = {"extensions": 0, "skills": 1, "prompts": 2, "themes": 3}
    for group in groups:
        group["subgroups"].sort(key=lambda sg: type_order[sg["type"]])
        for subgroup in group["subgroups"]:
            subgroup["items"].sort(key=lambda item: (item["displayName"].lower(), item["displayName"]))

    return groups


class ConfigSelectorHeader:
    def __init__(self, write_scope: str, project_mode_available: bool) -> None:
        self._write_scope = write_scope
        self._project_mode_available = project_mode_available

    def set_write_scope(self, write_scope: str) -> None:
        self._write_scope = write_scope

    def invalidate(self) -> None:
        pass

    def render(self, width: int) -> list:
        title = theme.bold("Project Local Resources" if self._write_scope == "project" else "Global Resources")
        sep = theme.fg("muted", " · ")
        switch_hint = key_hint("tui.input.tab", "switch mode") + sep if self._project_mode_available else ""
        if self._write_scope == "project":
            action_hint = raw_key_hint("space", "cycle inherit/+/-")
        else:
            action_hint = raw_key_hint("space", "toggle")
        hint = switch_hint + action_hint + sep + raw_key_hint("esc", "close")
        spacing = max(1, width - visible_width(title) - visible_width(hint))
        if self._write_scope == "project":
            scope_hint = theme.fg("muted", f"{CONFIG_DIR_NAME}/settings.json · inherited global resources are dimmed")
        else:
            scope_hint = theme.fg("muted", f"~/{CONFIG_DIR_NAME}/agent/settings.json")

        return [
            truncate_to_width(f"{title}{' ' * spacing}{hint}", width, ""),
            truncate_to_width(scope_hint, width, ""),
        ]


class ResourceList:
    def __init__(
        self,
        groups_by_scope: dict,
        settings_manager,
        cwd: str,
        agent_dir: str,
        terminal_height: int | None = None,
        write_scope: str = "global",
    ) -> None:
        self._groups_by_scope = groups_by_scope
        self._settings_manager = settings_manager
        self._cwd = cwd
        self._agent_dir = agent_dir
        self._write_scope = write_scope
        self._inherited_enabled_by_key = self._build_inherited_enabled_map(groups_by_scope["global"])
        self._search_input = Input()
        # 8 lines of chrome: top spacer + top border + spacer + header (2
        # lines) + spacer + bottom spacer + bottom border
        chrome = 8
        self._max_visible = max(5, (terminal_height if terminal_height is not None else 24) - chrome)
        self._flat_items: list = []
        self._filtered_items: list = []
        self._selected_index = 0
        self._focused = False

        self.on_cancel = None
        self.on_exit = None
        self.on_toggle = None
        self.on_switch_mode = None

        self._build_flat_list()
        self._filtered_items = list(self._flat_items)

    # Focusable implementation - propagate to search input for IME cursor
    # positioning
    @property
    def focused(self) -> bool:
        return self._focused

    @focused.setter
    def focused(self, value: bool) -> None:
        self._focused = value
        self._search_input.focused = value

    def set_write_scope(self, write_scope: str) -> None:
        self._write_scope = write_scope
        self._build_flat_list()
        self._filter_items(self._search_input.get_value())

    @property
    def _groups(self) -> list:
        return self._groups_by_scope[self._write_scope]

    def _build_inherited_enabled_map(self, groups: list) -> dict:
        result: dict = {}
        for group in groups:
            for subgroup in group["subgroups"]:
                for item in subgroup["items"]:
                    result[self._get_resource_item_key(item)] = item["enabled"]
        return result

    def _build_flat_list(self) -> None:
        self._flat_items = []
        for group in self._groups:
            self._flat_items.append({"type": "group", "group": group})
            for subgroup in group["subgroups"]:
                self._flat_items.append({"type": "subgroup", "subgroup": subgroup, "group": group})
                for item in subgroup["items"]:
                    self._flat_items.append({"type": "item", "item": item})
        # Start selection on first item (not header)
        self._selected_index = next((i for i, e in enumerate(self._flat_items) if e["type"] == "item"), -1)
        self._selected_index = max(self._selected_index, 0)

    def _find_next_item(self, from_index: int, direction: int) -> int:
        idx = from_index + direction
        while 0 <= idx < len(self._filtered_items):
            if self._filtered_items[idx]["type"] == "item":
                return idx
            idx += direction
        return from_index  # Stay at current if no item found

    def _filter_items(self, query: str) -> None:
        if not query.strip():
            self._filtered_items = list(self._flat_items)
            self._select_first_item()
            return

        lower_query = query.lower()
        matching_items: set = set()
        matching_subgroups: set = set()
        matching_groups: set = set()

        for entry in self._flat_items:
            if entry["type"] == "item":
                item = entry["item"]
                if (
                    lower_query in item["displayName"].lower()
                    or lower_query in item["resourceType"].lower()
                    or lower_query in item["path"].lower()
                ):
                    matching_items.add(id(item))

        # Find which subgroups and groups contain matching items
        for group in self._groups:
            for subgroup in group["subgroups"]:
                for item in subgroup["items"]:
                    if id(item) in matching_items:
                        matching_subgroups.add(id(subgroup))
                        matching_groups.add(id(group))

        self._filtered_items = []
        for entry in self._flat_items:
            if (
                entry["type"] == "group"
                and id(entry["group"]) in matching_groups
                or entry["type"] == "subgroup"
                and id(entry["subgroup"]) in matching_subgroups
                or entry["type"] == "item"
                and id(entry["item"]) in matching_items
            ):
                self._filtered_items.append(entry)

        self._select_first_item()

    def _select_first_item(self) -> None:
        first_item_index = next((i for i, e in enumerate(self._filtered_items) if e["type"] == "item"), -1)
        self._selected_index = max(first_item_index, 0)

    def update_item(self, item: dict, enabled: bool) -> None:
        item["enabled"] = enabled
        # Update in groups too
        for group in self._groups:
            for subgroup in group["subgroups"]:
                found = next(
                    (
                        i
                        for i in subgroup["items"]
                        if i["path"] == item["path"] and i["resourceType"] == item["resourceType"]
                    ),
                    None,
                )
                if found is not None:
                    found["enabled"] = enabled
                    return

    def invalidate(self) -> None:
        pass

    def render(self, width: int) -> list:
        lines: list = []

        # Search input
        lines.extend(self._search_input.render(width))
        lines.append("")

        if not self._filtered_items:
            lines.append(theme.fg("muted", "  No resources found"))
            return lines

        # Calculate visible range
        start_index = max(
            0,
            min(self._selected_index - self._max_visible // 2, len(self._filtered_items) - self._max_visible),
        )
        end_index = min(start_index + self._max_visible, len(self._filtered_items))

        for i in range(start_index, end_index):
            entry = self._filtered_items[i]
            is_selected = i == self._selected_index

            if entry["type"] == "group":
                # Main group header (no cursor)
                inherited = self._write_scope == "project" and entry["group"]["scope"] == "user"
                label = theme.bold(f"{entry['group']['label']}{' · inherited global' if inherited else ''}")
                group_line = theme.fg("dim" if inherited else "accent", label)
                lines.append(truncate_to_width(f"  {group_line}", width, ""))
            elif entry["type"] == "subgroup":
                # Subgroup header (indented, no cursor)
                color = "dim" if self._write_scope == "project" and entry["group"]["scope"] == "user" else "muted"
                subgroup_line = theme.fg(color, entry["subgroup"]["label"])
                lines.append(truncate_to_width(f"    {subgroup_line}", width, ""))
            else:
                # Resource item (cursor only on items)
                item = entry["item"]
                cursor = "> " if is_selected else "  "
                dimmed = self._is_dimmed_item(item)
                name_text = theme.bold(item["displayName"]) if is_selected and not dimmed else item["displayName"]
                name = theme.fg("dim", name_text) if dimmed else name_text
                lines.append(
                    truncate_to_width(
                        f"{cursor}    {self._render_checkbox(item)} {name}{self._get_item_suffix(item)}",
                        width,
                        "...",
                    )
                )

        # Scroll indicator
        if start_index > 0 or end_index < len(self._filtered_items):
            item_count = sum(1 for e in self._filtered_items if e["type"] == "item")
            current_item_index = sum(1 for e in self._filtered_items[: self._selected_index] if e["type"] == "item") + 1
            lines.append(theme.fg("dim", f"  ({current_item_index}/{item_count})"))

        return lines

    async def handle_input(self, data: str) -> None:
        kb = get_keybindings()

        if kb.matches(data, "tui.select.up"):
            self._selected_index = self._find_next_item(self._selected_index, -1)
            return
        if kb.matches(data, "tui.select.down"):
            self._selected_index = self._find_next_item(self._selected_index, 1)
            return
        if kb.matches(data, "tui.select.pageUp"):
            # Jump up by max_visible, then find nearest item
            target = max(0, self._selected_index - self._max_visible)
            while target < len(self._filtered_items) and self._filtered_items[target]["type"] != "item":
                target += 1
            if target < len(self._filtered_items):
                self._selected_index = target
            return
        if kb.matches(data, "tui.select.pageDown"):
            # Jump down by max_visible, then find nearest item
            target = min(len(self._filtered_items) - 1, self._selected_index + self._max_visible)
            while target >= 0 and self._filtered_items[target]["type"] != "item":
                target -= 1
            if target >= 0:
                self._selected_index = target
            return
        if kb.matches(data, "tui.select.cancel"):
            if self.on_cancel is not None:
                self.on_cancel()
            return
        if matches_key(data, "ctrl+c"):
            if self.on_exit is not None:
                self.on_exit()
            return
        if kb.matches(data, "tui.input.tab"):
            if self.on_switch_mode is not None:
                self.on_switch_mode()
            return
        if data == " " or kb.matches(data, "tui.select.confirm"):
            if 0 <= self._selected_index < len(self._filtered_items):
                entry = self._filtered_items[self._selected_index]
                if entry["type"] == "item" and (
                    self._write_scope == "project" or self._get_item_scope(entry["item"]) == "user"
                ):
                    new_enabled = self._toggle_resource(entry["item"])
                    if new_enabled is not None:
                        self.update_item(entry["item"], new_enabled)
                        if self.on_toggle is not None:
                            self.on_toggle(entry["item"], new_enabled)
            return

        # Pass to search input
        await self._search_input.handle_input(data)
        self._filter_items(self._search_input.get_value())

    def _toggle_resource(self, item: dict) -> bool | None:
        if self._write_scope == "project":
            state = self._get_next_override_state(item)
            if not self._set_project_resource_override(item, state):
                return None
            if state == "inherit":
                return self._get_inherited_enabled(item)
            return state == "load"

        enabled = not item["enabled"]
        if item["metadata"].origin == "top-level":
            self._toggle_top_level_resource(item, enabled)
        else:
            self._toggle_package_resource(item, enabled)
        return enabled

    def _toggle_top_level_resource(self, item: dict, enabled: bool) -> None:
        scope = item["metadata"].scope
        settings = (
            self._settings_manager.get_project_settings()
            if scope == "project"
            else self._settings_manager.get_global_settings()
        )

        array_key = item["resourceType"]
        current = list(settings.get(array_key) or [])

        # Generate pattern for this resource
        pattern = self._get_resource_pattern(item)
        disable_pattern = f"-{pattern}"
        enable_pattern = f"+{pattern}"

        # Filter out existing patterns for this resource
        updated = [p for p in current if self._get_pattern_entry_target(p) != pattern]

        updated.append(enable_pattern if enabled else disable_pattern)

        if scope == "project":
            if array_key == "extensions":
                self._settings_manager.set_project_extension_paths(updated)
            elif array_key == "skills":
                self._settings_manager.set_project_skill_paths(updated)
            elif array_key == "prompts":
                self._settings_manager.set_project_prompt_template_paths(updated)
            elif array_key == "themes":
                self._settings_manager.set_project_theme_paths(updated)
        elif array_key == "extensions":
            self._settings_manager.set_extension_paths(updated)
        elif array_key == "skills":
            self._settings_manager.set_skill_paths(updated)
        elif array_key == "prompts":
            self._settings_manager.set_prompt_template_paths(updated)
        elif array_key == "themes":
            self._settings_manager.set_theme_paths(updated)

    def _toggle_package_resource(self, item: dict, enabled: bool) -> None:
        scope = item["metadata"].scope
        settings = (
            self._settings_manager.get_project_settings()
            if scope == "project"
            else self._settings_manager.get_global_settings()
        )

        packages = list(settings.get("packages") or [])
        pkg_index = next(
            (
                i
                for i, pkg in enumerate(packages)
                if (pkg if isinstance(pkg, str) else pkg.get("source")) == item["metadata"].source
            ),
            -1,
        )

        if pkg_index == -1:
            return

        pkg = packages[pkg_index]

        # Convert string to object form if needed
        if isinstance(pkg, str):
            pkg = {"source": pkg}
            packages[pkg_index] = pkg

        # Get the resource array for this type
        array_key = item["resourceType"]
        current = list(pkg.get(array_key) or [])

        # Generate pattern relative to package root
        pattern = self._get_package_resource_pattern(item)
        disable_pattern = f"-{pattern}"
        enable_pattern = f"+{pattern}"

        # Filter out existing patterns for this resource
        updated = [p for p in current if self._get_pattern_entry_target(p) != pattern]

        updated.append(enable_pattern if enabled else disable_pattern)

        if updated:
            pkg[array_key] = updated
        else:
            pkg.pop(array_key, None)

        # Clean up empty filter object
        has_filters = any(pkg.get(k) is not None for k in RESOURCE_TYPES)
        if not has_filters:
            packages[pkg_index] = pkg["source"]

        if scope == "project":
            self._settings_manager.set_project_packages(packages)
        else:
            self._settings_manager.set_packages(packages)

    def _render_checkbox(self, item: dict) -> str:
        if self._write_scope == "project":
            state = self._get_project_override_state(item)
            if state == "load":
                return theme.fg("success", "[+]")
            if state == "unload":
                return theme.fg("warning", "[-]")
            return theme.fg("dim", "[x]" if item["enabled"] else "[ ]")
        return theme.fg("success", "[x]") if item["enabled"] else theme.fg("dim", "[ ]")

    def _get_item_suffix(self, item: dict) -> str:
        if self._write_scope != "project":
            return ""
        state = self._get_project_override_state(item)
        if state == "load":
            return theme.fg("muted", "  project load")
        if state == "unload":
            return theme.fg("muted", "  project unload")
        return theme.fg("dim", "  inherited global") if self._is_inherited_global_item(item) else ""

    def _is_dimmed_item(self, item: dict) -> bool:
        return (
            self._write_scope == "project"
            and self._is_inherited_global_item(item)
            and self._get_project_override_state(item) == "inherit"
        )

    def _set_project_resource_override(self, item: dict, state: str) -> bool:
        if item["metadata"].origin == "top-level":
            return self._set_project_top_level_override(item, state)
        return self._set_project_package_override(item, state)

    def _set_project_top_level_override(self, item: dict, state: str) -> bool:
        current = list(self._settings_manager.get_project_settings().get(item["resourceType"]) or [])
        if self._is_inherited_global_item(item):
            pattern = item["path"]
        else:
            pattern = self._get_resource_pattern_for_scope(item, "project")
        patterns = self._get_top_level_override_patterns(item, "project")
        updated = []
        for entry in current:
            target = self._get_pattern_entry_target(entry)
            if entry.startswith(("!", "+", "-")) and target in patterns:
                continue
            if state == "inherit" and self._is_inherited_global_item(item) and target == pattern:
                continue
            updated.append(entry)
        if state != "inherit":
            if self._is_inherited_global_item(item) and pattern not in updated:
                updated.append(pattern)
            updated.append(f"{'+' if state == 'load' else '-'}{pattern}")
        self._set_project_top_level_paths(item["resourceType"], updated)
        return True

    def _set_project_top_level_paths(self, key: str, paths: list) -> None:
        if key == "extensions":
            self._settings_manager.set_project_extension_paths(paths)
        elif key == "skills":
            self._settings_manager.set_project_skill_paths(paths)
        elif key == "prompts":
            self._settings_manager.set_project_prompt_template_paths(paths)
        else:
            self._settings_manager.set_project_theme_paths(paths)

    def _set_project_package_override(self, item: dict, state: str) -> bool:
        packages = list(self._settings_manager.get_project_settings().get("packages") or [])
        pkg_index = next(
            (
                i
                for i, pkg in enumerate(packages)
                if self._package_source_string_matches(
                    item["metadata"].source,
                    self._get_item_scope(item),
                    pkg if isinstance(pkg, str) else pkg.get("source"),
                    "project",
                )
            ),
            -1,
        )
        if pkg_index == -1:
            if state == "inherit":
                return False
            packages.append(self._create_package_override_source(item))
            pkg_index = len(packages) - 1
        pkg = packages[pkg_index]
        if pkg is None:
            return False
        if isinstance(pkg, str):
            pkg = {"source": pkg}
            packages[pkg_index] = pkg
        pattern = self._get_package_resource_pattern(item)
        updated = [
            entry for entry in (pkg.get(item["resourceType"]) or []) if self._get_pattern_entry_target(entry) != pattern
        ]
        if state != "inherit":
            updated.append(f"{'+' if state == 'load' else '-'}{pattern}")
        if updated:
            pkg[item["resourceType"]] = updated
        else:
            pkg.pop(item["resourceType"], None)
        if not any(pkg.get(key) is not None for key in RESOURCE_TYPES):
            if pkg.get("autoload") is False:
                packages.pop(pkg_index)
            else:
                packages[pkg_index] = pkg["source"]
        self._settings_manager.set_project_packages(packages)
        return True

    def _get_next_override_state(self, item: dict) -> str:
        state = self._get_project_override_state(item)
        inherited_enabled = self._get_inherited_enabled(item)
        if state == "inherit":
            return "unload" if inherited_enabled else "load"
        if state == "unload":
            return "load" if inherited_enabled else "inherit"
        return "inherit" if inherited_enabled else "unload"

    def _get_project_override_state(self, item: dict) -> str:
        if self._write_scope != "project":
            return "inherit"
        if item["metadata"].origin == "top-level":
            return self._get_override_state_from_entries(
                list(self._settings_manager.get_project_settings().get(item["resourceType"]) or []),
                self._get_top_level_override_patterns(item, "project"),
                False,
            )
        pkg = self._find_matching_package_source(item, "project")
        if not isinstance(pkg, dict):
            return "inherit"
        entries = pkg.get(item["resourceType"])
        if entries is None:
            return "inherit"
        return self._get_override_state_from_entries(
            entries,
            {self._get_package_resource_pattern(item)},
            pkg.get("autoload") is not False,
        )

    def _get_override_state_from_entries(self, entries: list, patterns: set, empty_array_is_unload: bool) -> str:
        if not entries and empty_array_is_unload:
            return "unload"
        state = "inherit"
        for entry in entries:
            if self._get_pattern_entry_target(entry) not in patterns:
                continue
            state = "unload" if entry.startswith(("!", "-")) else "load"
        return state

    def _get_inherited_enabled(self, item: dict) -> bool:
        key = self._get_resource_item_key(item)
        if key in self._inherited_enabled_by_key:
            return self._inherited_enabled_by_key[key]
        return item["enabled"] if self._get_item_scope(item) == "user" else True

    def _is_inherited_global_item(self, item: dict) -> bool:
        return (
            self._get_item_scope(item) == "user" or self._get_resource_item_key(item) in self._inherited_enabled_by_key
        )

    def _get_top_level_override_patterns(self, item: dict, scope: str) -> set:
        base_dir = self._get_top_level_base_dir(scope)
        patterns = {
            self._get_resource_pattern_for_scope(item, scope),
            item["path"],
            os.path.relpath(item["path"], base_dir),
        }
        if item["metadata"].base_dir:
            patterns.add(os.path.relpath(item["path"], item["metadata"].base_dir))
        return patterns

    def _get_resource_pattern_for_scope(self, item: dict, scope: str) -> str:
        source_scope = self._get_item_scope(item)
        if scope != source_scope:
            return item["path"]
        base_dir = item["metadata"].base_dir or self._get_top_level_base_dir(source_scope)
        return os.path.relpath(item["path"], base_dir)

    def _create_package_override_source(self, item: dict):
        source = item["metadata"].source
        if not is_local_path(source):
            return {"source": source, "autoload": False}
        source_path = resolve_path(source, self._get_top_level_base_dir(self._get_item_scope(item)), trim=True)
        relative = os.path.relpath(source_path, self._get_top_level_base_dir("project"))
        if relative == ".":
            relative = ""
        return {"source": relative or ".", "autoload": False}

    def _package_source_string_matches(
        self, left_source: str, left_scope: str, right_source: str, right_scope: str
    ) -> bool:
        if left_source == right_source:
            return True
        if not is_local_path(left_source) or not is_local_path(right_source):
            return False
        left = resolve_path(left_source, self._get_top_level_base_dir(left_scope), trim=True)
        right = resolve_path(right_source, self._get_top_level_base_dir(right_scope), trim=True)
        return left == right

    def _find_matching_package_source(self, item: dict, target_scope: str):
        settings = (
            self._settings_manager.get_project_settings()
            if target_scope == "project"
            else self._settings_manager.get_global_settings()
        )
        return next(
            (
                pkg
                for pkg in settings.get("packages") or []
                if self._package_source_string_matches(
                    item["metadata"].source,
                    self._get_item_scope(item),
                    pkg if isinstance(pkg, str) else pkg.get("source"),
                    target_scope,
                )
            ),
            None,
        )

    def _get_pattern_entry_target(self, entry: str) -> str:
        return entry[1:] if entry.startswith(("!", "+", "-")) else entry

    def _get_resource_item_key(self, item: dict) -> str:
        return f"{item['resourceType']}:{canonicalize_path(item['path'])}"

    def _get_item_scope(self, item: dict) -> str:
        return "project" if item["metadata"].scope == "project" else "user"

    def _get_top_level_base_dir(self, scope: str) -> str:
        return os.path.join(self._cwd, CONFIG_DIR_NAME) if scope == "project" else self._agent_dir

    def _get_resource_pattern(self, item: dict) -> str:
        scope = item["metadata"].scope
        base_dir = item["metadata"].base_dir or self._get_top_level_base_dir(scope)
        return os.path.relpath(item["path"], base_dir)

    def _get_package_resource_pattern(self, item: dict) -> str:
        base_dir = item["metadata"].base_dir or os.path.dirname(item["path"])
        return os.path.relpath(item["path"], base_dir)


class ConfigSelectorComponent(Container):
    def __init__(
        self,
        resolved_paths: dict,
        settings_manager,
        cwd: str,
        agent_dir: str,
        on_close,
        on_exit,
        request_render,
        terminal_height: int | None = None,
        write_scope: str = "global",
        project_mode_available: bool = True,
    ) -> None:
        super().__init__()

        self._write_scope = write_scope
        self._focused = False
        groups_by_scope = {
            "global": build_groups(resolved_paths["global"], agent_dir),
            "project": build_groups(resolved_paths["project"], agent_dir),
        }

        # Add header
        self.add_child(Spacer(1))
        self.add_child(DynamicBorder())
        self.add_child(Spacer(1))
        self._header = ConfigSelectorHeader(self._write_scope, project_mode_available)
        self.add_child(self._header)
        self.add_child(Spacer(1))

        # Resource list
        self._resource_list = ResourceList(
            groups_by_scope,
            settings_manager,
            cwd,
            agent_dir,
            terminal_height,
            self._write_scope,
        )
        self._resource_list.on_cancel = on_close
        self._resource_list.on_exit = on_exit
        self._resource_list.on_toggle = lambda _item, _enabled: request_render()
        if project_mode_available:

            def switch_mode() -> None:
                self._switch_write_scope()
                request_render()

            self._resource_list.on_switch_mode = switch_mode
        self.add_child(self._resource_list)

        # Bottom border
        self.add_child(Spacer(1))
        self.add_child(DynamicBorder())

    # Focusable implementation
    @property
    def focused(self) -> bool:
        return self._focused

    @focused.setter
    def focused(self, value: bool) -> None:
        self._focused = value
        self._resource_list.focused = value

    def _switch_write_scope(self) -> None:
        self._write_scope = "project" if self._write_scope == "global" else "global"
        self._header.set_write_scope(self._write_scope)
        self._resource_list.set_write_scope(self._write_scope)

    def get_resource_list(self) -> ResourceList:
        return self._resource_list
