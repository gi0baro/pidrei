"""Mirror of pi coding-agent src/modes/interactive/components/scoped-models-selector.ts.

EnabledIds: None = all enabled (no filter), list = explicit ordered list.
"""

from pidrei_tui import Container, Input, Key, Spacer, Text, fuzzy_filter, get_keybindings, matches_key

from ..model_search import get_model_search_text
from ..theme import theme
from .dynamic_border import DynamicBorder
from .keybinding_hints import key_text


def _is_enabled(enabled_ids, model_id: str) -> bool:
    return enabled_ids is None or model_id in enabled_ids


def _toggle(enabled_ids, model_id: str):
    if enabled_ids is None:
        return [model_id]  # First toggle: start with only this one
    if model_id in enabled_ids:
        return [entry for entry in enabled_ids if entry != model_id]
    return [*enabled_ids, model_id]


def _enable_all(enabled_ids, all_ids: list, target_ids: list | None = None):
    if enabled_ids is None:
        return None  # Already all enabled
    targets = target_ids if target_ids is not None else all_ids
    result = list(enabled_ids)
    for model_id in targets:
        if model_id not in result:
            result.append(model_id)
    return None if len(result) == len(all_ids) else result


def _clear_all(enabled_ids, all_ids: list, target_ids: list | None = None):
    if enabled_ids is None:
        if target_ids is not None:
            return [model_id for model_id in all_ids if model_id not in target_ids]
        return []
    targets = set(target_ids if target_ids is not None else enabled_ids)
    return [model_id for model_id in enabled_ids if model_id not in targets]


def _move(enabled_ids, model_id: str, delta: int):
    if enabled_ids is None:
        return None
    result = list(enabled_ids)
    if model_id not in result:
        return result
    index = result.index(model_id)
    new_index = index + delta
    if new_index < 0 or new_index >= len(result):
        return result
    result[index], result[new_index] = result[new_index], result[index]
    return result


def _get_sorted_ids(enabled_ids, all_ids: list) -> list:
    if enabled_ids is None:
        return all_ids
    enabled_set = set(enabled_ids)
    return [*enabled_ids, *(model_id for model_id in all_ids if model_id not in enabled_set)]


class ScopedModelsSelectorComponent(Container):
    """Component for enabling/disabling models for Ctrl+P cycling.

    Changes are session-only until explicitly persisted with Ctrl+S.
    ``config`` is ``{"allModels", "enabledModelIds"}``; ``callbacks`` is
    ``{"onChange", "onPersist", "onCancel"}``.
    """

    def __init__(self, config: dict, callbacks: dict) -> None:
        super().__init__()
        self._callbacks = callbacks
        self._models_by_id: dict = {}
        self._all_ids: list = []
        self._max_visible = 8
        self._is_dirty = False
        self._selected_index = 0
        self._focused = False

        for model in config["allModels"]:
            full_id = f"{model.provider}/{model.id}"
            self._models_by_id[full_id] = model
            self._all_ids.append(full_id)

        enabled_model_ids = config["enabledModelIds"]
        self._enabled_ids = None if enabled_model_ids is None else list(enabled_model_ids)
        self._filtered_items = self._build_items()

        # Header
        self.add_child(DynamicBorder())
        self.add_child(Spacer(1))
        self.add_child(Text(theme.fg("accent", theme.bold("Model Configuration")), 0, 0))
        self.add_child(
            Text(theme.fg("muted", f"Session-only. {key_text('app.models.save')} to save to settings."), 0, 0)
        )
        self.add_child(Spacer(1))

        # Search input
        self._search_input = Input()
        self.add_child(self._search_input)
        self.add_child(Spacer(1))

        # List container
        self._list_container = Container()
        self.add_child(self._list_container)

        # Footer hint
        self.add_child(Spacer(1))
        self._footer_text = Text(self._get_footer_text(), 0, 0)
        self.add_child(self._footer_text)

        self.add_child(DynamicBorder())
        self._update_list()

    # Focusable implementation - propagate to search input for IME cursor
    # positioning
    @property
    def focused(self) -> bool:
        return self._focused

    @focused.setter
    def focused(self, value: bool) -> None:
        self._focused = value
        self._search_input.focused = value

    def _build_items(self) -> list:
        # Filter out IDs that no longer have a corresponding model (e.g.,
        # after logout)
        return [
            {
                "fullId": model_id,
                "model": self._models_by_id[model_id],
                "enabled": _is_enabled(self._enabled_ids, model_id),
            }
            for model_id in _get_sorted_ids(self._enabled_ids, self._all_ids)
            if model_id in self._models_by_id
        ]

    def _get_footer_text(self) -> str:
        enabled_count = len(self._enabled_ids) if self._enabled_ids is not None else len(self._all_ids)
        all_enabled = self._enabled_ids is None
        count_text = "all enabled" if all_enabled else f"{enabled_count}/{len(self._all_ids)} enabled"
        parts = [
            f"{key_text('tui.select.confirm')} toggle",
            f"{key_text('app.models.enableAll')} all",
            f"{key_text('app.models.clearAll')} clear",
            f"{key_text('app.models.toggleProvider')} provider",
            f"{key_text('app.models.reorderUp')}/{key_text('app.models.reorderDown')} reorder",
            f"{key_text('app.models.save')} save",
            count_text,
        ]
        if self._is_dirty:
            return theme.fg("dim", f"  {' · '.join(parts)} ") + theme.fg("warning", "(unsaved)")
        return theme.fg("dim", f"  {' · '.join(parts)}")

    def _refresh(self) -> None:
        query = self._search_input.get_value()
        items = self._build_items()
        if query:
            self._filtered_items = fuzzy_filter(
                items,
                query,
                lambda i: get_model_search_text(
                    {"id": i["model"].id, "provider": i["model"].provider, "name": i["model"].name}
                ),
            )
        else:
            self._filtered_items = items
        self._selected_index = min(self._selected_index, max(0, len(self._filtered_items) - 1))
        self._update_list()
        self._footer_text.set_text(self._get_footer_text())

    def _notify_change(self) -> None:
        self._callbacks["onChange"](None if self._enabled_ids is None else list(self._enabled_ids))

    def _update_list(self) -> None:
        self._list_container.clear()

        if not self._filtered_items:
            self._list_container.add_child(Text(theme.fg("muted", "  No matching models"), 0, 0))
            return

        start_index = max(
            0,
            min(self._selected_index - self._max_visible // 2, len(self._filtered_items) - self._max_visible),
        )
        end_index = min(start_index + self._max_visible, len(self._filtered_items))
        all_enabled = self._enabled_ids is None

        for i in range(start_index, end_index):
            item = self._filtered_items[i]
            is_selected = i == self._selected_index
            prefix = theme.fg("accent", "→ ") if is_selected else "  "
            model_text = theme.fg("accent", item["model"].id) if is_selected else item["model"].id
            provider_badge = theme.fg("muted", f" [{item['model'].provider}]")
            if all_enabled:
                status = ""
            elif item["enabled"]:
                status = theme.fg("success", " ✓")
            else:
                status = theme.fg("dim", " ✗")
            self._list_container.add_child(Text(f"{prefix}{model_text}{provider_badge}{status}", 0, 0))

        # Add scroll indicator if needed
        if start_index > 0 or end_index < len(self._filtered_items):
            self._list_container.add_child(
                Text(theme.fg("muted", f"  ({self._selected_index + 1}/{len(self._filtered_items)})"), 0, 0)
            )

        if self._filtered_items:
            selected = self._filtered_items[self._selected_index]
            self._list_container.add_child(Spacer(1))
            self._list_container.add_child(Text(theme.fg("muted", f"  Model Name: {selected['model'].name}"), 0, 0))

    def handle_input(self, data: str) -> None:
        kb = get_keybindings()

        # Navigation
        if kb.matches(data, "tui.select.up"):
            if not self._filtered_items:
                return
            self._selected_index = (
                len(self._filtered_items) - 1 if self._selected_index == 0 else self._selected_index - 1
            )
            self._update_list()
            return
        if kb.matches(data, "tui.select.down"):
            if not self._filtered_items:
                return
            self._selected_index = (
                0 if self._selected_index == len(self._filtered_items) - 1 else self._selected_index + 1
            )
            self._update_list()
            return

        # Reorder enabled models
        reorder_up = kb.matches(data, "app.models.reorderUp")
        reorder_down = kb.matches(data, "app.models.reorderDown")
        if reorder_up or reorder_down:
            if self._enabled_ids is None:
                return
            item = self._filtered_items[self._selected_index] if self._filtered_items else None
            if item is not None and _is_enabled(self._enabled_ids, item["fullId"]):
                delta = -1 if reorder_up else 1
                current_index = self._enabled_ids.index(item["fullId"])
                new_index = current_index + delta
                # Only move if within bounds
                if 0 <= new_index < len(self._enabled_ids):
                    self._enabled_ids = _move(self._enabled_ids, item["fullId"], delta)
                    self._is_dirty = True
                    self._selected_index += delta
                    self._refresh()
                    self._notify_change()
            return

        # Toggle on Enter
        if kb.matches(data, "tui.select.confirm"):
            if self._filtered_items:
                item = self._filtered_items[self._selected_index]
                self._enabled_ids = _toggle(self._enabled_ids, item["fullId"])
                self._is_dirty = True
                self._refresh()
                self._notify_change()
            return

        # Enable all (filtered if search active, otherwise all)
        if kb.matches(data, "app.models.enableAll"):
            target_ids = [i["fullId"] for i in self._filtered_items] if self._search_input.get_value() else None
            self._enabled_ids = _enable_all(self._enabled_ids, self._all_ids, target_ids)
            self._is_dirty = True
            self._refresh()
            self._notify_change()
            return

        # Clear all (filtered if search active, otherwise all)
        if kb.matches(data, "app.models.clearAll"):
            target_ids = [i["fullId"] for i in self._filtered_items] if self._search_input.get_value() else None
            self._enabled_ids = _clear_all(self._enabled_ids, self._all_ids, target_ids)
            self._is_dirty = True
            self._refresh()
            self._notify_change()
            return

        # Toggle provider of current item
        if kb.matches(data, "app.models.toggleProvider"):
            if self._filtered_items:
                item = self._filtered_items[self._selected_index]
                provider = item["model"].provider
                provider_ids = [
                    model_id for model_id in self._all_ids if self._models_by_id[model_id].provider == provider
                ]
                all_enabled = all(_is_enabled(self._enabled_ids, model_id) for model_id in provider_ids)
                if all_enabled:
                    self._enabled_ids = _clear_all(self._enabled_ids, self._all_ids, provider_ids)
                else:
                    self._enabled_ids = _enable_all(self._enabled_ids, self._all_ids, provider_ids)
                self._is_dirty = True
                self._refresh()
                self._notify_change()
            return

        # Save/persist to settings
        if kb.matches(data, "app.models.save"):
            self._callbacks["onPersist"](None if self._enabled_ids is None else list(self._enabled_ids))
            self._is_dirty = False
            self._footer_text.set_text(self._get_footer_text())
            return

        # Ctrl+C - clear search or cancel if empty
        if matches_key(data, Key.ctrl("c")):
            if self._search_input.get_value():
                self._search_input.set_value("")
                self._refresh()
            else:
                self._callbacks["onCancel"]()
            return

        # Escape - cancel
        if matches_key(data, Key.escape):
            self._callbacks["onCancel"]()
            return

        # Pass everything else to search input
        self._search_input.handle_input(data)
        self._refresh()

    def get_search_input(self) -> Input:
        return self._search_input
