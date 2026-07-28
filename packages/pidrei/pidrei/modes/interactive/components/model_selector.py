"""Mirror of pi coding-agent src/modes/interactive/components/model-selector.ts.

Scoped model items are ``{"model", "thinkingLevel"?}`` records.
"""

import tonio.colored as tonio

from pidrei_ai.registry import ModelsRefreshOptions, models_are_equal
from pidrei_ai.utils.cancel import CancelToken
from pidrei_tui import Container, Input, Spacer, Text, fuzzy_filter, get_keybindings
from pidrei_tui._timers import Timeout

from ..model_search import get_model_selector_search_text
from ..theme import theme
from .dynamic_border import DynamicBorder
from .keybinding_hints import key_hint


class ModelSelectorComponent(Container):
    """Component that renders a model selector with search."""

    def __init__(
        self,
        tui,
        current_model,
        settings_manager,
        model_runtime,
        scoped_models: list,
        on_select,
        on_cancel,
        initial_search_input: str | None = None,
    ) -> None:
        super().__init__()

        self._tui = tui
        self._current_model = current_model
        self._settings_manager = settings_manager
        self._model_runtime = model_runtime
        self._scoped_models = list(scoped_models)
        self._scope = "scoped" if scoped_models else "all"
        self._on_select_callback = on_select
        self._on_cancel_callback = on_cancel

        self._all_models: list = []
        self._scoped_model_items: list = []
        self._active_models: list = []
        self._filtered_models: list = []
        self._selected_index = 0
        self._error_message: str | None = None
        self._refresh_status_message = "Refreshing model catalogs…"
        self._refresh_status_success = False
        self._scope_text: Text | None = None
        self._scope_hint_text: Text | None = None
        self._refresh_abort_controller = CancelToken()
        self._refresh_timeout: Timeout | None = None
        self._closed = False
        self._focused = False

        # Add top border
        self.add_child(DynamicBorder())
        self.add_child(Spacer(1))

        # Add hint about model filtering
        if scoped_models:
            self._scope_text = Text(self._get_scope_text(), 0, 0)
            self.add_child(self._scope_text)
            self._scope_hint_text = Text(self._get_scope_hint_text(), 0, 0)
            self.add_child(self._scope_hint_text)
        else:
            hint_text = "Only showing models from configured providers. Use /login to add providers."
            self.add_child(Text(theme.fg("warning", hint_text), 0, 0))
        self.add_child(Spacer(1))

        # Create search input
        self._search_input = Input()
        if initial_search_input:
            self._search_input.set_value(initial_search_input)

        def on_submit(_value=None) -> None:
            # Enter on search input selects the first filtered item
            if 0 <= self._selected_index < len(self._filtered_models):
                self._handle_select(self._filtered_models[self._selected_index]["model"])

        self._search_input.on_submit = on_submit
        self.add_child(self._search_input)

        self.add_child(Spacer(1))

        # Create list container
        self._list_container = Container()
        self.add_child(self._list_container)

        self.add_child(Spacer(1))

        # Add bottom border
        self.add_child(DynamicBorder())

        # Render the current snapshot immediately, then refresh in the
        # background.
        self._load_models_from_snapshot()
        if initial_search_input:
            self._filter_models(initial_search_input)
        else:
            self._update_list()
        self._tui.request_render()
        tonio.spawn.without_tracking(self._refresh_models())

    # Focusable implementation - propagate to search input for IME cursor
    # positioning
    @property
    def focused(self) -> bool:
        return self._focused

    @focused.setter
    def focused(self, value: bool) -> None:
        self._focused = value
        self._search_input.focused = value

    def _load_models_from_snapshot(self) -> None:
        models = [
            {"provider": model.provider, "id": model.id, "model": model}
            for model in self._model_runtime.get_available_snapshot()
        ]
        self._all_models = self._sort_models(models)
        refreshed_scoped = []
        for scoped in self._scoped_models:
            refreshed = self._model_runtime.get_model(scoped["model"].provider, scoped["model"].id)
            refreshed_scoped.append({**scoped, "model": refreshed} if refreshed is not None else scoped)
        self._scoped_models = refreshed_scoped
        self._scoped_model_items = [
            {"provider": scoped["model"].provider, "id": scoped["model"].id, "model": scoped["model"]}
            for scoped in self._scoped_models
        ]
        self._active_models = self._scoped_model_items if self._scope == "scoped" else self._all_models
        self._filtered_models = self._active_models
        current_index = next(
            (i for i, item in enumerate(self._filtered_models) if models_are_equal(self._current_model, item["model"])),
            -1,
        )
        if current_index >= 0:
            self._selected_index = current_index
        else:
            self._selected_index = min(self._selected_index, max(0, len(self._filtered_models) - 1))

    async def _refresh_models(self) -> None:
        timeout_ms = 15_000
        timed_out = False

        async def on_timeout() -> None:
            nonlocal timed_out
            timed_out = True
            self._refresh_abort_controller.cancel()

        self._refresh_timeout = Timeout(timeout_ms, on_timeout)
        try:
            result = await self._model_runtime.refresh(ModelsRefreshOptions(cancel=self._refresh_abort_controller))
            if self._closed:
                return
            self._refresh_status_message = ""
            if result.aborted and timed_out:
                self._error_message = "Model refresh timed out; showing cached models."
            elif len(result.errors) == 1:
                provider = next(iter(result.errors))
                self._error_message = f"Could not refresh {provider}; showing cached models."
            elif len(result.errors) > 1:
                self._error_message = f"Could not refresh {len(result.errors)} model catalogs; showing cached models."
            else:
                self._error_message = self._model_runtime.get_error()
                if not self._error_message:
                    self._refresh_status_message = "Model catalogs refreshed."
                    self._refresh_status_success = True
            self._load_models_from_snapshot()
            self._filter_models(self._search_input.get_value())
            self._tui.request_render()
        finally:
            if self._refresh_timeout is not None:
                self._refresh_timeout.cancel()

    def _close(self) -> None:
        self._closed = True
        if self._refresh_timeout is not None:
            self._refresh_timeout.cancel()
        self._refresh_abort_controller.cancel()

    def _sort_models(self, models: list) -> list:
        def sort_key(item: dict):
            is_current = models_are_equal(self._current_model, item["model"])
            return (0 if is_current else 1, item["provider"].lower(), item["provider"])

        return sorted(models, key=sort_key)

    def _get_scope_text(self) -> str:
        all_text = theme.fg("accent", "all") if self._scope == "all" else theme.fg("muted", "all")
        scoped_text = theme.fg("accent", "scoped") if self._scope == "scoped" else theme.fg("muted", "scoped")
        return f"{theme.fg('muted', 'Scope: ')}{all_text}{theme.fg('muted', ' | ')}{scoped_text}"

    def _get_scope_hint_text(self) -> str:
        return key_hint("tui.input.tab", "scope") + theme.fg("muted", " (all/scoped)")

    def _set_scope(self, scope: str) -> None:
        if self._scope == scope:
            return
        self._scope = scope
        self._active_models = self._scoped_model_items if self._scope == "scoped" else self._all_models
        current_index = next(
            (i for i, item in enumerate(self._active_models) if models_are_equal(self._current_model, item["model"])),
            -1,
        )
        self._selected_index = max(current_index, 0)
        self._filter_models(self._search_input.get_value())
        if self._scope_text is not None:
            self._scope_text.set_text(self._get_scope_text())

    def _filter_models(self, query: str) -> None:
        if query:
            self._filtered_models = fuzzy_filter(
                self._active_models,
                query,
                lambda item: get_model_selector_search_text(
                    {"id": item["id"], "provider": item["provider"], "name": item["model"].name}
                ),
            )
        else:
            self._filtered_models = self._active_models
        self._selected_index = min(self._selected_index, max(0, len(self._filtered_models) - 1))
        self._update_list()

    def _update_list(self) -> None:
        self._list_container.clear()

        max_visible = 10
        start_index = max(
            0,
            min(self._selected_index - max_visible // 2, len(self._filtered_models) - max_visible),
        )
        end_index = min(start_index + max_visible, len(self._filtered_models))

        # Show visible slice of filtered models
        for i in range(start_index, end_index):
            item = self._filtered_models[i]
            is_selected = i == self._selected_index
            is_current = models_are_equal(self._current_model, item["model"])

            provider_badge = theme.fg("muted", f"[{item['provider']}]")
            checkmark = theme.fg("success", " ✓") if is_current else ""
            if is_selected:
                prefix = theme.fg("accent", "→ ")
                line = f"{prefix + theme.fg('accent', item['id'])} {provider_badge}{checkmark}"
            else:
                line = f"  {item['id']} {provider_badge}{checkmark}"

            self._list_container.add_child(Text(line, 0, 0))

        # Add scroll indicator if needed
        if start_index > 0 or end_index < len(self._filtered_models):
            scroll_info = theme.fg("muted", f"  ({self._selected_index + 1}/{len(self._filtered_models)})")
            self._list_container.add_child(Text(scroll_info, 0, 0))

        # Show error message or "no results" if empty
        if self._error_message:
            # Show error in red
            for line in self._error_message.split("\n"):
                self._list_container.add_child(Text(theme.fg("error", line), 0, 0))
        elif not self._filtered_models:
            self._list_container.add_child(Text(theme.fg("muted", "  No matching models"), 0, 0))
        else:
            selected = self._filtered_models[self._selected_index]
            self._list_container.add_child(Spacer(1))
            self._list_container.add_child(Text(theme.fg("muted", f"  Model Name: {selected['model'].name}"), 0, 0))
        if self._refresh_status_message:
            self._list_container.add_child(Spacer(1))
            self._list_container.add_child(
                Text(
                    theme.fg(
                        "success" if self._refresh_status_success else "muted",
                        f"  {self._refresh_status_message}",
                    ),
                    0,
                    0,
                )
            )

    async def handle_input(self, key_data: str) -> None:
        kb = get_keybindings()
        if kb.matches(key_data, "tui.input.tab"):
            if self._scoped_model_items:
                next_scope = "scoped" if self._scope == "all" else "all"
                self._set_scope(next_scope)
                if self._scope_hint_text is not None:
                    self._scope_hint_text.set_text(self._get_scope_hint_text())
            return
        # Up arrow - wrap to bottom when at top
        if kb.matches(key_data, "tui.select.up"):
            if not self._filtered_models:
                return
            self._selected_index = (
                len(self._filtered_models) - 1 if self._selected_index == 0 else self._selected_index - 1
            )
            self._update_list()
        # Down arrow - wrap to top when at bottom
        elif kb.matches(key_data, "tui.select.down"):
            if not self._filtered_models:
                return
            self._selected_index = (
                0 if self._selected_index == len(self._filtered_models) - 1 else self._selected_index + 1
            )
            self._update_list()
        # Enter
        elif kb.matches(key_data, "tui.select.confirm"):
            if 0 <= self._selected_index < len(self._filtered_models):
                self._handle_select(self._filtered_models[self._selected_index]["model"])
        # Escape or Ctrl+C
        elif kb.matches(key_data, "tui.select.cancel"):
            self._close()
            self._on_cancel_callback()
        # Pass everything else to search input
        else:
            await self._search_input.handle_input(key_data)
            self._filter_models(self._search_input.get_value())

    def _handle_select(self, model) -> None:
        self._close()
        # Save as new default
        self._settings_manager.set_default_model_and_provider(model.provider, model.id)
        self._on_select_callback(model)

    def get_search_input(self) -> Input:
        return self._search_input
