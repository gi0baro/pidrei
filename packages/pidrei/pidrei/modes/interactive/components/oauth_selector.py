"""Mirror of pi coding-agent src/modes/interactive/components/oauth-selector.ts.

Provider records are ``{"id", "name", "authType", "method"?, "status"?}``;
``status`` may be an ``AuthCheck`` dataclass (live code) or a camelCase dict
(mirrored tests pass plain records, like pi's object literals).
"""

import re

from pidrei_tui import Container, Input, Spacer, TruncatedText, fuzzy_filter, get_keybindings

from ..theme import theme
from .dynamic_border import DynamicBorder


_ENV_SOURCE_RE = re.compile(r"^[A-Z][A-Z0-9_]*(?:, [A-Z][A-Z0-9_]*)*$")


def format_auth_selector_provider_type(auth_type: str) -> str:
    return "subscription" if auth_type == "oauth" else "API key"


def _status_get(status, key: str):
    if isinstance(status, dict):
        return status.get(key)
    return getattr(status, key, None)


class OAuthSelectorComponent(Container):
    """Component that renders an auth provider selector."""

    def __init__(self, mode: str, providers: list, on_select, on_cancel, initial_search_input: str | None = None):
        super().__init__()

        self._mode = mode
        self._all_providers = providers
        self._filtered_providers = providers
        self._show_auth_type_labels = len({provider["authType"] for provider in providers}) > 1
        self._selected_index = 0
        self._on_select_callback = on_select
        self._on_cancel_callback = on_cancel
        self._focused = False

        # Add top border
        self.add_child(DynamicBorder())
        self.add_child(Spacer(1))

        # Add title
        title = "Select provider to configure:" if mode == "login" else "Select provider to logout:"
        self.add_child(TruncatedText(theme.fg("accent", theme.bold(title)), 1, 0))
        self.add_child(Spacer(1))

        self._search_input = Input()
        if initial_search_input:
            self._search_input.set_value(initial_search_input)

        def on_submit(_value=None) -> None:
            if 0 <= self._selected_index < len(self._filtered_providers):
                selected_provider = self._filtered_providers[self._selected_index]
                self._on_select_callback(selected_provider["id"], selected_provider["authType"])

        self._search_input.on_submit = on_submit
        self.add_child(self._search_input)
        self.add_child(Spacer(1))

        # Create list container
        self._list_container = Container()
        self.add_child(self._list_container)

        self.add_child(Spacer(1))

        # Add bottom border
        self.add_child(DynamicBorder())

        # Initial render
        self._filter_providers(initial_search_input or "")

    # Focusable implementation - propagate to search input for IME cursor
    # positioning
    @property
    def focused(self) -> bool:
        return self._focused

    @focused.setter
    def focused(self, value: bool) -> None:
        self._focused = value
        self._search_input.focused = value

    def _filter_providers(self, query: str) -> None:
        if query:
            self._filtered_providers = fuzzy_filter(
                self._all_providers,
                query,
                lambda provider: (
                    f"{provider['name']} {provider['id']} {provider['authType']} "
                    f"{(provider.get('method') or {}).get('name', '') if isinstance(provider.get('method'), dict) else getattr(provider.get('method'), 'name', '') or ''}"
                ),
            )
        else:
            self._filtered_providers = self._all_providers
        self._selected_index = max(0, min(self._selected_index, max(0, len(self._filtered_providers) - 1)))
        self._update_list()

    def _update_list(self) -> None:
        self._list_container.clear()

        max_visible = 8
        start_index = max(
            0,
            min(self._selected_index - max_visible // 2, len(self._filtered_providers) - max_visible),
        )
        end_index = min(start_index + max_visible, len(self._filtered_providers))

        for i in range(start_index, end_index):
            provider = self._filtered_providers[i]
            is_selected = i == self._selected_index

            status_indicator = self._format_status_indicator(provider)
            auth_type_label = (
                theme.fg("muted", f" [{format_auth_selector_provider_type(provider['authType'])}]")
                if self._show_auth_type_labels
                else ""
            )
            if is_selected:
                prefix = theme.fg("accent", "→ ")
                text = theme.fg("accent", provider["name"])
                line = prefix + text + auth_type_label + status_indicator
            else:
                text = f"  {theme.fg('text', provider['name'])}"
                line = text + auth_type_label + status_indicator

            self._list_container.add_child(TruncatedText(line, 1, 0))

        if start_index > 0 or end_index < len(self._filtered_providers):
            scroll_info = theme.fg("muted", f"  ({self._selected_index + 1}/{len(self._filtered_providers)})")
            self._list_container.add_child(TruncatedText(scroll_info, 1, 0))

        # Show "no providers" if empty
        if not self._filtered_providers:
            if not self._all_providers:
                if self._mode == "login":
                    message = "No providers available"
                else:
                    message = "No providers logged in. Use /login first."
            else:
                message = "No matching providers"
            self._list_container.add_child(TruncatedText(theme.fg("muted", f"  {message}"), 1, 0))

    def _format_status_indicator(self, provider: dict) -> str:
        status = provider.get("status")
        if not status:
            return theme.fg("muted", " • unconfigured")
        status_type = _status_get(status, "type")
        source = _status_get(status, "source")
        if status_type != provider["authType"]:
            label = "subscription configured" if status_type == "oauth" else "API key configured"
            return theme.fg("muted", " • ") + theme.fg("warning", label)
        if not source or source in ("OAuth", "stored credential"):
            return theme.fg("success", " ✓ configured")
        display_source = f"env: {source}" if _ENV_SOURCE_RE.match(source) else source
        return theme.fg("success", f" ✓ {display_source}")

    async def handle_input(self, key_data: str) -> None:
        kb = get_keybindings()
        # Up arrow
        if kb.matches(key_data, "tui.select.up"):
            if not self._filtered_providers:
                return
            self._selected_index = max(0, self._selected_index - 1)
            self._update_list()
        # Down arrow
        elif kb.matches(key_data, "tui.select.down"):
            if not self._filtered_providers:
                return
            self._selected_index = min(len(self._filtered_providers) - 1, self._selected_index + 1)
            self._update_list()
        # Enter
        elif kb.matches(key_data, "tui.select.confirm"):
            if 0 <= self._selected_index < len(self._filtered_providers):
                selected_provider = self._filtered_providers[self._selected_index]
                self._on_select_callback(selected_provider["id"], selected_provider["authType"])
        # Escape or Ctrl+C
        elif kb.matches(key_data, "tui.select.cancel"):
            self._on_cancel_callback()
        # Pass everything else to search input
        else:
            await self._search_input.handle_input(key_data)
            self._filter_providers(self._search_input.get_value())
