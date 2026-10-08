"""Mirror of pi coding-agent src/extensions/mcp/ui.ts: the `/mcp` manager
view: menus that rebuild while servers connect, a read-only status screen,
and the sign-in screen that accepts a pasted redirect URL.

The view lives on the UI island (spec/ui-island.md). `manage()` runs on its
own coroutine (spawned by the factory) and changes the view only through
`tui.apply`, as do the rebuilds a connection change asks for. Key handlers
run under the UI state lock and settle the open screen's answer, which the
manage coroutine awaits. The menu builders read the extension's state
without its lock, so the extension lock is never taken under the UI lock.
"""

import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import tonio.colored as tonio

from pidrei_tui import Container, Input, SelectList, Spacer, Text, truncate_to_width, visible_width
from pidrei_utils.cancel import CancelToken

from ...modes.interactive.components.auth_url import AuthUrlComponent
from ...modes.interactive.components.dynamic_border import DynamicBorder
from ...modes.interactive.components.keybinding_hints import key_hint
from ...modes.interactive.theme import get_select_list_theme


@dataclass(slots=True)
class McpMenu:
    title: str
    items: list[dict[str, str]]
    # What the confirm key does, for the key hint.
    confirm_label: str
    # What the cancel key does, for the key hint.
    cancel_label: str
    # Shown below the title.
    details: str | None = None
    # Shown below the details in the error color.
    error: str | None = None
    # Shown when there are no items.
    empty: str | None = None
    # Value of the item selected when the menu opens.
    selected: str | None = None


type Subscribe = Callable[[Callable[[], None]], Callable[[], None]]


class _Answer:
    """What a screen resolves to; the first settle wins."""

    __slots__ = ("_guard", "done", "value")

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self.done = tonio.Event()
        self.value: str | None = None

    @property
    def settled(self) -> bool:
        return self.done.is_set()

    def settle(self, value: str | None) -> None:
        with self._guard:
            if self.done.is_set():
                return
            self.value = value
            self.done.set()


def _frame(theme: Any, title: str, body: list[Any], footer: str | None = None) -> Container:
    container = Container()
    container.add_child(DynamicBorder(lambda text: theme.fg("accent", text)))
    container.add_child(Text(theme.fg("accent", theme.bold(title)), 1, 0))
    for child in body:
        container.add_child(child)
    if footer:
        container.add_child(Spacer(1))
        container.add_child(Text(theme.fg("dim", footer), 1, 0))
    container.add_child(DynamicBorder(lambda text: theme.fg("accent", text)))
    return container


_MAX_VISIBLE_ITEMS = 12


class McpManagerView:
    """`menu`, `status` and `redirect_url` are called from the manage
    coroutine; `render`, `handle_input`, `invalidate` and `focused` from the
    UI, under its lock."""

    def __init__(self, tui: Any, theme: Any, keybindings: Any) -> None:
        self._tui = tui
        self._theme = theme
        self._keybindings = keybindings
        self._content = _frame(theme, "MCP servers", [Text(theme.fg("muted", "Loading…"), 1, 1)])
        self._input_handler: Callable[[str], None] | None = None
        self._input_target: Any = None
        self._focused = False

    @property
    def focused(self) -> bool:
        return self._focused

    @focused.setter
    def focused(self, value: bool) -> None:
        self._focused = value
        if self._input_target is not None:
            self._input_target.focused = value

    def _set_content(
        self, content: Container, input_handler: Callable[[str], None] | None = None, input_target: Any = None
    ) -> None:
        """Under the UI lock."""
        if self._input_target is not None:
            self._input_target.focused = False
        self._content = content
        self._input_handler = input_handler
        self._input_target = input_target
        if input_target is not None:
            input_target.focused = self._focused
        self._tui.request_render()

    async def menu(self, build: Callable[[], McpMenu], subscribe: Subscribe | None = None) -> str | None:
        """Show a menu and wait for the chosen item's value, or None when
        cancelled. `subscribe` rebuilds the menu on every change, keeping the
        selected item."""
        answer = _Answer()
        selected: list[str | None] = [None]
        theme = self._theme

        def render() -> None:
            if answer.settled:
                return
            menu = build()
            wanted = selected[0] if selected[0] is not None else menu.selected
            body: list[Any] = []
            if menu.details:
                body.append(Text(theme.fg("muted", menu.details), 1, 0))
            if menu.error:
                body.append(Text(theme.fg("error", menu.error), 1, 0))
            body.append(Spacer(1))
            footer = f"{key_hint('tui.select.confirm', menu.confirm_label)} • {key_hint('tui.select.cancel', menu.cancel_label)}"
            if not menu.items:
                body.append(Text(theme.fg("muted", menu.empty or "Nothing to show."), 1, 0))

                def on_empty_input(data: str) -> None:
                    if self._keybindings.matches(data, "tui.select.cancel"):
                        answer.settle(None)

                self._set_content(
                    _frame(theme, menu.title, body, key_hint("tui.select.cancel", menu.cancel_label)), on_empty_input
                )
                return
            items = menu.items
            select_list = SelectList(items, min(len(items), _MAX_VISIBLE_ITEMS), get_select_list_theme())
            index = next((position for position, item in enumerate(items) if item["value"] == wanted), -1)
            if index != -1:
                select_list.set_selected_index(index)
            current = select_list.get_selected_item()
            selected[0] = current["value"] if current is not None else None

            def on_selection_change(item: dict[str, str]) -> None:
                selected[0] = item["value"]

            select_list.on_selection_change = on_selection_change
            select_list.on_select = lambda item: answer.settle(item["value"])
            select_list.on_cancel = lambda: answer.settle(None)
            body.append(select_list)
            self._set_content(_frame(theme, menu.title, body, footer), select_list.handle_input)

        self._tui.apply(render)
        unsubscribe = subscribe(lambda: self._tui.apply(render)) if subscribe is not None else None
        try:
            await answer.done.wait()
        finally:
            if unsubscribe is not None:
                unsubscribe()
        return answer.value

    def status(self, title: str, message: str, on_cancel: Callable[[], None] | None = None) -> None:
        """Show a message while an operation runs. With `on_cancel`, the cancel
        key calls it (on the input path: it must be synchronous)."""
        theme = self._theme

        def show() -> None:
            body = [Spacer(1), Text(theme.fg("muted", message), 1, 0)]
            if on_cancel is None:
                self._set_content(_frame(theme, title, body))
                return

            def on_input(data: str) -> None:
                if self._keybindings.matches(data, "tui.select.cancel"):
                    on_cancel()

            self._set_content(_frame(theme, title, body, key_hint("tui.select.cancel", "cancel")), on_input)

        self._tui.apply(show)

    async def redirect_url(self, title: str, authorization_url: str, cancel: CancelToken) -> str | None:
        """Show the authorization URL and wait for a pasted redirect URL. None
        when cancelled or when `cancel` fires (the browser reached the callback)."""
        if cancel.cancelled:
            return None
        answer = _Answer()
        theme = self._theme

        def show() -> None:
            if answer.settled:
                return
            field_input = Input()
            link = AuthUrlComponent(self._tui, authorization_url)
            body = [
                Spacer(1),
                Text(theme.fg("muted", "Approve access in your browser. If it did not open, visit:"), 1, 0),
                link,
                Spacer(1),
                Text(
                    theme.fg("muted", "If the browser runs on another machine, paste the URL it was redirected to:"),
                    1,
                    0,
                ),
                field_input,
            ]

            def on_input(data: str) -> None:
                if self._keybindings.matches(data, "tui.select.confirm"):
                    value = field_input.get_value().strip()
                    if value:
                        answer.settle(value)
                    return
                if self._keybindings.matches(data, "tui.select.cancel"):
                    answer.settle(None)
                    return
                if self._keybindings.matches(data, "app.message.copy"):
                    # pi's `void link.copy()`; `copy` reports its own failure in the hint.
                    self._tui.spawn(link.copy())
                    return
                field_input.handle_input(data)

            footer = f"{key_hint('tui.select.confirm', 'submit')} • {key_hint('tui.select.cancel', 'cancel')}"
            self._set_content(_frame(theme, title, body, footer), on_input, field_input)

        self._tui.apply(show)
        unsubscribe = cancel.on_cancel(lambda _reason: answer.settle(None))
        try:
            await answer.done.wait()
        finally:
            unsubscribe()
        return answer.value

    def handle_input(self, data: str) -> None:
        if self._input_handler is not None:
            self._input_handler(data)
        self._tui.request_render()

    def render(self, width: int) -> list[str]:
        return [
            truncate_to_width(line, width, "") if visible_width(line) > width else line
            for line in self._content.render(width)
        ]

    def invalidate(self) -> None:
        self._content.invalidate()


async def show_mcp_manager(ctx: Any, manage: Callable[[McpManagerView], Awaitable[None]]) -> None:
    """Run `manage` in the manager view until it returns."""

    def factory(tui: Any, theme: Any, keybindings: Any, done: Callable[[Any], None]) -> McpManagerView:
        view = McpManagerView(tui, theme, keybindings)

        async def run() -> None:
            try:
                await manage(view)
            except Exception as error:
                ctx.ui.notify(str(error), "error")
            finally:
                done(None)

        tui.spawn(run())
        return view

    await ctx.ui.custom(factory)
