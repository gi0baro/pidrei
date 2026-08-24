"""Mirror of pi coding-agent src/modes/interactive/components/settings-submenu.ts.

pi's step callbacks (`options`, `preselect`, `title`, `description`) are plain
synchronous functions and stay that way here; only the callbacks that reach the
settings manager or the session — `on_select`, `on_cancel`, `on_complete` — are
awaited, matching the rest of the settings surface.
"""

from pidrei_tui import Container, Input, SelectList, Spacer, Text, fuzzy_filter, get_keybindings

from ..theme import get_select_list_theme, theme


__all__ = ["SelectSubmenu", "SteppedSubmenu"]


SUBMENU_SELECT_LIST_LAYOUT = {"minPrimaryColumnWidth": 12, "maxPrimaryColumnWidth": 32}


class SelectSubmenu(Container):
    """Single-step submenu that shows a titled select list.

    With ``searchable`` set in ``submenu_options``, typing filters the list
    using fuzzy matching.
    """

    def __init__(
        self,
        title: str,
        description: str,
        options: list,
        current_value: str,
        on_select,
        on_cancel,
        on_selection_change=None,
        submenu_options: dict | None = None,
    ) -> None:
        super().__init__()

        submenu_options = submenu_options or {}
        self._all_options = options
        self._list_layout = submenu_options.get("layout") or SUBMENU_SELECT_LIST_LAYOUT
        self._on_select = on_select
        self._on_cancel = on_cancel
        self._on_selection_change = on_selection_change
        self._searchable = bool(submenu_options.get("searchable"))

        # Title
        self.add_child(Text(theme.bold(theme.fg("accent", title)), 0, 0))

        # Description
        if description:
            self.add_child(Spacer(1))
            self.add_child(Text(theme.fg("muted", description), 0, 0))

        # Search input. pi also wires `searchInput.onSubmit` to forward Enter to
        # the list; Enter is a `tui.select.confirm` nav key, so `handle_input`
        # below routes it to the list before the input ever sees it, in both
        # ports. The hook is unreachable, and a sync-only `on_submit` could not
        # await the list here anyway, so it is left unwired.
        self._search_input = None
        if self._searchable:
            self.add_child(Spacer(1))
            self._search_input = Input()
            self.add_child(self._search_input)

        # Spacer
        self.add_child(Spacer(1))

        # Select list
        self._select_list = self._build_select_list(options, current_value)
        self._list_child_index = len(self.children)
        self.add_child(self._select_list)

        # Hint
        self.add_child(Spacer(1))
        hint = (
            "  Type to filter · Enter to select · Esc to go back"
            if self._searchable
            else "  Enter to select · Esc to go back"
        )
        self.add_child(Text(theme.fg("dim", hint), 0, 0))

    def _build_select_list(self, options: list, preselect: str) -> SelectList:
        select_list = SelectList(options, min(len(options), 10), get_select_list_theme(), self._list_layout)

        index = next((i for i, option in enumerate(options) if option["value"] == preselect), -1)
        if index != -1:
            select_list.set_selected_index(index)

        select_list.on_select = lambda item: self._on_select(item["value"])
        select_list.on_cancel = self._on_cancel
        if self._on_selection_change is not None:
            callback = self._on_selection_change
            select_list.on_selection_change = lambda item: callback(item["value"])

        return select_list

    def _apply_filter(self, query: str) -> None:
        filtered = (
            fuzzy_filter(self._all_options, query, lambda item: f"{item['label']} {item.get('description') or ''}")
            if query
            else self._all_options
        )

        new_list = self._build_select_list(filtered, "")
        children = list(self.children)
        children[self._list_child_index] = new_list
        self.set_children(children)
        self._select_list = new_list

    async def handle_input(self, data: str) -> None:
        if self._search_input is not None:
            kb = get_keybindings()
            is_nav = (
                kb.matches(data, "tui.select.up")
                or kb.matches(data, "tui.select.down")
                or kb.matches(data, "tui.select.confirm")
                or kb.matches(data, "tui.select.cancel")
            )
            if is_nav:
                await self._select_list.handle_input(data)
            else:
                await self._search_input.handle_input(data)
                self._apply_filter(self._search_input.get_value())
        else:
            await self._select_list.handle_input(data)


# ============================================================================
# SteppedSubmenu — reusable multi-step selector
# ============================================================================


class SteppedSubmenu(Container):
    """Generic N-step submenu built on top of :class:`SelectSubmenu`.

    Each step's options can depend on prior selections via the shared context.
    Esc goes back one step; Esc at step 0 cancels. With ``loop`` set,
    completing the final step invokes ``on_complete`` then returns to step 0.

    A step is a dict mirroring pi's ``SteppedSubmenuStep``: ``key`` (the
    context key the selected value is stored under), ``title`` and
    ``description`` (a string, or a callable taking the context), ``options``
    (callable taking the context, called fresh each time the step is shown),
    and the optional ``preselect`` (callable), ``searchable`` and ``layout``.
    """

    def __init__(self, steps: list, on_complete, on_cancel, opts: dict | None = None) -> None:
        super().__init__()
        self._steps = steps
        self._on_complete = on_complete
        self._on_cancel = on_cancel
        self._opts = opts or {}
        self._context: dict = dict(self._opts.get("initialContext") or {})
        self._active_component = self._build_step(self._opts.get("startAtStep") or 0)

    def _build_step(self, step_index: int):
        step = self._steps[step_index]
        total = len(self._steps)
        step_label = f"Step {step_index + 1}/{total} · " if total > 1 else ""

        title = step["title"](self._context) if callable(step["title"]) else step["title"]
        desc = step["description"](self._context) if callable(step["description"]) else step["description"]
        items = step["options"](self._context)
        preselect = (step.get("preselect")(self._context) if step.get("preselect") is not None else None) or ""

        async def on_select(value: str) -> None:
            self._context[step["key"]] = value

            if step_index < total - 1:
                # Advance to next step
                self._active_component = self._build_step(step_index + 1)
            else:
                # Final step — deliver result
                await self._on_complete(dict(self._context))

                if self._opts.get("loop"):
                    self._context = {}
                    self._active_component = self._build_step(0)
                else:
                    await self._on_cancel()

        async def on_cancel() -> None:
            if step_index > 0:
                self._context.pop(step["key"], None)
                self._active_component = self._build_step(step_index - 1)
            else:
                await self._on_cancel()

        submenu_options = (
            {"searchable": step.get("searchable"), "layout": step.get("layout")}
            if step.get("searchable") or step.get("layout")
            else None
        )
        return SelectSubmenu(
            title,
            f"{step_label}{desc}",
            items,
            preselect,
            on_select,
            on_cancel,
            None,
            submenu_options,
        )

    def render(self, width: int) -> list[str]:
        return self._active_component.render(width)

    async def handle_input(self, data: str) -> None:
        handle_input = getattr(self._active_component, "handle_input", None)
        if handle_input is not None:
            await handle_input(data)

    def invalidate(self) -> None:
        invalidate = getattr(self._active_component, "invalidate", None)
        if invalidate is not None:
            invalidate()
