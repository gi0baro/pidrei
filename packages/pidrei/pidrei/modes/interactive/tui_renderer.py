"""Mirror of pi coding-agent src/modes/interactive/tui-renderer.ts."""

from pidrei_tui import TUI, ProcessTerminal, TuiAltScreen, TuiMainScreen

from ...utils.clipboard import copy_to_clipboard
from ...utils.open_browser import open_browser
from .components.keybinding_hints import key_display_text
from .theme import theme


def create_interactive_tui(
    *,
    tui_mode: str,
    show_hardware_cursor: bool,
    log_directory: str,
    terminal=None,
    fullscreen_copy_on_select: bool | None = None,
) -> TUI:
    """Composition root shared by coding-agent presentations."""
    terminal = terminal if terminal is not None else ProcessTerminal()
    if tui_mode == "fullscreen":

        def style_search_match(text: str) -> str:
            return theme.bg("searchMatchBg", theme.fg("searchMatchText", text))

        async def copy_selection(text: str) -> bool:
            try:
                await copy_to_clipboard(text)
                return True
            except Exception:
                return False

        def scroll_to_end_indicator() -> str:
            shortcut = key_display_text("tui.altScreen.bottom")
            label = f" ↓ Jump to latest message{f' · {shortcut}' if shortcut else ''} "
            return theme.bg("selectedBg", theme.fg("text", label))

        return TuiAltScreen(
            terminal,
            show_hardware_cursor,
            log_directory,
            search_match_style=lambda text: theme.underline(style_search_match(text)),
            search_current_match_style=lambda text: theme.bold(theme.inverse(style_search_match(text))),
            search_navigation_button_style=lambda text, hovered: theme.underline(text) if hovered else text,
            scroll_to_end_indicator=scroll_to_end_indicator,
            open_url=open_browser,
            copy_on_select=fullscreen_copy_on_select,
            copy_selection=copy_selection,
        )
    return TuiMainScreen(terminal, show_hardware_cursor, log_directory)


class _InteractiveTuiReference:
    """Stable reference for components while InteractiveMode replaces the active renderer.

    pi uses a `Proxy`; attribute delegation is the Python equivalent. `isinstance`
    is deliberately NOT faked (pi's `getPrototypeOf` trap makes `instanceof` see
    through), so the few renderer-type checks read `self._renderer` instead.
    """

    def __init__(self, get_tui) -> None:
        object.__setattr__(self, "_get_tui", get_tui)

    def __getattr__(self, name: str):
        get_tui = object.__getattribute__(self, "_get_tui")
        tui = get_tui()
        value = getattr(tui, name)
        if not callable(value):
            return value

        # A captured method must follow a later renderer swap (pi #7731), but it
        # must not re-resolve on every call either: a wrapper installed *onto*
        # the reference lives on the renderer, so re-reading the attribute each
        # time would call the wrapper from inside itself.
        bound = [tui, value]

        def call(*args, **kwargs):
            current = get_tui()
            if current is not bound[0]:
                method = getattr(current, name)
                if not callable(method):
                    raise TypeError(f"TUI property {name} is not callable")
                bound[0], bound[1] = current, method
            return bound[1](*args, **kwargs)

        return call

    def __setattr__(self, name: str, value) -> None:
        setattr(object.__getattribute__(self, "_get_tui")(), name, value)


def create_interactive_tui_reference(get_tui) -> TUI:
    return _InteractiveTuiReference(get_tui)
