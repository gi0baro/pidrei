"""Mirror of pi coding-agent src/modes/interactive/components/bordered-loader.ts."""

from pidrei_tui import CancellableLoader, Container, Loader, Spacer, Text
from pidrei_tui.components.cancellable_loader import CancelToken

from .dynamic_border import DynamicBorder
from .keybinding_hints import key_hint


class BorderedLoader(Container):
    """Loader wrapped with borders for extension UI."""

    def __init__(self, tui, theme, message: str, options: dict | None = None) -> None:
        super().__init__()
        options = options or {}
        cancellable = options.get("cancellable")
        self._cancellable = cancellable if cancellable is not None else True
        self._signal_controller: CancelToken | None = None

        def border_color(s: str) -> str:
            return theme.fg("border", s)

        self.add_child(DynamicBorder(border_color))
        if self._cancellable:
            self._loader = CancellableLoader(
                tui,
                lambda s: theme.fg("accent", s),
                lambda s: theme.fg("muted", s),
                message,
            )
        else:
            self._signal_controller = CancelToken()
            self._loader = Loader(
                tui,
                lambda s: theme.fg("accent", s),
                lambda s: theme.fg("muted", s),
                message,
            )
        self.add_child(self._loader)
        if self._cancellable:
            self.add_child(Spacer(1))
            self.add_child(Text(key_hint("tui.select.cancel", "cancel"), 1, 0))
        self.add_child(Spacer(1))
        self.add_child(DynamicBorder(border_color))

    @property
    def signal(self):
        if self._cancellable:
            return self._loader.signal
        return self._signal_controller if self._signal_controller is not None else CancelToken()

    @property
    def on_abort(self):
        return self._loader.on_abort if self._cancellable else None

    @on_abort.setter
    def on_abort(self, fn) -> None:
        if self._cancellable:
            self._loader.on_abort = fn

    def handle_input(self, data: str) -> None:
        if self._cancellable:
            self._loader.handle_input(data)

    def dispose(self) -> None:
        dispose = getattr(self._loader, "dispose", None)
        if callable(dispose):
            dispose()
            return
        stop = getattr(self._loader, "stop", None)
        if callable(stop):
            stop()
