"""Mirror of pi coding-agent src/modes/interactive/components/extension-input.ts.

Simple text input component for extensions.
"""

from pidrei_tui import Container, Input, Spacer, Text, get_keybindings

from ..theme import theme
from .countdown_timer import CountdownTimer
from .dynamic_border import DynamicBorder
from .keybinding_hints import key_hint


class ExtensionInputComponent(Container):
    """Options: ``{"tui"?, "timeout"?}``."""

    def __init__(self, title: str, _placeholder, on_submit, on_cancel, opts: dict | None = None) -> None:
        super().__init__()
        opts = opts or {}

        self._on_submit_callback = on_submit
        self._on_cancel_callback = on_cancel
        self._base_title = title
        self._countdown: CountdownTimer | None = None
        self._focused = False

        self.add_child(DynamicBorder())
        self.add_child(Spacer(1))

        self._title_text = Text(theme.fg("accent", title), 1, 0)
        self.add_child(self._title_text)
        self.add_child(Spacer(1))

        timeout = opts.get("timeout")
        if timeout and timeout > 0 and opts.get("tui") is not None:
            self._countdown = CountdownTimer(
                timeout,
                opts["tui"],
                lambda s: self._title_text.set_text(theme.fg("accent", f"{self._base_title} ({s}s)")),
                lambda: self._on_cancel_callback(),
            )

        self._input = Input()
        self.add_child(self._input)
        self.add_child(Spacer(1))
        self.add_child(
            Text(f"{key_hint('tui.select.confirm', 'submit')}  {key_hint('tui.select.cancel', 'cancel')}", 1, 0)
        )
        self.add_child(Spacer(1))
        self.add_child(DynamicBorder())

    # Focusable implementation - propagate to input for IME cursor positioning
    @property
    def focused(self) -> bool:
        return self._focused

    @focused.setter
    def focused(self, value: bool) -> None:
        self._focused = value
        self._input.focused = value

    def handle_input(self, key_data: str) -> None:
        kb = get_keybindings()
        if kb.matches(key_data, "tui.select.confirm") or key_data == "\n":
            self._on_submit_callback(self._input.get_value())
        elif kb.matches(key_data, "tui.select.cancel"):
            self._on_cancel_callback()
        else:
            self._input.handle_input(key_data)

    def dispose(self) -> None:
        if self._countdown is not None:
            self._countdown.dispose()
