"""Mirror of pi coding-agent src/modes/interactive/components/trust-selector.ts."""

from pidrei_tui import Container, Spacer, Text, get_keybindings

from ....core.trust_manager import get_project_trust_options
from ..theme import theme
from .dynamic_border import DynamicBorder
from .keybinding_hints import key_hint, raw_key_hint


def _format_decision(trust_path: str | None, decision) -> str:
    if decision is None:
        return "none"
    label = "trusted" if decision.decision else "untrusted"
    if trust_path is not None and decision.path != trust_path:
        return f"{label} (inherited from {decision.path})"
    return f"{label} ({decision.path})"


class TrustSelectorComponent(Container):
    """Options: ``{"cwd", "savedDecision", "projectTrusted", "onSelect", "onCancel"}``.

    ``onSelect`` receives a ``{"trusted", "updates"}`` record and must be
    coroutine-returning (it persists trust decisions — pi's sync ``onSelect``
    blocks its event loop on the store write, which the never-block rule
    forbids here). ``onCancel`` is sync.
    """

    def __init__(self, options: dict) -> None:
        super().__init__()

        self._saved_decision = options["savedDecision"]
        self._trust_options = get_project_trust_options(options["cwd"])
        self._selected_index = max(
            0,
            next((i for i, option in enumerate(self._trust_options) if self._is_saved_option(option)), -1),
        )
        self._on_select_callback = options["onSelect"]
        self._on_cancel_callback = options["onCancel"]

        self.add_child(DynamicBorder())
        self.add_child(Spacer(1))
        self.add_child(Text(theme.fg("accent", theme.bold("Project trust")), 1, 0))
        self.add_child(Text(theme.fg("muted", options["cwd"]), 1, 0))
        self.add_child(Spacer(1))
        saved_path = self._trust_options[0].saved_path if self._trust_options else None
        self.add_child(
            Text(
                theme.fg("muted", f"Saved decision: {_format_decision(saved_path, options['savedDecision'])}"),
                1,
                0,
            )
        )
        self.add_child(
            Text(
                theme.fg("muted", f"Current session: {'trusted' if options['projectTrusted'] else 'untrusted'}"),
                1,
                0,
            )
        )
        self.add_child(Spacer(1))

        self._list_container = Container()
        self.add_child(self._list_container)
        self.add_child(Spacer(1))
        self.add_child(
            Text(
                raw_key_hint("↑↓", "navigate")
                + "  "
                + key_hint("tui.select.confirm", "save")
                + "  "
                + key_hint("tui.select.cancel", "cancel"),
                1,
                0,
            )
        )
        self.add_child(Spacer(1))
        self.add_child(DynamicBorder())

        self._update_list()

    def _is_saved_option(self, option) -> bool:
        return (
            option.saved_path is not None
            and self._saved_decision is not None
            and self._saved_decision.decision == option.trusted
            and self._saved_decision.path == option.saved_path
        )

    def _update_list(self) -> None:
        self._list_container.clear()
        for i, option in enumerate(self._trust_options):
            is_selected = i == self._selected_index
            is_current = self._is_saved_option(option)
            current_marker = theme.fg("accent", "✓ ") if is_current else "  "
            prefix = theme.fg("accent", "→ ") if is_selected else "  "
            label = theme.fg("accent", option.label) if is_selected else theme.fg("text", option.label)
            self._list_container.add_child(Text(f"{prefix}{current_marker}{label}", 1, 0))

    async def handle_input(self, key_data: str) -> None:
        kb = get_keybindings()
        if kb.matches(key_data, "tui.select.up") or key_data == "k":
            self._selected_index = max(0, self._selected_index - 1)
            self._update_list()
        elif kb.matches(key_data, "tui.select.down") or key_data == "j":
            self._selected_index = min(len(self._trust_options) - 1, self._selected_index + 1)
            self._update_list()
        elif kb.matches(key_data, "tui.select.confirm") or key_data == "\n":
            if 0 <= self._selected_index < len(self._trust_options):
                selected = self._trust_options[self._selected_index]
                await self._on_select_callback({"trusted": selected.trusted, "updates": selected.updates})
        elif kb.matches(key_data, "tui.select.cancel"):
            self._on_cancel_callback()
