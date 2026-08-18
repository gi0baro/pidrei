"""Custom Footer

Demonstrates `ctx.ui.set_footer()`. `/footer` toggles a one-line footer with
token stats on the left and model + git branch on the right.

The footer data provider passed to the factory exposes data not otherwise
accessible:
- `get_git_branch()`: current git branch
- `get_extension_statuses()`: texts from `ctx.ui.set_status()`

Token stats come from `ctx.session_manager` / `ctx.model` (already
accessible).

Start pidrei with this extension:
    pidrei -e ./examples/extensions/custom_footer.py
"""

from pidrei_tui import truncate_to_width, visible_width


def format_tokens(count: int) -> str:
    return str(count) if count < 1000 else f"{count / 1000:.1f}k"


class CustomFooter:
    """Footer component: `render(width) -> list[str]`, `invalidate()`, and a
    `dispose()` that drops the branch-change subscription."""

    def __init__(self, ui, theme, footer_data, ctx) -> None:
        self._theme = theme
        self._footer_data = footer_data
        self._ctx = ctx
        # Re-render when the git branch changes under us.
        self._unsubscribe = footer_data.on_branch_change(lambda: ui.request_render())

    def dispose(self) -> None:
        self._unsubscribe()

    def invalidate(self) -> None:
        pass

    def render(self, width: int) -> list[str]:
        # Compute tokens from ctx (already accessible to extensions)
        input_tokens = 0
        output_tokens = 0
        cost = 0.0
        for entry in self._ctx.session_manager.get_branch():
            message = entry.get("message")
            if entry.get("type") == "message" and getattr(message, "role", None) == "assistant":
                input_tokens += message.usage.input
                output_tokens += message.usage.output
                cost += message.usage.cost.total

        # Get git branch (not otherwise accessible)
        branch = self._footer_data.get_git_branch()

        left = self._theme.fg("dim", f"↑{format_tokens(input_tokens)} ↓{format_tokens(output_tokens)} ${cost:.3f}")
        branch_str = f" ({branch})" if branch else ""
        model = self._ctx.model
        right = self._theme.fg("dim", f"{model.id if model is not None else 'no-model'}{branch_str}")

        pad = " " * max(1, width - visible_width(left) - visible_width(right))
        return [truncate_to_width(left + pad + right, width)]


def extension(pi):
    state = {"enabled": False}

    async def toggle_footer(_args, ctx) -> None:
        state["enabled"] = not state["enabled"]

        if state["enabled"]:
            ctx.ui.set_footer(lambda ui, theme, footer_data: CustomFooter(ui, theme, footer_data, ctx))
            ctx.ui.notify("Custom footer enabled", "info")
        else:
            ctx.ui.set_footer(None)
            ctx.ui.notify("Default footer restored", "info")

    pi.register_command("footer", handler=toggle_footer, description="Toggle custom footer")
