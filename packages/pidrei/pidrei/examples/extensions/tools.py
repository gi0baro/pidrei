"""Tools Extension

Provides a /tools command to enable/disable tools interactively. Tool
selection persists across session reloads (as a custom session entry) and
respects branch navigation: switching branches restores the selection saved
on that branch.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/tools.py
"""

from pidrei.modes.interactive.theme import get_settings_list_theme
from pidrei_tui import Container, SettingsList, Spacer, Text


class ToolsExtension:
    def __init__(self, pi):
        self.pi = pi
        self.enabled_tools: set[str] = set()
        self.all_tools: list = []

    def wire(self) -> None:
        self.pi.register_command("tools", description="Enable/disable tools", handler=self.tools_command)
        self.pi.on("session_start", self.restore)
        # Restore state when navigating the session tree.
        self.pi.on("session_tree", self.restore)

    async def persist_state(self) -> None:
        await self.pi.append_entry("tools-config", {"enabledTools": sorted(self.enabled_tools)})

    def apply_tools(self) -> None:
        self.pi.set_active_tools(list(self.enabled_tools))

    async def restore(self, _event, ctx) -> None:
        """Restore the last tools-config entry in the current branch."""
        self.all_tools = self.pi.get_all_tools()

        # Get entries in the current branch only.
        saved_tools: list[str] | None = None
        for entry in ctx.session_manager.get_branch():
            if entry.get("type") == "custom" and entry.get("customType") == "tools-config":
                data = entry.get("data") or {}
                if data.get("enabledTools") is not None:
                    saved_tools = data["enabledTools"]

        if saved_tools is not None:
            # Restore the saved selection, filtered to tools that still exist.
            all_tool_names = {tool.name for tool in self.all_tools}
            self.enabled_tools = {name for name in saved_tools if name in all_tool_names}
            self.apply_tools()
        else:
            # No saved state - sync with the currently active tools.
            self.enabled_tools = set(self.pi.get_active_tools())

    async def tools_command(self, _args: str, ctx) -> None:
        if ctx.mode != "tui":
            ctx.ui.notify("/tools requires TUI mode", "error")
            return

        # Refresh the tool list.
        self.all_tools = self.pi.get_all_tools()

        def factory(_tui, theme, _kb, done):
            # A settings row per tool, toggling between enabled and disabled.
            items = [
                {
                    "id": tool.name,
                    "label": tool.name,
                    "currentValue": "enabled" if tool.name in self.enabled_tools else "disabled",
                    "values": ["enabled", "disabled"],
                }
                for tool in self.all_tools
            ]

            container = Container()
            container.add_child(Text(theme.fg("accent", theme.bold("Tool Configuration")), 1, 0))
            container.add_child(Spacer(1))

            async def on_change(tool_id: str, new_value: str) -> None:
                # Update the enabled state and apply immediately.
                if new_value == "enabled":
                    self.enabled_tools.add(tool_id)
                else:
                    self.enabled_tools.discard(tool_id)
                self.apply_tools()
                await self.persist_state()

            async def on_cancel() -> None:
                # Close the dialog.
                done(None)

            settings_list = SettingsList(
                items, min(len(items) + 2, 15), get_settings_list_theme(), on_change, on_cancel
            )
            container.add_child(settings_list)

            class ToolsDialog:
                def render(self, width: int) -> list[str]:
                    return container.render(width)

                def invalidate(self) -> None:
                    container.invalidate()

                async def handle_input(self, data: str) -> None:
                    await settings_list.handle_input(data)

            return ToolsDialog()

        await ctx.ui.custom(factory)


def extension(pi):
    ToolsExtension(pi).wire()
