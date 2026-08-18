"""Preset Extension

Named presets that configure model, thinking level, tools, and system prompt
instructions. Presets are defined in JSON config files and can be activated
via CLI flag, /preset command, or Ctrl+Shift+U to cycle.

Config files (merged, project takes precedence):
- ~/.pidrei/agent/presets.json (global)
- <cwd>/.pidrei/presets.json (project-local)

Example presets.json:
    {
      "plan": {
        "provider": "openai-codex",
        "model": "gpt-5.2-codex",
        "thinkingLevel": "high",
        "tools": ["read", "grep", "find", "ls"],
        "instructions": "You are in PLANNING MODE. Do not make any changes..."
      },
      "implement": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5",
        "thinkingLevel": "high",
        "tools": ["read", "bash", "edit", "write"],
        "instructions": "You are in IMPLEMENTATION MODE. Keep scope tight..."
      }
    }

Preset fields (all optional): "provider" and "model" select a model,
"thinkingLevel" is one of off/minimal/low/medium/high/xhigh/max, "tools"
replaces the active tool set, "instructions" is appended to the system prompt.

Usage:
- `pidrei --preset plan` - start with the plan preset
- `/preset` - show a selector to switch presets mid-session
- `/preset implement` - switch to a preset directly
- `Ctrl+Shift+U` - cycle through presets

CLI flags always override preset values.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/preset.py
"""

import json
import os

from tonio.colored import fs

from pidrei.config import CONFIG_DIR_NAME, get_agent_dir
from pidrei.modes.interactive.components import DynamicBorder
from pidrei_tui import Container, SelectList, Text
from pidrei_tui.keys import Key


DEFAULT_TOOLS = ["read", "bash", "edit", "write"]


async def _read_presets_file(path: str, ctx) -> dict:
    if not await fs.Path(path).exists():
        return {}
    try:
        return json.loads(await fs.Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        ctx.ui.notify(f"Failed to load presets from {path}: {error}", "warning")
        return {}


class PresetExtension:
    """Closure in pi; enough state (presets, active preset, pre-preset
    snapshot) that an object reads better in Python."""

    def __init__(self, pi):
        self.pi = pi
        self.presets: dict[str, dict] = {}
        self.active_name: str | None = None
        self.active_preset: dict | None = None
        # Snapshot of model/thinking/tools taken before the first preset is
        # applied, so "(none)" can restore it.
        self.original_state: dict | None = None

    def wire(self) -> None:
        self.pi.register_flag("preset", type="string", description="Preset configuration to use")
        self.pi.register_command("preset", description="Switch preset configuration", handler=self.preset_command)
        self.pi.register_shortcut(Key.ctrl_shift("u"), description="Cycle presets", handler=self.cycle_preset)
        self.pi.on("before_agent_start", self.on_before_agent_start)
        self.pi.on("session_start", self.on_session_start)
        self.pi.on("turn_start", self.on_turn_start)

    # -- config ------------------------------------------------------------------

    def config_paths(self, cwd: str) -> tuple[str, str]:
        return (
            os.path.join(get_agent_dir(), "presets.json"),
            os.path.join(cwd, CONFIG_DIR_NAME, "presets.json"),
        )

    async def load_presets(self, ctx) -> None:
        """Load presets from config files; project-local presets override
        global presets with the same name."""
        global_path, project_path = self.config_paths(ctx.cwd)
        global_presets = await _read_presets_file(global_path, ctx)
        project_presets = await _read_presets_file(project_path, ctx)
        self.presets = {**global_presets, **project_presets}

    # -- apply / clear -----------------------------------------------------------

    async def apply_preset(self, name: str, preset: dict, ctx) -> None:
        # Snapshot state only when transitioning from no-preset.
        if self.active_name is None:
            self.original_state = {
                "model": ctx.model,
                "thinking_level": self.pi.get_thinking_level(),
                "tools": self.pi.get_active_tools(),
            }

        provider = preset.get("provider")
        model_id = preset.get("model")
        if provider and model_id:
            model = ctx.model_registry.find(provider, model_id)
            if model is not None:
                if not await self.pi.set_model(model):
                    ctx.ui.notify(f'Preset "{name}": No API key for {provider}/{model_id}', "warning")
            else:
                ctx.ui.notify(f'Preset "{name}": Model {provider}/{model_id} not found', "warning")

        if preset.get("thinkingLevel"):
            await self.pi.set_thinking_level(preset["thinkingLevel"])

        tools = preset.get("tools")
        if tools:
            all_tool_names = [tool.name for tool in self.pi.get_all_tools()]
            valid_tools = [tool for tool in tools if tool in all_tool_names]
            invalid_tools = [tool for tool in tools if tool not in all_tool_names]

            if invalid_tools:
                ctx.ui.notify(f'Preset "{name}": Unknown tools: {", ".join(invalid_tools)}', "warning")
            if valid_tools:
                self.pi.set_active_tools(valid_tools)

        # Store active preset for system prompt injection.
        self.active_name = name
        self.active_preset = preset

    async def activate(self, name: str, ctx) -> None:
        await self.apply_preset(name, self.presets[name], ctx)
        ctx.ui.notify(f'Preset "{name}" activated', "info")
        self.update_status(ctx)

    async def clear_preset(self, ctx) -> None:
        """Clear the active preset and restore the pre-preset state."""
        self.active_name = None
        self.active_preset = None
        if self.original_state is not None:
            if self.original_state["model"] is not None:
                await self.pi.set_model(self.original_state["model"])
            await self.pi.set_thinking_level(self.original_state["thinking_level"])
            self.pi.set_active_tools(self.original_state["tools"])
        else:
            self.pi.set_active_tools(DEFAULT_TOOLS)
        ctx.ui.notify("Preset cleared, defaults restored", "info")
        self.update_status(ctx)

    def notify_no_presets(self, ctx) -> None:
        global_path, project_path = self.config_paths(ctx.cwd)
        ctx.ui.notify(f"No presets defined. Add presets to {global_path} or {project_path}", "warning")

    # -- UI ----------------------------------------------------------------------

    def build_preset_description(self, preset: dict) -> str:
        parts: list[str] = []
        if preset.get("provider") and preset.get("model"):
            parts.append(f"{preset['provider']}/{preset['model']}")
        if preset.get("thinkingLevel"):
            parts.append(f"thinking:{preset['thinkingLevel']}")
        if preset.get("tools"):
            parts.append(f"tools:{','.join(preset['tools'])}")
        instructions = preset.get("instructions")
        if instructions:
            truncated = f"{instructions[:27]}..." if len(instructions) > 30 else instructions
            parts.append(f'"{truncated}"')
        return " | ".join(parts)

    async def show_preset_selector(self, ctx) -> None:
        """Show the preset selector using a custom SelectList component."""
        if not self.presets:
            self.notify_no_presets(ctx)
            return

        items = [
            {
                "value": name,
                "label": f"{name} (active)" if name == self.active_name else name,
                "description": self.build_preset_description(preset),
            }
            for name, preset in self.presets.items()
        ]
        items.append({"value": "(none)", "label": "(none)", "description": "Clear active preset, restore defaults"})

        def factory(_tui, theme, _kb, done):
            def accent(text: str) -> str:
                return theme.fg("accent", text)

            container = Container()
            container.add_child(DynamicBorder(accent))
            container.add_child(Text(accent(theme.bold("Select Preset")), 1, 0))

            select_list = SelectList(
                items,
                min(len(items), 10),
                {
                    "selectedPrefix": accent,
                    "selectedText": accent,
                    "description": lambda text: theme.fg("muted", text),
                    "scrollInfo": lambda text: theme.fg("dim", text),
                    "noMatch": lambda text: theme.fg("warning", text),
                },
            )

            async def on_select(item) -> None:
                done(item["value"])

            async def on_cancel() -> None:
                done(None)

            select_list.on_select = on_select
            select_list.on_cancel = on_cancel
            container.add_child(select_list)

            container.add_child(Text(theme.fg("dim", "↑↓ navigate • enter select • esc cancel"), 1, 0))
            container.add_child(DynamicBorder(accent))

            class Selector:
                def render(self, width: int) -> list[str]:
                    return container.render(width)

                def invalidate(self) -> None:
                    container.invalidate()

                # The TUI re-renders after focused input, so delegating is all
                # that is needed here.
                async def handle_input(self, data: str) -> None:
                    await select_list.handle_input(data)

            return Selector()

        result = await ctx.ui.custom(factory)
        if not result:
            return

        if result == "(none)":
            await self.clear_preset(ctx)
        elif result in self.presets:
            await self.activate(result, ctx)

    def update_status(self, ctx) -> None:
        if self.active_name is not None:
            ctx.ui.set_status("preset", ctx.ui.theme.fg("accent", f"preset:{self.active_name}"))
        else:
            ctx.ui.set_status("preset", None)

    # -- command / shortcut ------------------------------------------------------

    async def cycle_preset(self, ctx) -> None:
        preset_names = sorted(self.presets)
        if not preset_names:
            self.notify_no_presets(ctx)
            return

        cycle_list = ["(none)", *preset_names]
        current_name = self.active_name if self.active_name is not None else "(none)"
        current_index = cycle_list.index(current_name) if current_name in cycle_list else -1
        next_name = cycle_list[0] if current_index == -1 else cycle_list[(current_index + 1) % len(cycle_list)]

        if next_name == "(none)":
            await self.clear_preset(ctx)
        else:
            await self.activate(next_name, ctx)

    async def preset_command(self, args: str, ctx) -> None:
        # If a preset name was provided, apply directly.
        name = args.strip() if args else ""
        if name:
            if name not in self.presets:
                available = ", ".join(self.presets) or "(none defined)"
                ctx.ui.notify(f'Unknown preset "{name}". Available: {available}', "error")
                return
            await self.activate(name, ctx)
            return

        # Otherwise show the selector.
        await self.show_preset_selector(ctx)

    # -- events ------------------------------------------------------------------

    async def on_before_agent_start(self, event, _ctx):
        # Inject preset instructions into the system prompt.
        if self.active_preset is not None and self.active_preset.get("instructions"):
            return {"systemPrompt": f"{event['systemPrompt']}\n\n{self.active_preset['instructions']}"}

    async def on_session_start(self, _event, ctx) -> None:
        await self.load_presets(ctx)

        # Check for the --preset flag.
        preset_flag = self.pi.get_flag("preset")
        if isinstance(preset_flag, str) and preset_flag:
            if preset_flag in self.presets:
                await self.apply_preset(preset_flag, self.presets[preset_flag], ctx)
                ctx.ui.notify(f'Preset "{preset_flag}" activated', "info")
            else:
                available = ", ".join(self.presets) or "(none defined)"
                ctx.ui.notify(f'Unknown preset "{preset_flag}". Available: {available}', "warning")

        # Restore the preset from session state.
        preset_entries = [
            entry
            for entry in ctx.session_manager.get_entries()
            if entry.get("type") == "custom" and entry.get("customType") == "preset-state"
        ]
        if preset_entries and not preset_flag:
            name = (preset_entries[-1].get("data") or {}).get("name")
            if name in self.presets:
                # Don't re-apply model/tools on restore, just keep the name
                # for instructions.
                self.active_name = name
                self.active_preset = self.presets[name]

        self.update_status(ctx)

    async def on_turn_start(self, _event, _ctx) -> None:
        # Persist preset state.
        if self.active_name is not None:
            await self.pi.append_entry("preset-state", {"name": self.active_name})


def extension(pi):
    PresetExtension(pi).wire()
