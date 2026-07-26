"""Mirror of pi coding-agent src/core/keybindings.ts.

App-level keybinding definitions layered over the tui defaults, plus the
legacy config-name migration and the file-backed KeybindingsManager.
POSIX-only: pi's win32 default-key variants are not ported.
"""

import json
import os
import sys

from pidrei_tui.keybindings import TUI_KEYBINDINGS, KeybindingsManager as TuiKeybindingsManager

from ..config import get_agent_dir


_DARWIN = sys.platform == "darwin"

KEYBINDINGS = {
    **TUI_KEYBINDINGS,
    "app.interrupt": {"defaultKeys": "escape", "description": "Cancel or abort"},
    "app.clear": {"defaultKeys": "ctrl+c", "description": "Clear editor"},
    "app.exit": {"defaultKeys": "ctrl+d", "description": "Exit when editor is empty"},
    "app.suspend": {"defaultKeys": "ctrl+z", "description": "Suspend to background"},
    "app.thinking.cycle": {"defaultKeys": "shift+tab", "description": "Cycle thinking level"},
    "app.model.cycleForward": {"defaultKeys": "ctrl+p", "description": "Cycle to next model"},
    "app.model.cycleBackward": {"defaultKeys": "shift+ctrl+p", "description": "Cycle to previous model"},
    "app.model.select": {"defaultKeys": "ctrl+l", "description": "Open model selector"},
    "app.tools.expand": {"defaultKeys": "ctrl+o", "description": "Toggle tool output"},
    "app.thinking.toggle": {"defaultKeys": "ctrl+t", "description": "Toggle thinking blocks"},
    "app.session.toggleNamedFilter": {"defaultKeys": "ctrl+n", "description": "Toggle named session filter"},
    "app.editor.external": {"defaultKeys": "ctrl+g", "description": "Open external editor"},
    "app.message.copy": {"defaultKeys": "ctrl+x", "description": "Copy message to clipboard"},
    "app.message.followUp": {"defaultKeys": "alt+enter", "description": "Queue follow-up message"},
    "app.message.dequeue": {"defaultKeys": "alt+up", "description": "Restore queued messages"},
    "app.clipboard.pasteImage": {
        "defaultKeys": "ctrl+v",
        "description": "Paste image from clipboard (text fallback)",
    },
    "app.session.new": {"defaultKeys": [], "description": "Start a new session"},
    "app.session.tree": {"defaultKeys": [], "description": "Open session tree"},
    "app.session.fork": {"defaultKeys": [], "description": "Fork current session"},
    "app.session.resume": {"defaultKeys": [], "description": "Resume a session"},
    "app.tree.foldOrUp": {
        "defaultKeys": ["alt+left", "ctrl+left"] if _DARWIN else ["ctrl+left", "alt+left"],
        "description": "Fold tree branch or move up",
    },
    "app.tree.unfoldOrDown": {
        "defaultKeys": ["alt+right", "ctrl+right"] if _DARWIN else ["ctrl+right", "alt+right"],
        "description": "Unfold tree branch or move down",
    },
    "app.tree.editLabel": {"defaultKeys": "shift+l", "description": "Edit tree label"},
    "app.tree.toggleLabelTimestamp": {"defaultKeys": "shift+t", "description": "Toggle tree label timestamps"},
    "app.session.togglePath": {"defaultKeys": "ctrl+p", "description": "Toggle session path display"},
    "app.session.toggleSort": {"defaultKeys": "ctrl+s", "description": "Toggle session sort mode"},
    "app.session.rename": {"defaultKeys": "ctrl+r", "description": "Rename session"},
    "app.session.delete": {"defaultKeys": "ctrl+d", "description": "Delete session"},
    "app.session.deleteNoninvasive": {
        "defaultKeys": "ctrl+backspace",
        "description": "Delete session when query is empty",
    },
    "app.models.save": {"defaultKeys": "ctrl+s", "description": "Save model selection"},
    "app.models.enableAll": {"defaultKeys": "ctrl+a", "description": "Enable all models"},
    "app.models.clearAll": {"defaultKeys": "ctrl+x", "description": "Clear all models"},
    "app.models.toggleProvider": {"defaultKeys": "ctrl+p", "description": "Toggle all models for provider"},
    "app.models.reorderUp": {"defaultKeys": "alt+up", "description": "Move model up in order"},
    "app.models.reorderDown": {"defaultKeys": "alt+down", "description": "Move model down in order"},
    "app.tree.filter.default": {"defaultKeys": "ctrl+d", "description": "Tree filter: default view"},
    "app.tree.filter.noTools": {"defaultKeys": "ctrl+t", "description": "Tree filter: hide tool results"},
    "app.tree.filter.userOnly": {"defaultKeys": "ctrl+u", "description": "Tree filter: user messages only"},
    "app.tree.filter.labeledOnly": {"defaultKeys": "ctrl+l", "description": "Tree filter: labeled entries only"},
    "app.tree.filter.all": {"defaultKeys": "ctrl+a", "description": "Tree filter: show all entries"},
    "app.tree.filter.cycleForward": {"defaultKeys": "ctrl+o", "description": "Tree filter: cycle forward"},
    "app.tree.filter.cycleBackward": {"defaultKeys": "shift+ctrl+o", "description": "Tree filter: cycle backward"},
}

KEYBINDING_NAME_MIGRATIONS = {
    "cursorUp": "tui.editor.cursorUp",
    "cursorDown": "tui.editor.cursorDown",
    "cursorLeft": "tui.editor.cursorLeft",
    "cursorRight": "tui.editor.cursorRight",
    "cursorWordLeft": "tui.editor.cursorWordLeft",
    "cursorWordRight": "tui.editor.cursorWordRight",
    "cursorLineStart": "tui.editor.cursorLineStart",
    "cursorLineEnd": "tui.editor.cursorLineEnd",
    "jumpForward": "tui.editor.jumpForward",
    "jumpBackward": "tui.editor.jumpBackward",
    "pageUp": "tui.editor.pageUp",
    "pageDown": "tui.editor.pageDown",
    "deleteCharBackward": "tui.editor.deleteCharBackward",
    "deleteCharForward": "tui.editor.deleteCharForward",
    "deleteWordBackward": "tui.editor.deleteWordBackward",
    "deleteWordForward": "tui.editor.deleteWordForward",
    "deleteToLineStart": "tui.editor.deleteToLineStart",
    "deleteToLineEnd": "tui.editor.deleteToLineEnd",
    "yank": "tui.editor.yank",
    "yankPop": "tui.editor.yankPop",
    "undo": "tui.editor.undo",
    "newLine": "tui.input.newLine",
    "submit": "tui.input.submit",
    "tab": "tui.input.tab",
    "copy": "tui.input.copy",
    "selectUp": "tui.select.up",
    "selectDown": "tui.select.down",
    "selectPageUp": "tui.select.pageUp",
    "selectPageDown": "tui.select.pageDown",
    "selectConfirm": "tui.select.confirm",
    "selectCancel": "tui.select.cancel",
    "interrupt": "app.interrupt",
    "clear": "app.clear",
    "exit": "app.exit",
    "suspend": "app.suspend",
    "cycleThinkingLevel": "app.thinking.cycle",
    "cycleModelForward": "app.model.cycleForward",
    "cycleModelBackward": "app.model.cycleBackward",
    "selectModel": "app.model.select",
    "expandTools": "app.tools.expand",
    "toggleThinking": "app.thinking.toggle",
    "toggleSessionNamedFilter": "app.session.toggleNamedFilter",
    "externalEditor": "app.editor.external",
    "followUp": "app.message.followUp",
    "dequeue": "app.message.dequeue",
    "pasteImage": "app.clipboard.pasteImage",
    "newSession": "app.session.new",
    "tree": "app.session.tree",
    "fork": "app.session.fork",
    "resume": "app.session.resume",
    "treeFoldOrUp": "app.tree.foldOrUp",
    "treeUnfoldOrDown": "app.tree.unfoldOrDown",
    "treeEditLabel": "app.tree.editLabel",
    "treeToggleLabelTimestamp": "app.tree.toggleLabelTimestamp",
    "toggleSessionPath": "app.session.togglePath",
    "toggleSessionSort": "app.session.toggleSort",
    "renameSession": "app.session.rename",
    "deleteSession": "app.session.delete",
    "deleteSessionNoninvasive": "app.session.deleteNoninvasive",
}


def _to_keybindings_config(value: dict) -> dict:
    config: dict = {}
    for key, binding in value.items():
        if isinstance(binding, str):
            config[key] = binding
            continue
        if isinstance(binding, list) and all(isinstance(entry, str) for entry in binding):
            config[key] = binding
    return config


def migrate_keybindings_config(raw_config: dict) -> dict:
    """Rename legacy keybinding names; returns ``{"config", "migrated"}``."""
    config: dict = {}
    migrated = False

    for key, value in raw_config.items():
        next_key = KEYBINDING_NAME_MIGRATIONS.get(key, key)
        if next_key != key:
            migrated = True
        if key != next_key and next_key in raw_config:
            migrated = True
            continue
        config[next_key] = value

    return {"config": _order_keybindings_config(config), "migrated": migrated}


def _order_keybindings_config(config: dict) -> dict:
    ordered: dict = {}
    for keybinding in KEYBINDINGS:
        if keybinding in config:
            ordered[keybinding] = config[keybinding]

    for key in sorted(key for key in config if key not in ordered):
        ordered[key] = config[key]

    return ordered


def _load_raw_config(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            parsed = json.load(f)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


class KeybindingsManager(TuiKeybindingsManager):
    def __init__(self, user_bindings: dict | None = None, config_path: str | None = None) -> None:
        super().__init__(KEYBINDINGS, user_bindings or {})
        self._config_path = config_path

    @staticmethod
    def create(agent_dir: str | None = None) -> KeybindingsManager:
        if agent_dir is None:
            agent_dir = get_agent_dir()
        config_path = os.path.join(agent_dir, "keybindings.json")
        user_bindings = KeybindingsManager._load_from_file(config_path)
        return KeybindingsManager(user_bindings, config_path)

    def reload(self) -> None:
        if not self._config_path:
            return
        self.set_user_bindings(KeybindingsManager._load_from_file(self._config_path))

    def get_effective_config(self) -> dict:
        return self.get_resolved_bindings()

    @staticmethod
    def _load_from_file(path: str) -> dict:
        raw_config = _load_raw_config(path)
        if not raw_config:
            return {}
        return _to_keybindings_config(migrate_keybindings_config(raw_config)["config"])
