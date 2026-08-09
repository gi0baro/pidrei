"""Mirror of pi tui src/keybindings.ts.

pi's ``Keybindings`` interface is a compile-time registry extended by
downstream packages through declaration merging; here keybinding names are
plain strings validated against the definitions dict at rebuild time.
Definitions are camelCase records: ``{"defaultKeys": KeyId | [KeyId],
"description": str}``; conflicts are ``{"key": KeyId, "keybindings": [str]}``.
"""

from .keys import matches_key


TUI_KEYBINDINGS = {
    "tui.editor.cursorUp": {"defaultKeys": "up", "description": "Move cursor up"},
    "tui.editor.cursorDown": {"defaultKeys": "down", "description": "Move cursor down"},
    "tui.editor.historyPrevious": {
        "defaultKeys": [],
        "description": "Select previous prompt history entry",
    },
    "tui.editor.historyNext": {
        "defaultKeys": [],
        "description": "Select next prompt history entry",
    },
    "tui.editor.cursorLeft": {
        "defaultKeys": ["left", "ctrl+b"],
        "description": "Move cursor left",
    },
    "tui.editor.cursorRight": {
        "defaultKeys": ["right", "ctrl+f"],
        "description": "Move cursor right",
    },
    "tui.editor.cursorWordLeft": {
        "defaultKeys": ["alt+left", "ctrl+left", "alt+b"],
        "description": "Move cursor word left",
    },
    "tui.editor.cursorWordRight": {
        "defaultKeys": ["alt+right", "ctrl+right", "alt+f"],
        "description": "Move cursor word right",
    },
    "tui.editor.cursorLineStart": {
        "defaultKeys": ["home", "ctrl+home", "ctrl+a"],
        "description": "Move to line start",
    },
    "tui.editor.cursorLineEnd": {
        "defaultKeys": ["end", "ctrl+end", "ctrl+e"],
        "description": "Move to line end",
    },
    "tui.editor.jumpForward": {
        "defaultKeys": "ctrl+]",
        "description": "Jump forward to character",
    },
    "tui.editor.jumpBackward": {
        "defaultKeys": "ctrl+alt+]",
        "description": "Jump backward to character",
    },
    "tui.editor.pageUp": {"defaultKeys": ["pageUp", "ctrl+pageUp"], "description": "Page up"},
    "tui.editor.pageDown": {"defaultKeys": ["pageDown", "ctrl+pageDown"], "description": "Page down"},
    "tui.editor.deleteCharBackward": {
        "defaultKeys": "backspace",
        "description": "Delete character backward",
    },
    "tui.editor.deleteCharForward": {
        "defaultKeys": ["delete", "ctrl+d"],
        "description": "Delete character forward",
    },
    "tui.editor.deleteWordBackward": {
        "defaultKeys": ["ctrl+w", "alt+backspace"],
        "description": "Delete word backward",
    },
    "tui.editor.deleteWordForward": {
        "defaultKeys": ["alt+d", "alt+delete"],
        "description": "Delete word forward",
    },
    "tui.editor.deleteToLineStart": {
        "defaultKeys": "ctrl+u",
        "description": "Delete to line start",
    },
    "tui.editor.deleteToLineEnd": {
        "defaultKeys": "ctrl+k",
        "description": "Delete to line end",
    },
    "tui.editor.yank": {"defaultKeys": "ctrl+y", "description": "Yank"},
    "tui.editor.yankPop": {"defaultKeys": "alt+y", "description": "Yank pop"},
    "tui.editor.undo": {"defaultKeys": "ctrl+-", "description": "Undo"},
    "tui.input.newLine": {"defaultKeys": ["shift+enter", "ctrl+j"], "description": "Insert newline"},
    "tui.input.submit": {"defaultKeys": "enter", "description": "Submit input"},
    "tui.input.tab": {"defaultKeys": "tab", "description": "Tab / autocomplete"},
    "tui.input.copy": {"defaultKeys": "ctrl+c", "description": "Copy selection"},
    "tui.select.up": {"defaultKeys": "up", "description": "Move selection up"},
    "tui.select.down": {"defaultKeys": "down", "description": "Move selection down"},
    "tui.select.pageUp": {"defaultKeys": "pageUp", "description": "Selection page up"},
    "tui.select.pageDown": {
        "defaultKeys": "pageDown",
        "description": "Selection page down",
    },
    "tui.select.confirm": {"defaultKeys": "enter", "description": "Confirm selection"},
    "tui.select.cancel": {
        "defaultKeys": ["escape", "ctrl+c"],
        "description": "Cancel selection",
    },
    # Alternate-screen viewport navigation.
    # These intentionally shadow the unmodified editor bindings in fullscreen mode.
    "tui.altScreen.pageUp": {"defaultKeys": "pageUp", "description": "Scroll viewport up one page"},
    "tui.altScreen.pageDown": {"defaultKeys": "pageDown", "description": "Scroll viewport down one page"},
    "tui.altScreen.halfPageUp": {
        "defaultKeys": [],
        "description": "Scroll viewport up half a page",
    },
    "tui.altScreen.halfPageDown": {
        "defaultKeys": [],
        "description": "Scroll viewport down half a page",
    },
    "tui.altScreen.previousPrompt": {
        "defaultKeys": "ctrl+shift+up",
        "description": "Jump to previous semantic prompt",
    },
    "tui.altScreen.nextPrompt": {
        "defaultKeys": "ctrl+shift+down",
        "description": "Jump to next semantic prompt",
    },
    "tui.altScreen.top": {"defaultKeys": "home", "description": "Scroll viewport to top"},
    "tui.altScreen.bottom": {"defaultKeys": "end", "description": "Scroll viewport to bottom"},
}


def _normalize_keys(keys) -> list[str]:
    if keys is None:
        return []
    key_list = keys if isinstance(keys, list) else [keys]
    seen = set()
    result: list[str] = []
    for key in key_list:
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


class KeybindingsManager:
    def __init__(self, definitions: dict, user_bindings: dict | None = None) -> None:
        self._definitions = definitions
        self._user_bindings = user_bindings if user_bindings is not None else {}
        self._keys_by_id: dict[str, list[str]] = {}
        self._conflicts: list[dict] = []
        self._rebuild()

    def _rebuild(self) -> None:
        self._keys_by_id = {}
        self._conflicts = []

        # Insertion-ordered claimant lists (pi uses Map<KeyId, Set<Keybinding>>;
        # both preserve insertion order, Python sets would not).
        user_claims: dict[str, list[str]] = {}
        for keybinding, keys in self._user_bindings.items():
            if keybinding not in self._definitions:
                continue
            for key in _normalize_keys(keys):
                claimants = user_claims.setdefault(key, [])
                if keybinding not in claimants:
                    claimants.append(keybinding)

        for key, keybindings in user_claims.items():
            if len(keybindings) > 1:
                self._conflicts.append({"key": key, "keybindings": list(keybindings)})

        for binding_id, definition in self._definitions.items():
            user_keys = self._user_bindings.get(binding_id)
            keys = _normalize_keys(definition["defaultKeys"]) if user_keys is None else _normalize_keys(user_keys)
            self._keys_by_id[binding_id] = keys

    def matches(self, data: str, keybinding: str) -> bool:
        keys = self._keys_by_id.get(keybinding) or []
        return any(matches_key(data, key) for key in keys)

    def get_keys(self, keybinding: str) -> list[str]:
        return list(self._keys_by_id.get(keybinding) or [])

    def get_definition(self, keybinding: str) -> dict:
        return self._definitions[keybinding]

    def get_conflicts(self) -> list[dict]:
        return [{**conflict, "keybindings": list(conflict["keybindings"])} for conflict in self._conflicts]

    def set_user_bindings(self, user_bindings: dict) -> None:
        self._user_bindings = user_bindings
        self._rebuild()

    def get_user_bindings(self) -> dict:
        return dict(self._user_bindings)

    def get_resolved_bindings(self) -> dict:
        resolved: dict = {}
        for binding_id in self._definitions:
            keys = self._keys_by_id.get(binding_id) or []
            resolved[binding_id] = keys[0] if len(keys) == 1 else list(keys)
        return resolved


_global_keybindings: KeybindingsManager | None = None


def set_keybindings(keybindings: KeybindingsManager) -> None:
    global _global_keybindings
    _global_keybindings = keybindings


def get_keybindings() -> KeybindingsManager:
    global _global_keybindings
    if _global_keybindings is None:
        _global_keybindings = KeybindingsManager(TUI_KEYBINDINGS)
    return _global_keybindings
