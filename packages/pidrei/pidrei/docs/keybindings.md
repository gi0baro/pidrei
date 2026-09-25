# Keybindings

Every binding has an **action id** (`app.model.select`) and a set of keys. Show
the active shortcuts for the editor and application with `/hotkeys`; rebind
them in `~/.pidrei/agent/keybindings.json` (see [Customizing](#customizing)).

## Defaults

### Global

| Keys | Action | Does |
|------|--------|------|
| `escape` | `app.interrupt` | Cancel or abort |
| `ctrl+c` | `app.clear` | Clear the editor (first press) / exit (second) |
| `ctrl+d` | `app.exit` | Exit when the editor is empty |
| `ctrl+z` | `app.suspend` | Suspend to background |
| `shift+tab` | `app.thinking.cycle` | Cycle thinking level |
| `ctrl+s` | `app.thinking.save` | Save the current thinking level to settings (inside the thinking selector) |
| `ctrl+s` | `app.models.save` | Save the selected default model or scoped model configuration to settings (inside the model selectors) |
| `ctrl+t` | `app.thinking.toggle` | Show or hide thinking blocks |
| `ctrl+p` / `shift+ctrl+p` | `app.model.cycleForward` / `Backward` | Cycle model |
| `ctrl+l` | `app.model.select` | Open the model selector |
| `ctrl+o` | `app.tools.expand` | Expand or collapse tool output |
| `ctrl+g` | `app.editor.external` | Open the external editor (`externalEditor` setting, `$VISUAL`, `$EDITOR`, else `nano`) |
| `ctrl+v` | `app.clipboard.pasteImage` | Paste an image, or text, from the clipboard |
| `ctrl+x` | `app.message.copy` | Copy the selected message in `/tree`; in fullscreen mode, copy the active selection when `fullscreenCopyOnSelect` is `false`; otherwise copy the last assistant message |
| `alt+enter` | `app.message.followUp` | Queue a follow-up message |
| `alt+up` | `app.message.dequeue` | Restore queued messages |
| `ctrl+n` | `app.session.toggleNamedFilter` | Toggle the named-session filter |

`app.session.new`, `.tree`, `.fork` and `.resume` have no default key; bind them
if you want them.

Under WSL the Windows terminal swallows several of these chords, so the
defaults shift there: `alt+p` cycles to the previous model, `ctrl+q` queues a
follow-up and `alt+q` restores queued messages, `alt+v` pastes an image,
`alt+z` undoes an edit (leaving `ctrl+z` free to suspend), and in the
alternate screen `ctrl+f` searches while `ctrl+up`/`ctrl+down` jump between
marked messages. Every one of them is still just a binding, so
`keybindings.json` overrides it.

### Prompt history

`up`/`down` (`tui.editor.cursorUp`/`.cursorDown`) move the cursor and browse
history at the first and last line. `tui.editor.historyPrevious` and
`tui.editor.historyNext` are unbound by default and always change history
entries, wherever the cursor sits in a multiline prompt. An explicit history
binding beats an application action while the main editor is focused, so
binding `tui.editor.historyPrevious` to `ctrl+p` overrides model cycling there
without changing `ctrl+p` in selectors.

### Session tree

| Keys | Action |
|------|--------|
| `left` / `right` | Page up / down |
| `ctrl+left`, `alt+left` / `ctrl+right`, `alt+right` | Fold the branch segment or jump to the previous segment start / unfold or jump to the next (`app.tree.foldOrUp` / `.unfoldOrDown`) |
| `shift+l` | Edit the entry label |
| `shift+t` | Toggle label timestamps |
| `ctrl+d` / `ctrl+t` / `ctrl+u` / `ctrl+l` / `ctrl+a` | Filter: default, no tools, user only, labeled only, all |
| `ctrl+o` / `shift+ctrl+o` | Cycle the filter forward / backward |

### Session and model selectors

| Keys | Action |
|------|--------|
| `ctrl+p` / `ctrl+s` / `ctrl+r` / `ctrl+d` | Toggle path, toggle sort, rename, delete |
| `ctrl+s` / `ctrl+a` / `ctrl+x` / `ctrl+p` | Save, enable all, clear all, toggle provider |
| `alt+up` / `alt+down` | Reorder a model |

The same keys mean different things in different surfaces — bindings are scoped
to the component that has focus.

### Alternate-screen viewport

Only in `--tui-mode fullscreen`, and targeting the primary transcript scroll
region.
Two-finger trackpad and mouse-wheel input scroll the region under the pointer,
falling back to the transcript over the fixed editor/status/footer dock.
Clicking an OSC 8 hyperlink opens it in the default handler. Dragging with the
primary mouse button selects text and copies it to the clipboard; holding at
the transcript's top or bottom edge auto-scrolls into off-screen content.

| Keys | Action | Does |
|------|--------|------|
| `pageUp` | `tui.altScreen.pageUp` | Scroll the transcript up by one page |
| `pageDown` | `tui.altScreen.pageDown` | Scroll the transcript down by one page |
| *(none)* | `tui.altScreen.halfPageUp` | Scroll the transcript up by half a page |
| *(none)* | `tui.altScreen.halfPageDown` | Scroll the transcript down by half a page |
| *(none)* | `tui.altScreen.lineUp` | Scroll the transcript up by one line |
| *(none)* | `tui.altScreen.lineDown` | Scroll the transcript down by one line |
| `ctrl+shift+up`, `ctrl+up` | `tui.altScreen.previousPrompt` | Jump to the previous marked message |
| `ctrl+shift+down`, `ctrl+down` | `tui.altScreen.nextPrompt` | Jump to the next marked message |
| `ctrl+shift+f` | `tui.altScreen.search` | Open or close the transcript search panel (it shows the previous/next shortcuts and clickable arrow controls) |
| `enter`, `ctrl+g` | `tui.altScreen.searchNext` | Select the next search match while searching |
| `shift+enter`, `ctrl+shift+g` | `tui.altScreen.searchPrevious` | Select the previous search match while searching |
| `escape` | `tui.altScreen.searchClose` | Close transcript search |
| `home` | `tui.altScreen.top` | Scroll to the beginning of the transcript |
| `end` | `tui.altScreen.bottom` | Scroll to the transcript end and follow new output |

These bindings take precedence over the editor's, so in fullscreen mode the
unmodified navigation keys drive the transcript and their `ctrl` variants
(`ctrl+home`, `ctrl+end`, `ctrl+pageUp`, `ctrl+pageDown`) drive the editor.
In regular mode, both variants drive the editor.

The routing is just action bindings, so it is configurable:
`"tui.altScreen.pageUp": "ctrl+pageUp"` gives `pageUp` back to the editor, and
`"tui.altScreen.pageUp": []` disables the transcript shortcut. Bind
`tui.altScreen.halfPageUp`/`halfPageDown` for smaller steps while keeping the
full-page bindings. A user binding replaces that action's defaults.

### Editor and input

The prompt editor and every list share these `tui.*` actions:

| Keys | Action |
|------|--------|
| `left`, `ctrl+b` / `right`, `ctrl+f` | `tui.editor.cursorLeft` / `cursorRight` |
| `alt+left`, `ctrl+left`, `alt+b` / `alt+right`, `ctrl+right`, `alt+f` | `tui.editor.cursorWordLeft` / `cursorWordRight` |
| `home`, `ctrl+home`, `ctrl+a` / `end`, `ctrl+end`, `ctrl+e` | `tui.editor.cursorLineStart` / `cursorLineEnd` |
| `ctrl+]` / `ctrl+alt+]` | `tui.editor.jumpForward` / `jumpBackward` (to a typed character) |
| `pageUp`, `ctrl+pageUp` / `pageDown`, `ctrl+pageDown` | `tui.editor.pageUp` / `pageDown` |
| `backspace` / `delete`, `ctrl+d` | `tui.editor.deleteCharBackward` / `deleteCharForward` |
| `ctrl+w`, `alt+backspace` / `alt+d`, `alt+delete` | `tui.editor.deleteWordBackward` / `deleteWordForward` |
| `ctrl+u` / `ctrl+k` | `tui.editor.deleteToLineStart` / `deleteToLineEnd` |
| `ctrl+y` / `alt+y` | `tui.editor.yank` / `yankPop` (kill ring) |
| `ctrl+-` | `tui.editor.undo` |
| `shift+enter`, `ctrl+j` / `enter` / `tab` | `tui.input.newLine` / `submit` / `tab` (autocomplete) |
| `ctrl+c` | `tui.input.copy` |
| `up` / `down` / `pageUp` / `pageDown` | `tui.select.up` / `down` / `pageUp` / `pageDown` |
| `enter` / `escape`, `ctrl+c` | `tui.select.confirm` / `cancel` |

## Customizing

Create `~/.pidrei/agent/keybindings.json` (under `PIDREI_CODING_AGENT_DIR` if
set), keyed by action id:

```json
{
  "app.model.select": "ctrl+m",
  "app.session.tree": ["ctrl+b", "f2"],
  "app.suspend": []
}
```

A string binds one key, a list binds several, and an empty list unbinds. A
configured value replaces that action's defaults. Run `/reload` after editing
to apply it to the running session. Pre-namespaced ids from older configs
(`cursorUp`, `expandTools`) are still accepted.

Key syntax is modifiers plus a key: `ctrl+`, `alt+`, `shift+`, `super+`, then a
letter, digit, symbol (`` ` ``, `-`, `=`, `[`, `]`, `\`, `;`, `'`, `,`, `.`,
`/` and the shifted ones), `f1`–`f12`, or a named key (`escape`/`esc`,
`enter`/`return`, `tab`, `space`, `backspace`, `delete`, `insert`, `clear`,
`home`, `end`, `pageUp`, `pageDown`, `up`, `down`, `left`, `right`). Modifiers
combine: `ctrl+shift+x`, `alt+ctrl+x`, `super+k`, `ctrl+super+k`, `ctrl+1`.

`super` bindings need a terminal that reports the modifier separately, in
practice one speaking the Kitty keyboard protocol; elsewhere they never fire.

## Extension shortcuts

Extensions register keys with `pi.register_shortcut(...)`. A shortcut that
collides with one of these reserved actions is refused with a diagnostic rather
than silently shadowing it:

```
app.interrupt, app.clear, app.exit, app.suspend, app.thinking.cycle,
app.thinking.toggle, app.model.cycleForward, app.model.cycleBackward,
app.model.select, app.tools.expand, app.editor.external, app.message.copy,
app.message.followUp, tui.input.submit, tui.input.copy,
tui.select.confirm, tui.select.cancel, tui.editor.deleteToLineEnd
```

Matching is case-insensitive, so `Ctrl+C` collides with `ctrl+c`. Rebinding a
reserved action yourself in `keybindings.json` is allowed — the restriction is on
extensions, not on you.

## Terminal limits

Some combinations never reach the application: terminals intercept `ctrl+s` and
`ctrl+q` for flow control, and many cannot distinguish `ctrl+i` from `tab` or
`ctrl+m` from `enter`. If a binding seems ignored, try another key before
assuming a bug.
