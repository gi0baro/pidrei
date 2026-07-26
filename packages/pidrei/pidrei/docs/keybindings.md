# Keybindings

Every binding has an **action id** (`app.model.select`) and a set of keys. Show
the current bindings with `/keybindings`.

## Defaults

### Global

| Keys | Action | Does |
|------|--------|------|
| `escape` | `app.interrupt` | Cancel or abort |
| `ctrl+c` | `app.clear` | Clear the editor |
| `ctrl+d` | `app.exit` | Exit when the editor is empty |
| `ctrl+z` | `app.suspend` | Suspend to background |
| `shift+tab` | `app.thinking.cycle` | Cycle thinking level |
| `ctrl+t` | `app.thinking.toggle` | Show or hide thinking blocks |
| `ctrl+p` / `shift+ctrl+p` | `app.model.cycleForward` / `Backward` | Cycle model |
| `ctrl+l` | `app.model.select` | Open the model selector |
| `ctrl+o` | `app.tools.expand` | Expand or collapse tool output |
| `ctrl+g` | `app.editor.external` | Open `$EDITOR` |
| `ctrl+x` | `app.message.copy` | Copy the last message |
| `alt+enter` | `app.message.followUp` | Queue a follow-up message |
| `alt+up` | `app.message.dequeue` | Restore queued messages |
| `ctrl+n` | `app.session.toggleNamedFilter` | Toggle the named-session filter |

`app.session.new`, `.tree`, `.fork` and `.resume` have no default key; bind them
if you want them.

### Session tree

| Keys | Action |
|------|--------|
| `left` / `right` | Fold / unfold |
| `shift+l` | Edit the entry label |
| `shift+t` | Toggle label timestamps |
| `ctrl+d` / `ctrl+t` / `ctrl+u` / `ctrl+l` / `ctrl+a` | Filter: default, no tools, user only, labeled only, all |

### Session and model selectors

| Keys | Action |
|------|--------|
| `ctrl+p` / `ctrl+s` / `ctrl+r` / `ctrl+d` | Toggle path, toggle sort, rename, delete |
| `ctrl+s` / `ctrl+a` / `ctrl+x` / `ctrl+p` | Save, enable all, clear all, toggle provider |
| `alt+up` / `alt+down` | Reorder a model |

The same keys mean different things in different surfaces — bindings are scoped
to the component that has focus.

## Customizing

Add a `keybindings` object to `settings.json`, keyed by action id:

```jsonc
{
  "keybindings": {
    "app.model.select": "ctrl+m",
    "app.session.tree": ["ctrl+b", "f2"],
    "app.suspend": []
  }
}
```

A string binds one key, a list binds several, and an empty list unbinds.

Key syntax is modifiers plus a key: `ctrl+`, `alt+`, `shift+`, then a letter,
digit, or a named key (`enter`, `escape`, `tab`, `space`, `up`, `down`, `left`,
`right`, `home`, `end`, `pageup`, `pagedown`, `f1`–`f12`).

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
reserved action yourself in `settings.json` is allowed — the restriction is on
extensions, not on you.

## Terminal limits

Some combinations never reach the application: terminals intercept `ctrl+s` and
`ctrl+q` for flow control, and many cannot distinguish `ctrl+i` from `tab` or
`ctrl+m` from `enter`. If a binding seems ignored, try another key before
assuming a bug.
