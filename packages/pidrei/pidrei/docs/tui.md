# TUI components

`pidrei_tui` is the terminal UI toolkit pidrei renders itself with. Extensions
use it for widgets, custom renderers and overlays — but start with the `ctx.ui`
helpers ([extensions.md](extensions.md)), and build a component only when the
interaction needs its own rendering, input, focus or lifecycle:

| Need | Use |
|------|-----|
| Select, confirm, input, multi-line edit | `ctx.ui.select()`, `confirm()`, `input()`, `editor()` |
| Non-blocking feedback | `ctx.ui.notify()` or `set_status()` |
| Persistent content near the editor | `ctx.ui.set_widget()` |
| Replace the header, footer or editor | `set_header()`, `set_footer()`, `set_editor_component()` |
| A temporary screen or overlay | `ctx.ui.custom()` |
| Custom rendering of a tool or message | An extension renderer |

These receive pidrei's active theme and keybindings; never start a second
terminal renderer from an extension.

```python
from pidrei_tui import Container, Text, SelectList, Spacer
```

The package never imports the agent or provider layers, so it is usable on its
own.

## Model

A UI is a tree of components. Each renders itself into lines for a given width;
the terminal diffs frames and writes only what changed.

A component implements `render(width) -> list[str]` and `invalidate()`, which
must drop any cached output — it is called on theme changes and whenever a
full re-render is needed. Keyboard input (`async handle_input(data)`) and
mouse input are optional. Every rendered line must fit `width` in terminal
cells (see [Width and unicode](#width-and-unicode)). Styling and hyperlinks are
reset after every line, so reapply styles per line. After changing state,
invalidate what changed and call the injected `tui.request_render()`; requests
are coalesced.

Two renderers implement that contract. `TuiMainScreen` (the default) renders
into the terminal's own screen and scrollback. `TuiAltScreen`, selected with
`--tui-mode fullscreen`, takes over the alternate screen and owns both the box
tree and the scrolling: the transcript scrolls inside a `ScrollView` while
queued messages, working status, extension widgets, editor and footer stay
fixed in a dock below it. `shift+pageUp`/`shift+pageDown`/`ctrl+home`/`ctrl+end`
move the transcript, the mouse wheel scrolls whichever region is under the
pointer, dragging selects text into the clipboard, and clicking an OSC 8 link
opens it. Inline images then need the Kitty graphics protocol (Kitty,
Ghostty); iTerm2 falls back to text placeholders because its protocol cannot
delete or crop placements while the application scrolls. In `regular` mode,
iTerm2 inline images render normally.

Either mode can be selected at runtime from **TUI mode** in `/settings`;
InteractiveMode swaps the renderer under a stable `ui` reference, so the whole
component tree is remounted rather than rebuilt. The setting is only consulted
for the initial renderer, which `--tui-mode` overrides for one run.

`VStack`/`HStack` (flex-style `basis`/`grow`/`shrink`/`minSize`/`maxSize`
entries) and `ScrollView` are the layout-aware components the alternate screen
measures; on the main screen they render as plain stacked output.

Fullscreen mode also routes normalized press, release, click, move, drag and
wheel events to components and overlays through an optional
`async handle_mouse(event)` method (a `TuiMouseEvent`). Return
`TuiMouseEventResult(handled=True)` to suppress default behavior,
`capture=True` to retain drag/release ownership, `focus=True` to request
keyboard focus, and `render=True` when a hover or release visibly changes the
component. Press, click, drag and wheel render by default; no-op move/release
events do not. `MouseRegion(component, on_mouse)` adds mouse behavior to an
existing component without changing its rendering:

```python
from pidrei_tui import MouseRegion, TuiMouseEventResult


def on_mouse(event):
    if event.type != "click" or event.button != "left":
        return None
    toggle()
    return TuiMouseEventResult(handled=True)


clickable = MouseRegion(content, on_mouse)
```

Unhandled wheel input scrolls the nearest `ScrollView`; unhandled
primary-button drags retain transcript selection. OSC 8 links take precedence
over parent click regions. `Input`, `Editor`, `SelectList` and `SettingsList`
include fullscreen mouse behavior. Regular mode does not capture mouse input
because the terminal owns its scrollback.

| Component | Purpose |
|-----------|---------|
| `Container` | Groups children; the basic building block |
| `Text` | A block of text, with padding |
| `TruncatedText` | Text clipped to the available width |
| `Markdown` | Rendered markdown, with syntax-highlighted code |
| `Spacer` | Blank lines |
| `Box` | A container with padding and a background |
| `VStack` / `HStack` | Flex-style vertical and horizontal layout |
| `ScrollView` | A bounded, scrollable viewport |
| `SelectList` | A selectable list with filtering |
| `SettingsList` | Rows of labelled, cycling values |
| `MouseRegion` | Adds mouse handling to a component without changing its rendering |
| `Editor` / `Input` | Multi-line and single-line text entry |
| `Loader` / `CancellableLoader` | Progress spinners |
| `Image` | Inline image, where the terminal supports it |

Helpers: `fuzzy_filter` / `fuzzy_match` for list filtering,
`get_capabilities()` for what the terminal supports, and
`get_cell_dimensions()` for pixel sizing.

## Widgets from an extension

```python
from pidrei_tui import Container, Text


async def extension(pi):
    async def on_turn_end(_event, ctx):
        if not ctx.has_ui:
            return

        def build(tui, theme):
            widget = Container()
            widget.add_child(Text(theme.fg("accent", "turn complete"), 1, 0))
            return widget

        ctx.ui.set_widget("my-widget", build)

    pi.on("turn_end", on_turn_end)
```

`set_widget(key, content, options=None)` installs or replaces a widget.
`content` is a list of strings (one `Text` row each) or a factory
`(tui, theme) -> component`; `None` removes the widget, and
`{"placement": "belowEditor"}` moves it under the editor (the default is
`"aboveEditor"`). `set_status(key, text)` is the one-line version, shown in
the footer.

## Theming

Do not hardcode colours. `ctx.ui.theme` resolves the active theme:

```python
theme.fg("accent", "text")  # semantic foreground
theme.bold("text")
theme.strikethrough("text")
```

Roles come from [themes.md](themes.md), so a widget follows whatever theme the
user has chosen. Use the theme handed to your factory or callback, and don't
keep themed strings in long-lived state unless `invalidate()` rebuilds them: a
theme change clears render caches but cannot recolour ANSI already baked into
your data. Styling during `render()` needs no special care. For Markdown that
matches the app, pass `get_markdown_theme()` (from
`pidrei.modes.interactive.theme`) to `Markdown`.

## Overlays and prompts

For transient interaction prefer the context helpers over building components:

```python
choice = await ctx.ui.select("Pick one", ["a", "b"])
text = await ctx.ui.editor("Edit this", "prefill")
```

Both return `None` if the user dismisses them. They only work when
`ctx.has_ui` is true.

When those are not enough, `await ctx.ui.custom(factory, options)` hands the
interactive area to one component until it finishes. The factory must be an
`async def` returning the component; it is awaited as
`await factory(tui, theme, keybindings, done)`, and calling `done(result)`
resolves `custom()` with `result` and disposes the component.
By default the component replaces the editor; `{"overlay": True}` draws it on
top of existing content instead, with `"overlayOptions"` (size, anchor,
offsets, margins, responsive visibility — a dict or a callable returning one)
and `"onHandle"` receiving an `OverlayHandle` for `focus()`, `unfocus()` and
`set_hidden()`. A focused overlay keeps input across ordinary renders; move
focus through the handle if something else should receive keys. Finish with
`done()` — never `handle.hide()` an overlay `custom()` created — and create a
fresh component for each interaction rather than reusing one.

## Keyboard and focus

Match input with `matches_key(data, ...)` and `Key`, which understand the
supported keyboard protocols and modifiers; for configurable app actions use
the `KeybindingsManager` the factory receives. A component that shows a text
cursor sets a `focused` attribute (the `Focusable` protocol) and emits
`CURSOR_MARKER` right before the cursor, so the hardware cursor — and IME
candidate windows for Chinese, Japanese, Korean input — land in the right
place. A container wrapping an `Input` or `Editor` must pass its own `focused`
state down to that child.

To replace the main editor, subclass `CustomEditor`
(`pidrei.modes.interactive.components`) so app shortcuts and agent controls
keep working, forward keys you don't handle to the base class, and install it
with `ctx.ui.set_editor_component(factory)`; `None` restores the default.

Even with fullscreen mouse support, give every interaction a keyboard path.

## Capabilities

`get_capabilities()` returns a dict with `images` (the image protocol, or
`None`), `trueColor` and `hyperlinks`. Check before using an optional
feature — `Image` degrades to a placeholder where images are unsupported, but
hyperlinks need a check.

## Responsiveness

Rendering runs on the interactive path. Cache expensive layout or
highlighting by width and content, and clear that cache in `invalidate()`.
Keep the default view compact and put detail behind expansion or a dedicated
screen. `PIDREI_TUI_WRITE_LOG=<path>` captures the raw ANSI stream when
debugging; test narrow widths, wide characters, resizes, theme changes, focus
changes, and both TUI modes.

## Width and unicode

Components lay out in terminal cells, not characters: wide CJK glyphs take two
cells, combining marks take none, and emoji vary. `pidrei_tui` handles this
through grapheme segmentation. Measure and cut text with the package's
helpers — `visible_width`, `truncate_to_width`, `slice_by_column`,
`wrap_text_with_ansi` — rather than `len()` and slicing.
