# TUI components

`pidrei_tui` is the terminal UI toolkit pidrei renders itself with. Extensions
use it for widgets, custom renderers and overlays.

```python
from pidrei_tui import Container, Text, SelectList, Spacer
```

The package never imports the agent or provider layers, so it is usable on its
own.

## Model

A UI is a tree of components. Each renders itself into lines for a given width;
the terminal diffs frames and writes only what changed.

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

| Component | Purpose |
|-----------|---------|
| `Container` | Groups children; the basic building block |
| `Text` | A block of text, with padding |
| `TruncatedText` | Text clipped to the available width |
| `Markdown` | Rendered markdown, with syntax-highlighted code |
| `Spacer` | Blank lines |
| `Box` | A bordered container |
| `SelectList` | A selectable list with filtering |
| `SettingsList` | Rows of labelled, cycling values |
| `Editor` / `Input` | Multi-line and single-line text entry |
| `Loader` / `CancellableLoader` | Progress spinners |
| `Image` | Inline image, where the terminal supports it |

Helpers: `fuzzy_filter` / `fuzzy_match` for list filtering,
`get_capabilities()` for what the terminal supports, and
`get_cell_dimensions()` for pixel sizing.

## Widgets from an extension

```python
from pidrei_tui import Container, Text


def extension(pi):
    async def on_turn_end(_event, ctx):
        if not ctx.has_ui:
            return
        widget = Container()
        widget.add_child(Text(ctx.ui.theme.fg("accent", "turn complete"), 1, 0))
        ctx.ui.set_widget("my-widget", widget)

    pi.on("turn_end", on_turn_end)
```

`set_widget(key, component)` installs or replaces a widget; passing `None`
removes it. `set_status(text)` is the one-line version.

## Theming

Do not hardcode colours. `ctx.ui.theme` resolves the active theme:

```python
theme.fg("accent", "text")  # semantic foreground
theme.bold("text")
theme.strikethrough("text")
```

Roles come from [themes.md](themes.md), so a widget follows whatever theme the
user has chosen.

## Overlays and prompts

For transient interaction prefer the context helpers over building components:

```python
choice = await ctx.ui.select("Pick one", ["a", "b"])
text = await ctx.ui.editor("Edit this", "prefill")
```

Both return `None` if the user dismisses them. They only work when
`ctx.has_ui` is true.

## Capabilities

`get_capabilities()` reports hyperlink support, image protocol, colour depth
and mouse support. Check before using an optional feature — `Image` degrades to
a placeholder where images are unsupported, but hyperlinks need a check.

## Width and unicode

Components lay out in terminal cells, not characters: wide CJK glyphs take two
cells, combining marks take none, and emoji vary. `pidrei_tui` handles this
through grapheme segmentation. Measure text with the package's helpers rather
than `len()`.
