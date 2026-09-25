# Themes

A theme is a JSON file mapping semantic colour roles to colours, used by
interactive mode and HTML exports. `dark` and `light` ship with pidrei; pick one
in `/settings` → **Theme**, which saves the `theme` setting.

## Writing one

Start from a copy of a built-in theme — `dark.json` and `light.json` ship in
the installed package under `pidrei/modes/interactive/theme/` — since `colors`
must define every required role. Save it as
`~/.pidrei/agent/themes/solar.json`, set `name` to `solar`, adjust `vars` and
`colors`, then select it in `/settings`. Abridged:

```jsonc
{
  "name": "solar",
  "vars": {
    "base":   "#002b36",
    "accent": "#268bd2",
    "text":   "#839496",
    "red":    "#dc322f",
    "green":  "#859900"
  },
  "colors": {
    "text":         "text",
    "accent":       "accent",
    "border":       "accent",
    "borderMuted":  "base",
    "success":      "green",
    "error":        "red",
    "warning":      "#b58900",
    "muted":        "text",
    "dim":          "base",
    "selectedBg":   "base"
    // …every other required role
  }
}
```

Name the file after the theme. pidrei hot-reloads the active theme only when
it lives at `~/.pidrei/agent/themes/<name>.json`; run `/reload` after adding or
changing a theme anywhere else.

| Key | Meaning |
|-----|---------|
| `name` | Theme name, unique among loaded themes. **Must not contain `/`** — that separates the light and dark halves of an automatic theme setting |
| `vars` | Optional reusable colours; a var may reference another var |
| `colors` | Semantic roles; the schema marks which are required |
| `export` | Optional `pageBg`, `cardBg` and `infoBg` for HTML exports; derived from `userMessageBg` when omitted |

A colour value is a `"#rrggbb"` hex string, a 256-colour palette index (`0`–
`255`), the name of a `vars` entry, or `""` for the terminal's default
foreground or background. Chained var references resolve; a missing or
circular reference makes the theme invalid. Invalid themes are reported at
startup and on `/reload`.

Defining a palette in `vars` and referring to it from `colors` keeps a theme
readable, but any `colors` entry can be a literal value.

Roles are named for interface areas rather than widgets: general UI (`accent`,
`border*`, `text`, `muted`, `dim`, `success`, `error`, `warning`), selection
and fullscreen (`selectedBg`, `searchMatch*`, `scrollbar*`), messages
(`userMessage*`, `customMessage*`, `thinkingText`), tool execution (`tool*`),
markdown (`md*`), diffs (`toolDiff*`), syntax highlighting (`syntax*`), and
editor modes (`thinking*`, `bashMode`).

The full list of roles is in `theme-schema.json`, shipped next to the built-in
themes. Point `$schema` at it and an editor will complete and validate as you
type. Five roles are optional, so older themes keep loading: `thinkingMax`
falls back to `thinkingXhigh`, `scrollbarTrack` (the fullscreen scrollbar
track foreground) falls back to `muted`, `scrollbarThumb` (the thumb
foreground, shared by the normal and expanded states) falls back to `text`,
`searchMatchBg` falls back to `selectedBg`, and `searchMatchText` falls back
to `text`. Non-current transcript search matches render as
`searchMatchText` on `searchMatchBg` with an underline; the current match
reverses that foreground/background pair and uses bold text.

## Initial theme

Start an interactive run with a theme without changing the saved setting:

```bash
pidrei --use-theme light
```

To follow terminal appearance, use `lightTheme/darkTheme` syntax:

```bash
pidrei --use-theme light/dark
```

The CLI value is the initial theme for that run. Choosing another theme later
in `/settings` applies it immediately and saves it normally.

## Automatic light/dark

Set the theme to `light-name/dark-name` (light first) and pidrei picks per the
terminal's reported background, switching again when the terminal reports an
appearance change:

```jsonc
{ "theme": "solar-light/solar-dark" }
```

This is why a theme name may not contain `/`. The theme selector in
`/settings` offers this as "Automatic".

## Locations

| Location | Scope |
|----------|-------|
| Built-in | `dark`, `light` |
| `~/.pidrei/agent/themes/` | User |
| `<project>/.pidrei/themes/` | Project, once the project is trusted |
| `themes` array in settings | Files or directories |
| `--theme <path>` | This run (repeatable) |

Packages may ship themes; see [packages.md](packages.md). `--no-themes`
disables everything but the built-ins and paths passed with `--theme`. Two
loaded themes with the same name are reported as a collision.

## Colour support

pidrei detects terminal capability and degrades: truecolour where available,
hex colours approximated to the 256-colour palette otherwise. If colours look
off, check the detection (`PIDREI_TRUE_COLOR`, see
[environment-variables.md](environment-variables.md)) and your terminal's
contrast settings. An empty string means "use the terminal's own default",
which is the right choice for backgrounds you want left alone.
