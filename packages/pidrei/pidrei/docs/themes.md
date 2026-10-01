# Themes

A theme is a JSON file mapping semantic colour roles to colours, used by
interactive mode and HTML exports. `system`, `dark` and `light` ship with
pidrei; pick one in `/settings` → **Theme**, which saves the `theme` setting.
Without a `theme` setting, pidrei uses `system`.

## The system theme

`system` builds pidrei's colours from your terminal's own theme instead of
bringing a palette of its own. pidrei asks the terminal for its default
foreground and background and its 16 ANSI colours; each role takes its hue
from one ANSI colour (errors from red, links from blue, …) and a lightness
that stands out from the background by a minimum contrast. Body text keeps at
least a 4.5:1 WCAG contrast ratio on the background and on every panel.

| Terminal reports | Result |
|------------------|--------|
| Background and ANSI colours | Colours from the terminal palette, placed for the actual background |
| Background only | pidrei's own hues, placed for the actual background |
| Nothing | ANSI colour indices and the terminal's default colours, which the terminal renders itself; secondary text is faint and panels have no background |

Terminals usually answer within a few milliseconds; pidrei waits at most
100 ms before drawing the startup header, falls back to the ANSI indices if
nothing arrived, and still applies the colours if they come later (over a slow
SSH link, say). When the terminal switches between light and dark, pidrei
asks again and rebuilds the theme. `system` is a reserved name: a custom
theme called `system` is ignored.

## Writing one

Start from a copy of a built-in theme — `dark.json` and `light.json` ship in
the installed package under `pidrei/modes/interactive/theme/` — since `colors`
must define every required role. The built-ins are written in OKHSL, with
`vars` for colours several roles share, so a hue, saturation or lightness can
be adjusted directly. Save it as
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
| `name` | Theme name, unique among loaded themes. **Must not contain `/`** — that separates the light and dark halves of an automatic theme setting — and cannot be `system` |
| `appearance` | Optional `"dark"` or `"light"`: the background the theme is designed for; detected from the theme's colours when omitted |
| `vars` | Optional reusable colours; a var may reference another var |
| `colors` | Semantic roles; the schema marks which are required |
| `export` | Optional `pageBg`, `cardBg` and `infoBg` for HTML exports; derived from `userMessageBg` when omitted |

A colour value is one of:

| Form | Example | Meaning |
|------|---------|---------|
| Hex | `"#0af"`, `"#00aaff"` | A three- or six-digit sRGB colour |
| OKLCH | `"oklch(62% 0.1 200)"` | Perceptual lightness, chroma and hue |
| OKHSL | `"okhsl(250 60% 55%)"` | Hue, saturation and lightness; saturation is relative to the most sRGB allows at that hue and lightness, so every value is in gamut and equal saturation looks equally colourful |
| Palette index | `39` | A 256-colour palette index, `0`–`255` |
| Var reference | `"primary"` | The value of a `vars` entry |
| Terminal default | `""` | The terminal's default foreground or background |

Chained var references resolve; a missing or circular reference makes the
theme invalid. Invalid themes are reported at startup and on `/reload`.

A terminal-default colour renders as the terminal's own. Where pidrei needs a
concrete value — HTML export, or colour maths in an extension — it uses the
default colour the terminal reported, or a black or white guess from the
theme's appearance.

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
terminal's appearance, switching again when the terminal reports an
appearance change:

```jsonc
{ "theme": "solar-light/solar-dark" }
```

This is why a theme name may not contain `/`. The theme selector in
`/settings` offers this as "automatic", right below `system`.

pidrei decides whether the terminal is light or dark from the background and
foreground colours it reports. A terminal that does not report its background
is judged by its light/dark notification, then the `COLORFGBG` environment
variable, then taken as dark. The same decision picks the half of a
light/dark pair and the appearance of `system`.

## Locations

| Location | Scope |
|----------|-------|
| Built-in | `system`, `dark`, `light` |
| `~/.pidrei/agent/themes/` | User |
| `<project>/.pidrei/themes/` | Project, once the project is trusted |
| `themes` array in settings | Files or directories |
| `--theme <path>` | This run (repeatable) |

Packages may ship themes; see [packages.md](packages.md). `--no-themes`
disables everything but the built-ins and paths passed with `--theme`. Two
loaded themes with the same name are reported as a collision.

## Colour support

pidrei detects terminal capability and degrades: truecolour where available
(`COLORTERM=truecolor`/`24bit`, or a `TERM` ending in `-direct`), colours
approximated to the 256-colour palette otherwise. OKLCH colours outside sRGB
are gamut-mapped, and HTML exports convert OKHSL values to hex because CSS
does not support them. If colours look off, check the detection
(`PIDREI_TRUE_COLOR`, see [environment-variables.md](environment-variables.md))
and your terminal's contrast settings. An empty string means "use the
terminal's own default", which is the right choice for backgrounds you want
left alone.
