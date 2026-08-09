# Themes

A theme is a JSON file mapping semantic colour roles to colours. `dark` and
`light` ship with pidrei; `/theme` switches between what is loaded.

## Writing one

Create `.pidrei/themes/solar.json`:

```jsonc
{
  "$schema": "../../theme-schema.json",
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
  }
}
```

| Key | Meaning |
|-----|---------|
| `name` | Theme name. **Must not contain `/`** — that separates the light and dark halves of an automatic theme setting |
| `vars` | Reusable colours: a hex value, a reference to another var, or `""` for the terminal default |
| `colors` | Semantic roles, each a hex value or a `vars` reference |
| `export` | Optional terminal palette export |

Defining a palette in `vars` and referring to it from `colors` keeps a theme
readable, but any `colors` entry can be a literal hex value.

The full list of roles is in `theme-schema.json`, shipped next to the built-in
themes. Point `$schema` at it and an editor will complete and validate as you
type. Two roles are optional, so older themes keep loading: `thinkingMax` falls
back to `thinkingXhigh`, and `scrollbarThumb` (the fullscreen scrollbar thumb)
falls back to `selectedBg`.

## Automatic light/dark

Set the theme to `light-name/dark-name` and pidrei picks per the terminal's
reported background:

```jsonc
{ "theme": "solar-light/solar-dark" }
```

This is why a theme name may not contain `/`. The `/theme` selector offers this
as "Automatic".

## Locations

| Location | Scope |
|----------|-------|
| Built-in | `dark`, `light` |
| `~/.pidrei/agent/themes/` | User |
| `<project>/.pidrei/themes/` | Project |
| `--theme <path>` | This run |

Packages may ship themes; see [packages.md](packages.md). `--no-themes`
disables everything but the built-ins.

## Colour support

pidrei detects terminal capability and degrades: truecolour where available,
256-colour otherwise. An empty string means "use the terminal's own default",
which is the right choice for backgrounds you want left alone.
