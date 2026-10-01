"""Mirror of pi coding-agent src/modes/interactive/theme/theme.ts.

Theme records mirror pi's JSON shape (camelCase color keys). The global
``theme`` export is a proxy over the module-level current theme (pi uses a
globalThis Proxy for the same purpose); ``init_theme()`` must run first.

Deviations:
- Validation is hand-rolled against pi's typebox schema (same required color
  set, same error message layout) instead of a JSON-schema engine.
- The custom-theme watcher polls (utils/fs_watch) instead of node fs.watch;
  the 100 ms debounced reload is a ``_timers.Timeout`` whose file read runs
  on the pool, on its own task. The module lock guards the theme globals
  against `set_theme` callers on other tasks.
- The terminal's reported colors (``_terminal_colors``, its pending flag and
  ``_terminal_color_scheme``) are copy-on-write globals under the same lock;
  their writer is the theme controller's terminal-colors listener, run by the
  TUI's one terminal-event consumer.
- Syntax highlighting is pygments (utils/syntax_highlight), keyed by the same
  scope names pi feeds cli-highlight.

ThemeStyle is an option dict: TextAttributes keys plus ``fg`` (a ThemeColor
token or a Color) and ``bg`` (a ThemeBg token or a Color). Tokens are only
accepted in their own slot, because "" (terminal default) means the default
foreground or background depending on the slot; use ``theme.colors[token]``
to use a token's color in the other slot.
"""

import contextlib
import json
import os
import re
import threading
import types
from collections.abc import Awaitable, Callable

import tonio.colored as tonio
from tonio.colored import fs

from pidrei_tui import (
    IndexedColor,
    background_ansi,
    color_to_hex,
    color_to_oklch,
    foreground_ansi,
    get_terminal_color_mode,
    indexed_color,
    mix_colors,
    parse_color,
    rgb_color,
    style_text_with_ansi,
)
from pidrei_tui._timers import Timeout

from ....config import get_custom_themes_dir, get_themes_dir
from ....utils import colors as chalk
from ....utils.fs_watch import close_watcher, watch_with_error_handler
from ....utils.syntax_highlight import highlight, supports_language
from ....utils.text import strip_bom
from .system_theme import SYSTEM_THEME_NAME, generate_system_theme_colors, terminal_appearance


# ============================================================================
# Types & Schema
# ============================================================================

# The schema that validates the theme document shape lives in `theme_json.py`
# (ColorValue: hex, OKLCH, OKHSL, var ref "primary", empty "", or 256-color
# index).

# pi: `let themeJsonValidator: ThemeJsonValidator | undefined`, set once at
# startup. Read under `_theme_state_lock` like the other theme globals.
_theme_json_validator = None


def set_theme_json_validator(validator) -> None:
    """Install full theme validation.

    Without it, documents are accepted as-is, which is what built-in themes
    already do: pi keeps validation out of a presentation that only uses
    built-in themes, and `main.py` installs it before the first theme loads.
    """
    global _theme_json_validator
    with _theme_state_lock:
        _theme_json_validator = validator


_BACKGROUND_TOKENS = frozenset(
    (
        "selectedBg",
        "searchMatchBg",
        "userMessageBg",
        "customMessageBg",
        "toolPendingBg",
        "toolSuccessBg",
        "toolErrorBg",
    )
)


# ============================================================================
# Color Utilities
# ============================================================================

_OK_COLOR_RE = re.compile(r"^ok(lch|hsl)\(", re.IGNORECASE)
_OKHSL_RE = re.compile(r"^okhsl\(", re.IGNORECASE)


def _resolve_var_refs(value, vars_map: dict, visited: set | None = None):
    if isinstance(value, int) or value == "" or value.startswith("#") or _OK_COLOR_RE.match(value):
        return value
    visited = visited if visited is not None else set()
    if value in visited:
        raise ValueError(f"Circular variable reference detected: {value}")
    if value not in vars_map:
        raise ValueError(f"Variable reference not found: {value}")
    visited.add(value)
    return _resolve_var_refs(vars_map[value], vars_map, visited)


def _resolve_theme_colors(colors: dict, vars_map: dict | None = None) -> dict:
    vars_map = vars_map or {}
    return {key: _resolve_var_refs(value, vars_map) for key, value in colors.items()}


def _with_theme_color_fallbacks(colors: dict) -> dict:
    scrollbar_track = colors.get("scrollbarTrack")
    if scrollbar_track is None:
        scrollbar_track = colors["muted"]
    scrollbar_thumb = colors.get("scrollbarThumb")
    if scrollbar_thumb is None:
        scrollbar_thumb = colors["text"]
    fallback = colors.get("thinkingMax")
    if fallback is None:
        fallback = colors["thinkingXhigh"]
    search_match_bg = colors.get("searchMatchBg")
    if search_match_bg is None:
        search_match_bg = colors["selectedBg"]
    search_match_text = colors.get("searchMatchText")
    if search_match_text is None:
        search_match_text = colors["text"]
    return {
        **colors,
        "scrollbarTrack": scrollbar_track,
        "scrollbarThumb": scrollbar_thumb,
        "thinkingMax": fallback,
        "searchMatchBg": search_match_bg,
        "searchMatchText": search_match_text,
    }


# ============================================================================
# Appearance & Terminal Default Colors
# ============================================================================

# ThemeAppearance, the background a theme is designed for, is a
# TerminalTheme: "dark" | "light".

# The terminal's reported colors (a TerminalColors record). Replaced (never
# mutated) on update, so themes can cache resolved colors by identity. Under
# `_theme_state_lock`, with the two below.
_terminal_colors: dict = {}
# While the terminal color query is in flight, the system theme renders in grayscale.
_terminal_colors_pending = False
# The terminal's last light/dark report (mode 2031). Only used while it has
# not reported a background.
_terminal_color_scheme: str | None = None


def set_terminal_colors(colors: dict) -> None:
    """Record the terminal's reported colors. Themes use the default colors
    for tokens set to "" (terminal default); the system theme is generated
    from all of them. Ends the pending state."""
    global _terminal_colors, _terminal_colors_pending
    with _theme_state_lock:
        _terminal_colors = {**colors}
        _terminal_colors_pending = False


def set_terminal_color_scheme(scheme: str | None) -> None:
    """Record the terminal's light/dark report, the fallback for terminals
    that do not report their background."""
    global _terminal_color_scheme
    with _theme_state_lock:
        _terminal_color_scheme = scheme


def mark_terminal_colors_pending() -> None:
    """Render the system theme in grayscale until `set_terminal_colors()`
    reports the terminal's colors."""
    global _terminal_colors_pending
    with _theme_state_lock:
        _terminal_colors_pending = True


# Assumed terminal default colors when the terminal does not report them.
_GUESSED_DEFAULT_COLORS = {
    "dark": {"foreground": parse_color("#e5e5e7"), "background": parse_color("#000000")},
    "light": {"foreground": parse_color("#000000"), "background": parse_color("#ffffff")},
}


def _average_lightness(colors: list) -> float | None:
    # Palette colors 0-15 follow the user's terminal palette, so they say
    # nothing about the theme.
    fixed = [color for color in colors if not isinstance(color, IndexedColor) or color.index >= 16]
    if not fixed:
        return None
    return sum(color_to_oklch(color)["l"] for color in fixed) / len(fixed)


def _detect_appearance(foregrounds: list, backgrounds: list) -> str | None:
    """Detect the background a theme is designed for from the lightness of its own colors."""
    fg = _average_lightness(foregrounds)
    bg = _average_lightness(backgrounds)
    if fg is not None and bg is not None:
        return "dark" if bg < fg else "light"
    if bg is not None:
        return "dark" if bg < 0.5 else "light"
    if fg is not None:
        return "dark" if fg > 0.5 else "light"
    return None


# ============================================================================
# Theme Class
# ============================================================================


class Theme:
    def __init__(self, fg_colors: dict, bg_colors: dict, mode: str, options: dict | None = None):
        """``options``: optional ``name``, ``sourcePath``, ``sourceInfo``,
        ``appearance`` and ``dim`` (foreground tokens to render faint, SGR 2)."""
        options = options or {}
        self.name = options.get("name")
        self.source_path = options.get("sourcePath")
        self.source_info = options.get("sourceInfo")
        self._mode = mode
        self._dim_tokens = frozenset(options.get("dim") or ())
        # Precomputed escape sequences keep fg()/bg() on the render hot path
        # to a lookup and concat.
        self._fg_ansi: dict[str, str] = {}
        self._bg_ansi: dict[str, str] = {}
        # Tokens set to "" have no color of their own; `colors` fills them
        # from the terminal defaults.
        self._concrete_colors: dict = {}
        self._default_foreground_tokens: list[str] = []
        self._default_background_tokens: list[str] = []
        # (terminal colors, resolved colors), swapped whole: see `colors`.
        self._resolved_colors: tuple | None = None
        thinking_max = fg_colors.get("thinkingMax")
        if thinking_max is None:
            thinking_max = fg_colors["thinkingXhigh"]
        search_match_text = fg_colors.get("searchMatchText")
        if search_match_text is None:
            search_match_text = fg_colors["text"]
        scrollbar_track = fg_colors.get("scrollbarTrack")
        scrollbar_thumb = fg_colors.get("scrollbarThumb")
        foregrounds = {
            **fg_colors,
            "scrollbarTrack": scrollbar_track if scrollbar_track is not None else fg_colors["muted"],
            "scrollbarThumb": scrollbar_thumb if scrollbar_thumb is not None else fg_colors["text"],
            "thinkingMax": thinking_max,
            "searchMatchText": search_match_text,
        }
        search_match_bg = bg_colors.get("searchMatchBg")
        backgrounds = {
            **bg_colors,
            "searchMatchBg": search_match_bg if search_match_bg is not None else bg_colors["selectedBg"],
        }
        concrete_foregrounds: list = []
        concrete_backgrounds: list = []

        def add_token(token: str, value, is_background: bool) -> str:
            """Returns the escape sequence for the token's own slot."""
            if value == "":
                (self._default_background_tokens if is_background else self._default_foreground_tokens).append(token)
                return "\x1b[49m" if is_background else "\x1b[39m"
            color = parse_color(value)
            self._concrete_colors[token] = color
            (concrete_backgrounds if is_background else concrete_foregrounds).append(color)
            return background_ansi(color, mode) if is_background else foreground_ansi(color, mode)

        for token, value in foregrounds.items():
            self._fg_ansi[token] = add_token(token, value, False)
        for token, value in backgrounds.items():
            self._bg_ansi[token] = add_token(token, value, True)
        appearance = options.get("appearance")
        self._own_appearance = (
            appearance if appearance is not None else _detect_appearance(concrete_foregrounds, concrete_backgrounds)
        )

    @property
    def appearance(self) -> str:
        """The background the theme is designed for: declared in the theme
        JSON, detected from its colors, or, for themes without usable colors,
        the terminal's appearance."""
        return self._own_appearance if self._own_appearance is not None else get_terminal_theme()

    @property
    def colors(self):
        """Concrete colors for all tokens (a read-only mapping). Tokens set to
        "" (terminal default) use the terminal's reported default colors, or a
        guess based on `appearance` when the terminal did not report them.
        Faint tokens are approximated by mixing their color toward the
        background."""
        terminal = _terminal_colors
        resolved = self._resolved_colors
        if resolved is not None and resolved[0] is terminal:
            return resolved[1]
        guess = _GUESSED_DEFAULT_COLORS[self.appearance]

        def to_color(rgb: dict | None, fallback):
            return rgb_color(rgb["r"], rgb["g"], rgb["b"]) if rgb else fallback

        foreground = to_color(terminal.get("foreground"), guess["foreground"])
        background = to_color(terminal.get("background"), guess["background"])
        colors = {**self._concrete_colors}
        for token in self._default_foreground_tokens:
            colors[token] = foreground
        for token in self._default_background_tokens:
            colors[token] = background
        for token in self._dim_tokens:
            color = colors.get(token)
            if color is not None:
                colors[token] = mix_colors(color, background, 0.4)
        frozen = types.MappingProxyType(colors)
        # Render and export paths read this in parallel: one assignment
        # publishes the pair, so no reader sees colors for another report.
        self._resolved_colors = (terminal, frozen)
        return frozen

    def style(self, text: str, options: dict) -> str:
        fg = options.get("fg")
        bg = options.get("bg")
        if isinstance(fg, str) and fg in self._dim_tokens:
            options = {**options, "dim": True}
        return style_text_with_ansi(
            text,
            None
            if fg is None
            else self._token_ansi(self._fg_ansi, fg)
            if isinstance(fg, str)
            else foreground_ansi(fg, self._mode),
            None
            if bg is None
            else self._token_ansi(self._bg_ansi, bg)
            if isinstance(bg, str)
            else background_ansi(bg, self._mode),
            options,
        )

    def fg(self, color: str, text: str) -> str:
        ansi = self._token_ansi(self._fg_ansi, color)
        if color in self._dim_tokens:
            return f"{ansi}\x1b[2m{text}\x1b[22;39m"
        return f"{ansi}{text}\x1b[39m"

    def bg(self, color: str, text: str) -> str:
        ansi = self._token_ansi(self._bg_ansi, color)
        return f"{ansi}{text}\x1b[49m"

    @staticmethod
    def _token_ansi(ansi: dict[str, str], token: str) -> str:
        value = ansi.get(token)
        if value is None:
            raise ValueError(f"Unknown theme color: {token}")
        return value

    def bold(self, text: str) -> str:
        return chalk.bold(text)

    def italic(self, text: str) -> str:
        return chalk.italic(text)

    def underline(self, text: str) -> str:
        return chalk.underline(text)

    def inverse(self, text: str) -> str:
        return chalk.inverse(text)

    def strikethrough(self, text: str) -> str:
        return chalk.strikethrough(text)

    def get_fg_ansi(self, color: str) -> str:
        """Opening escape sequence for a foreground token. Faint tokens
        include SGR 2, which ``\\x1b[22m`` closes."""
        ansi = self._token_ansi(self._fg_ansi, color)
        return f"{ansi}\x1b[2m" if color in self._dim_tokens else ansi

    def get_bg_ansi(self, color: str) -> str:
        return self._token_ansi(self._bg_ansi, color)

    def get_color_mode(self) -> str:
        return self._mode

    def get_thinking_border_color(self, level: str):
        # Map thinking levels to dedicated theme colors
        color_by_level = {
            "off": "thinkingOff",
            "minimal": "thinkingMinimal",
            "low": "thinkingLow",
            "medium": "thinkingMedium",
            "high": "thinkingHigh",
            "xhigh": "thinkingXhigh",
            "max": "thinkingMax",
        }
        color = color_by_level.get(level, "thinkingOff")
        return lambda text: self.fg(color, text)

    def get_bash_mode_border_color(self):
        return lambda text: self.fg("bashMode", text)


# ============================================================================
# Theme Loading
# ============================================================================

_BUILTIN_THEMES: dict | None = None


def _get_builtin_themes() -> dict:
    """The builtin themes from the cache `prime_theme_cache()` fills: no I/O."""
    themes = _BUILTIN_THEMES
    if themes is None:
        raise RuntimeError("builtin themes not loaded: await prime_theme_cache() first")
    return themes


def _load_builtin_themes_blocking() -> dict:
    """The builtin themes, reading them into the cache on first use."""
    global _BUILTIN_THEMES
    if _BUILTIN_THEMES is None:
        themes_dir = get_themes_dir()
        with open(os.path.join(themes_dir, "dark.json"), encoding="utf-8") as f:
            dark = json.loads(strip_bom(f.read()))
        with open(os.path.join(themes_dir, "light.json"), encoding="utf-8") as f:
            light = json.loads(strip_bom(f.read()))
        _BUILTIN_THEMES = {"dark": dark, "light": light}
    return _BUILTIN_THEMES


async def prime_theme_cache() -> None:
    """Warm `_BUILTIN_THEMES` off the runtime.

    pi caches these too, so priming is not a divergence — it only moves the
    one-time read off whatever thread happens to ask first. `set_theme` for a
    builtin or a registered theme then does no I/O at all, which matters
    because it is reached from a sync TUI callback. The async readers below
    go through it too, so the first of them to run does the read pool-side.
    """
    if _BUILTIN_THEMES is None:
        await tonio.spawn_blocking(_load_builtin_themes_blocking)


async def get_available_themes() -> list:
    return [info["name"] for info in await get_available_themes_with_paths()]


async def get_available_themes_with_paths() -> list:
    """Return ``{"name", "path"}`` records for every known theme."""
    themes_dir = get_themes_dir()
    result: list = []
    seen: set = set()

    def add_theme(theme_info: dict) -> None:
        if theme_info["name"] in seen:
            return
        seen.add(theme_info["name"])
        result.append(theme_info)

    # Built-in themes. The system theme is generated, so it has no file.
    add_theme({"name": SYSTEM_THEME_NAME, "path": None})
    await prime_theme_cache()
    for name in _get_builtin_themes():
        add_theme({"name": name, "path": os.path.join(themes_dir, f"{name}.json")})

    # Custom themes
    for theme_info in await _get_custom_theme_infos():
        add_theme(theme_info)

    for name, registered in _registered_themes.items():
        add_theme({"name": name, "path": registered.source_path})

    # The system theme comes first: it is the default and adapts to every terminal.
    return sorted(result, key=lambda info: (info["name"] != SYSTEM_THEME_NAME, info["name"].lower(), info["name"]))


def _scan_custom_theme_dir_blocking(custom_themes_dir: str) -> list[str]:
    """One pool hop for the exists+listdir pair."""
    if not os.path.exists(custom_themes_dir):
        return []
    return sorted(f for f in os.listdir(custom_themes_dir) if f.endswith(".json"))


async def _get_custom_theme_infos() -> list:
    """Re-scans on every call, like pi's `getCustomThemeInfos`.

    Deliberately not cached: pi picks up a theme file dropped in mid-session,
    and caching would silently require a restart. The scan is offloaded rather
    than memoised.
    """
    custom_themes_dir = get_custom_themes_dir()
    result: list = []
    entries = await tonio.spawn_blocking(_scan_custom_theme_dir_blocking, custom_themes_dir)
    for file in entries:
        theme_path = os.path.join(custom_themes_dir, file)
        # Invalid themes are ignored here; the resource loader reports them
        # during normal startup/reload.
        with contextlib.suppress(Exception):
            custom_theme = await load_theme_from_path(theme_path)
            if custom_theme.name:
                result.append({"name": custom_theme.name, "path": theme_path})
    return result


def _assert_theme_name_is_valid(name: str) -> None:
    if "/" in name:
        raise ValueError(
            f'Invalid theme name "{name}": theme names cannot contain "/" '
            "because it is reserved for automatic light/dark theme settings."
        )


def _parse_theme_json(label: str, json_value) -> dict:
    with _theme_state_lock:
        validator = _theme_json_validator
    if validator is not None:
        return validator(label, json_value)
    if not isinstance(json_value, dict) or "colors" not in json_value:
        raise ValueError(f'Invalid theme "{label}": expected an object with a "colors" map.')
    return json_value


def _parse_theme_json_content(label: str, content: str) -> dict:
    try:
        json_value = json.loads(strip_bom(content))
    except ValueError as error:
        raise ValueError(f"Failed to parse theme {label}: {error}") from None
    return _parse_theme_json(label, json_value)


async def _load_theme_json(name: str) -> dict:
    await prime_theme_cache()
    builtin_themes = _get_builtin_themes()
    if name in builtin_themes:
        return builtin_themes[name]
    registered_theme = _registered_themes.get(name)
    if registered_theme is not None and registered_theme.source_path:
        content = await fs.Path(registered_theme.source_path).read_text(encoding="utf-8")
        return _parse_theme_json_content(registered_theme.source_path, content)
    if registered_theme is not None:
        raise ValueError(f'Theme "{name}" does not have a source path for export')
    custom_themes_dir = get_custom_themes_dir()
    theme_path = os.path.join(custom_themes_dir, f"{name}.json")
    if not await fs.Path(theme_path).exists():
        raise ValueError(f"Theme not found: {name}")
    content = await fs.Path(theme_path).read_text(encoding="utf-8")
    return _parse_theme_json_content(name, content)


def _split_theme_colors(colors: dict) -> tuple[dict, dict]:
    fg_colors: dict = {}
    bg_colors: dict = {}
    for key, value in colors.items():
        if key in _BACKGROUND_TOKENS:
            bg_colors[key] = value
        else:
            fg_colors[key] = value
    return fg_colors, bg_colors


def _create_theme(theme_json: dict, mode: str | None = None, source_path: str | None = None) -> Theme:
    color_mode = mode or get_terminal_color_mode()
    resolved_colors = _resolve_theme_colors(_with_theme_color_fallbacks(theme_json["colors"]), theme_json.get("vars"))
    fg_colors, bg_colors = _split_theme_colors(resolved_colors)
    return Theme(
        fg_colors,
        bg_colors,
        color_mode,
        {"name": theme_json["name"], "sourcePath": source_path, "appearance": theme_json.get("appearance")},
    )


def _create_system_theme(mode: str | None = None) -> Theme:
    """Generate the system theme from the terminal's reported colors
    (grayscale while they are pending)."""
    with _theme_state_lock:
        terminal = _terminal_colors
        pending = _terminal_colors_pending
        scheme = _terminal_color_scheme
    generated = generate_system_theme_colors(
        {**terminal, "saturation": 0 if pending else 1, "appearanceHint": detect_terminal_theme(terminal, scheme)}
    )
    fg_colors, bg_colors = _split_theme_colors(generated["colors"])
    return Theme(
        fg_colors,
        bg_colors,
        mode or get_terminal_color_mode(),
        {"name": SYSTEM_THEME_NAME, "appearance": generated["appearance"], "dim": generated["dim"]},
    )


def _load_theme_from_path_blocking(theme_path: str, mode: str | None = None) -> Theme:
    """Blocking read+parse. Only for callers already off the runtime.

    The theme watcher's reload calls this through `spawn_blocking`.
    """
    with open(theme_path, encoding="utf-8") as f:
        content = f.read()
    theme_json = _parse_theme_json_content(theme_path, content)
    return _create_theme(theme_json, mode, theme_path)


def load_theme_from_path(theme_path: str, mode: str | None = None) -> Awaitable[Theme]:
    return tonio.spawn_blocking(_load_theme_from_path_blocking, theme_path, mode)


async def _load_theme(name: str, mode: str | None = None) -> Theme:
    # The system theme name is reserved: it takes precedence over custom themes of the same name.
    if name == SYSTEM_THEME_NAME:
        return _create_system_theme(mode)
    registered_theme = _registered_themes.get(name)
    if registered_theme is not None:
        return registered_theme
    theme_json = await _load_theme_json(name)
    return _create_theme(theme_json, mode)


async def get_theme_by_name(name: str) -> Theme | None:
    try:
        return await _load_theme(name)
    except Exception:
        return None


def parse_auto_theme_setting(theme_setting: str | None) -> dict | None:
    """Parse ``"light-name/dark-name"`` settings into a record, else None."""
    if not theme_setting:
        return None
    slash_index = theme_setting.find("/")
    if slash_index == -1 or theme_setting.find("/", slash_index + 1) != -1:
        return None

    light_theme = theme_setting[:slash_index].strip()
    dark_theme = theme_setting[slash_index + 1 :].strip()
    if not light_theme or not dark_theme:
        return None
    return {"lightTheme": light_theme, "darkTheme": dark_theme}


def resolve_theme_setting(theme_setting: str | None, terminal_theme: str) -> str | None:
    auto_theme = parse_auto_theme_setting(theme_setting)
    if auto_theme:
        return auto_theme["lightTheme"] if terminal_theme == "light" else auto_theme["darkTheme"]
    if theme_setting is not None and "/" in theme_setting:
        return None
    return theme_setting


_COLORFGBG_INDEX_RE = re.compile(r"^\d{1,2}$")


def detect_color_fg_bg_theme(env=None) -> str | None:
    """Dark or light from the `COLORFGBG` environment variable some terminals
    set, or None without a usable background index. The value is ``fg;bg`` or
    ``fg;xpm;bg`` (rxvt), where a field is an ANSI color index or ``default``
    when the color is not in the palette. The index refers to the terminal's
    own palette, whose colors are unknown here, so it is classified by index
    like Vim does: 0-6 and 8 (bright black, e.g. Solarized Dark's background)
    are dark, 7 and 9-15 are light."""
    if env is None:
        env = os.environ
    colorfgbg = env.get("COLORFGBG")
    bg = colorfgbg.split(";")[-1].strip() if colorfgbg is not None else None
    if not bg or not _COLORFGBG_INDEX_RE.match(bg):
        return None
    index = int(bg)
    if index > 15:
        return None
    return "dark" if index <= 6 or index == 8 else "light"


def detect_terminal_theme(colors: dict | None = None, reported_scheme: str | None = None, env=None) -> str:
    """Whether the terminal is dark or light. The background it renders
    decides, classified the same way the system theme does. Without a
    reported background: the terminal's light/dark report, then COLORFGBG,
    then dark."""
    colors = colors or {}
    background = colors.get("background")
    if background:
        return terminal_appearance(background, colors.get("foreground"))
    if reported_scheme is not None:
        return reported_scheme
    return detect_color_fg_bg_theme(env) or "dark"


def get_terminal_theme() -> str:
    """Whether the terminal is dark or light, from everything it reported so
    far. See `detect_terminal_theme()`."""
    with _theme_state_lock:
        terminal = _terminal_colors
        scheme = _terminal_color_scheme
    return detect_terminal_theme(terminal, scheme)


# ============================================================================
# Global Theme Instance
# ============================================================================

_current_theme: Theme | None = None


class _ThemeProxy:
    """Delegates to the active global theme (pi's globalThis Proxy)."""

    __slots__ = ()

    def __getattr__(self, name: str):
        if _current_theme is None:
            raise RuntimeError("Theme not initialized. Call init_theme() first.")
        return getattr(_current_theme, name)


theme = _ThemeProxy()


def _set_global_theme(theme_instance: Theme) -> None:
    global _current_theme
    _current_theme = theme_instance


# Watcher/reload state may be touched from watcher threads.
_theme_state_lock = threading.RLock()
_current_theme_name: str | None = None
_theme_watcher = None
_theme_reload_timer: Timeout | None = None
_on_theme_change_callback = None
# Copy-on-write: replaced whole under `_theme_state_lock`, never changed in
# place, so a reader iterating the dict it read cannot see it change.
_registered_themes: dict = {}


def set_registered_themes(themes: list) -> None:
    global _registered_themes
    registered = {}
    for theme_instance in themes:
        if theme_instance.name:
            _assert_theme_name_is_valid(theme_instance.name)
            registered[theme_instance.name] = theme_instance
    with _theme_state_lock:
        _registered_themes = registered


async def init_theme(theme_name: str | None = None, enable_watcher: bool = False) -> None:
    global _current_theme_name
    name = theme_name if theme_name is not None else SYSTEM_THEME_NAME
    loaded, fallback = await _load_theme_or_fallback(name)
    with _theme_state_lock:
        _current_theme_name = SYSTEM_THEME_NAME if fallback else name
        _set_global_theme(loaded)
    # No watcher for the fallback theme.
    if enable_watcher and not fallback:
        await _start_theme_watcher()


async def _load_theme_or_fallback(name: str) -> tuple[Theme, str | None]:
    """Load `name`, or the system theme if it is invalid.

    Both loads happen here, deliberately outside `_theme_state_lock`: the lock
    guards in-memory theme state only and must never be held across an await.
    Returns the theme plus the error that forced a fallback (None on success).
    """
    try:
        return await _load_theme(name), None
    except Exception as error:
        return await _load_theme(SYSTEM_THEME_NAME), str(error)


def _change_theme(apply: Callable[[], bool]) -> None:
    """Swap the theme through the registered change callback, which runs
    `apply` under the host's UI lock together with its refresh, so a frame
    never mixes two themes (spec/ui-island.md, whole changes). With no callback
    registered (the startup screens) the swap happens here."""
    with _theme_state_lock:
        callback = _on_theme_change_callback
    # Outside the lock: the callback takes the host's lock, then `apply`
    # takes this one (the host's lock always comes first).
    if callback is None:
        apply()
    else:
        callback(apply)


async def set_theme(name: str, enable_watcher: bool = False) -> dict:
    loaded, error = await _load_theme_or_fallback(name)

    def apply() -> bool:
        global _current_theme_name
        with _theme_state_lock:
            _current_theme_name = SYSTEM_THEME_NAME if error else name
            _set_global_theme(loaded)
        # pi notifies only a successful change (its fallback swaps silently).
        return not error

    _change_theme(apply)
    if error:
        return {"success": False, "error": error}
    if enable_watcher:
        await _start_theme_watcher()
    return {"success": True}


def set_theme_instance(theme_instance: Theme) -> None:
    def apply() -> bool:
        global _current_theme_name
        with _theme_state_lock:
            _set_global_theme(theme_instance)
            _current_theme_name = "<in-memory>"
            stop_theme_watcher()  # Can't watch a direct instance
        return True

    _change_theme(apply)


def on_theme_change(callback) -> None:
    """Register the host's ``callback(apply)``. A theme change (`set_theme`,
    `set_theme_instance`, the theme-file reload) calls it on the changing
    task; the host runs ``apply()`` under its UI lock, which swaps the theme
    and returns whether the change must be shown (invalidate and render)."""
    global _on_theme_change_callback
    _on_theme_change_callback = callback


async def _start_theme_watcher() -> None:
    """Watch the current custom theme's file. Call outside `_theme_state_lock`:
    the baseline snapshot is filesystem I/O (pool hop), and the watcher is
    adopted only if the theme is still the one it was started for."""
    global _theme_watcher
    stop_theme_watcher()
    with _theme_state_lock:
        watched_theme_name = _current_theme_name

    # Only watch if it's a custom theme (not built-in)
    if not watched_theme_name or watched_theme_name in ("dark", "light", SYSTEM_THEME_NAME):
        return

    custom_themes_dir = get_custom_themes_dir()
    watched_file_name = f"{watched_theme_name}.json"
    theme_file = os.path.join(custom_themes_dir, watched_file_name)

    # Only watch if the file exists
    if not await tonio.spawn_blocking(os.path.exists, theme_file):
        return

    def _reload_from_disk_blocking() -> Theme | None:
        # Keep the last successfully loaded theme active if the file is
        # temporarily missing or in an invalid state while being edited.
        if not os.path.exists(theme_file):
            return None
        try:
            return _load_theme_from_path_blocking(theme_file)
        except Exception:
            return None

    def reload_theme() -> None:
        # A `_timers.Timeout` fire; pi reads the file synchronously, here the
        # read is a pool hop on its own task.
        global _theme_reload_timer
        with _theme_state_lock:
            _theme_reload_timer = None
            # Ignore stale timers after switching themes or stopping the watcher
            if _current_theme_name != watched_theme_name:
                return
        tonio.spawn.without_tracking(apply_reloaded_theme())

    async def apply_reloaded_theme() -> None:
        reloaded_theme = await tonio.spawn_blocking(_reload_from_disk_blocking)
        if reloaded_theme is None:
            return

        def apply() -> bool:
            global _registered_themes
            with _theme_state_lock:
                if _current_theme_name != watched_theme_name:
                    return False
                # Refresh the registry cache and notify (to invalidate UI)
                _registered_themes = {**_registered_themes, watched_theme_name: reloaded_theme}
                _set_global_theme(reloaded_theme)
                return True

        _change_theme(apply)

    def schedule_reload() -> None:
        global _theme_reload_timer
        with _theme_state_lock:
            if _theme_reload_timer is not None:
                _theme_reload_timer.cancel()
            _theme_reload_timer = Timeout(100, reload_theme)

    def on_watch_event(_event_type: str, filename: str | None) -> None:
        with _theme_state_lock:
            if _current_theme_name != watched_theme_name:
                return
            if not filename:
                schedule_reload()
                return
            if filename != watched_file_name:
                return
            schedule_reload()

    def on_watch_error() -> None:
        global _theme_watcher
        with _theme_state_lock:
            close_watcher(_theme_watcher)
            _theme_watcher = None

    watcher = await watch_with_error_handler(custom_themes_dir, on_watch_event, on_watch_error)
    with _theme_state_lock:
        if _current_theme_name != watched_theme_name or _theme_watcher is not None:
            close_watcher(watcher)  # the theme changed (or was re-watched) meanwhile
            return
        _theme_watcher = watcher


def stop_theme_watcher() -> None:
    global _theme_reload_timer, _theme_watcher
    with _theme_state_lock:
        if _theme_reload_timer is not None:
            _theme_reload_timer.cancel()
            _theme_reload_timer = None
        close_watcher(_theme_watcher)
        _theme_watcher = None


# ============================================================================
# HTML Export Helpers
# ============================================================================


async def get_resolved_theme_colors(theme_name: str | None = None) -> dict:
    """Get resolved theme colors as CSS-compatible hex strings.

    Used by HTML export to generate CSS custom properties.
    """
    loaded = await _load_theme(theme_name or _current_theme_name or SYSTEM_THEME_NAME)
    return {token: color_to_hex(color) for token, color in loaded.colors.items()}


async def is_light_theme(theme_name: str | None = None) -> bool:
    """Check if a theme is a "light" theme (for CSS that needs light/dark variants)."""
    loaded = await _load_theme(theme_name or _current_theme_name or SYSTEM_THEME_NAME)
    return loaded.appearance == "light"


async def get_theme_export_colors(theme_name: str | None = None) -> dict:
    """Get explicit export colors from theme JSON, if specified.

    Returns None for each color that isn't explicitly set.
    """
    name = theme_name or _current_theme_name or SYSTEM_THEME_NAME
    if name == SYSTEM_THEME_NAME:
        return {}
    try:
        theme_json = await _load_theme_json(name)
        export_section = theme_json.get("export")
        if not export_section:
            return {}

        vars_map = theme_json.get("vars") or {}

        # Export colors end up in CSS, which understands hex and oklch()
        # values directly but not okhsl().
        def resolve(value):
            if value is None:
                return None
            resolved = _resolve_var_refs(value, vars_map)
            if isinstance(resolved, int):
                return color_to_hex(indexed_color(resolved))
            if resolved == "":
                return None
            if _OKHSL_RE.match(resolved):
                return color_to_hex(parse_color(resolved))
            return resolved

        return {
            "pageBg": resolve(export_section.get("pageBg")),
            "cardBg": resolve(export_section.get("cardBg")),
            "infoBg": resolve(export_section.get("infoBg")),
        }
    except Exception:
        return {}


# ============================================================================
# TUI Helpers
# ============================================================================

_cached_highlight_theme_for: Theme | None = None
_cached_cli_highlight_theme: dict | None = None


def _build_cli_highlight_theme(t: Theme) -> dict:
    return {
        "keyword": lambda s: t.fg("syntaxKeyword", s),
        "built_in": lambda s: t.fg("syntaxType", s),
        "literal": lambda s: t.fg("syntaxNumber", s),
        "number": lambda s: t.fg("syntaxNumber", s),
        "regexp": lambda s: t.fg("syntaxString", s),
        "string": lambda s: t.fg("syntaxString", s),
        "comment": lambda s: t.fg("syntaxComment", s),
        "doctag": lambda s: t.fg("syntaxComment", s),
        "meta": lambda s: t.fg("muted", s),
        "function": lambda s: t.fg("syntaxFunction", s),
        "title": lambda s: t.fg("syntaxFunction", s),
        "class": lambda s: t.fg("syntaxType", s),
        "type": lambda s: t.fg("syntaxType", s),
        "tag": lambda s: t.fg("syntaxPunctuation", s),
        "name": lambda s: t.fg("syntaxKeyword", s),
        "attr": lambda s: t.fg("syntaxVariable", s),
        "variable": lambda s: t.fg("syntaxVariable", s),
        "params": lambda s: t.fg("syntaxVariable", s),
        "operator": lambda s: t.fg("syntaxOperator", s),
        "punctuation": lambda s: t.fg("syntaxPunctuation", s),
        "emphasis": lambda s: t.italic(s),
        "strong": lambda s: t.bold(s),
        "link": lambda s: t.underline(s),
        "addition": lambda s: t.fg("toolDiffAdded", s),
        "deletion": lambda s: t.fg("toolDiffRemoved", s),
    }


def _get_cli_highlight_theme(t: Theme) -> dict:
    global _cached_highlight_theme_for, _cached_cli_highlight_theme
    if _cached_highlight_theme_for is not t or _cached_cli_highlight_theme is None:
        _cached_highlight_theme_for = t
        _cached_cli_highlight_theme = _build_cli_highlight_theme(t)
    return _cached_cli_highlight_theme


def highlight_code(code: str, lang: str | None = None) -> list:
    """Highlight code with syntax coloring; returns highlighted lines."""
    # Validate language before highlighting to avoid highlighting with a
    # bogus lexer
    valid_lang = lang if lang and supports_language(lang) else None
    # Skip highlighting when no valid language is specified: auto-detection
    # is unreliable and can misidentify prose, coloring random English words
    # as keywords.
    if not valid_lang:
        return [theme.fg("mdCodeBlock", line) for line in code.split("\n")]
    try:
        return highlight(
            code,
            language=valid_lang,
            ignore_illegals=True,
            theme=_get_cli_highlight_theme(_current_theme),
        ).split("\n")
    except Exception:
        return code.split("\n")


_EXT_TO_LANG = {
    "ts": "typescript",
    "tsx": "typescript",
    "js": "javascript",
    "jsx": "javascript",
    "mjs": "javascript",
    "cjs": "javascript",
    "py": "python",
    "rb": "ruby",
    "rs": "rust",
    "go": "go",
    "java": "java",
    "kt": "kotlin",
    "swift": "swift",
    "c": "c",
    "h": "c",
    "cpp": "cpp",
    "cc": "cpp",
    "cxx": "cpp",
    "hpp": "cpp",
    "cs": "csharp",
    "php": "php",
    "sh": "bash",
    "bash": "bash",
    "zsh": "bash",
    "fish": "fish",
    "ps1": "powershell",
    "sql": "sql",
    "html": "html",
    "htm": "html",
    "css": "css",
    "scss": "scss",
    "sass": "sass",
    "less": "less",
    "json": "json",
    "yaml": "yaml",
    "yml": "yaml",
    "toml": "toml",
    "xml": "xml",
    "md": "markdown",
    "markdown": "markdown",
    "dockerfile": "dockerfile",
    "makefile": "makefile",
    "cmake": "cmake",
    "lua": "lua",
    "perl": "perl",
    "r": "r",
    "scala": "scala",
    "clj": "clojure",
    "ex": "elixir",
    "exs": "elixir",
    "erl": "erlang",
    "hs": "haskell",
    "ml": "ocaml",
    "vim": "vim",
    "graphql": "graphql",
    "proto": "protobuf",
    "tf": "hcl",
    "hcl": "hcl",
}


def get_language_from_path(file_path: str) -> str | None:
    """Get language identifier from file path extension."""
    # JS split(".").pop(): the whole string when there is no dot
    ext = file_path.rsplit(".", 1)[-1].lower()
    if not ext:
        return None
    return _EXT_TO_LANG.get(ext)


def get_markdown_theme() -> dict:
    return {
        "heading": lambda text: theme.fg("mdHeading", text),
        "link": lambda text: theme.fg("mdLink", text),
        "linkUrl": lambda text: theme.fg("mdLinkUrl", text),
        "code": lambda text: theme.fg("mdCode", text),
        "codeBlock": lambda text: theme.fg("mdCodeBlock", text),
        "codeBlockBorder": lambda text: theme.fg("mdCodeBlockBorder", text),
        "quote": lambda text: theme.fg("mdQuote", text),
        "quoteBorder": lambda text: theme.fg("mdQuoteBorder", text),
        "hr": lambda text: theme.fg("mdHr", text),
        "listBullet": lambda text: theme.fg("mdListBullet", text),
        "bold": lambda text: theme.bold(text),
        "italic": lambda text: theme.italic(text),
        "underline": lambda text: theme.underline(text),
        "strikethrough": lambda text: chalk.strikethrough(text),
        "highlightCode": _markdown_highlight_code,
    }


def _markdown_highlight_code(code: str, lang: str | None = None) -> list:
    valid_lang = lang if lang and supports_language(lang) else None
    if not valid_lang:
        return [theme.fg("mdCodeBlock", line) for line in code.split("\n")]
    try:
        return highlight(
            code,
            language=valid_lang,
            ignore_illegals=True,
            theme=_get_cli_highlight_theme(_current_theme),
        ).split("\n")
    except Exception:
        return [theme.fg("mdCodeBlock", line) for line in code.split("\n")]


def get_select_list_theme() -> dict:
    return {
        "selectedPrefix": lambda text: theme.fg("accent", text),
        "selectedText": lambda text: theme.fg("accent", text),
        "description": lambda text: theme.fg("muted", text),
        "scrollInfo": lambda text: theme.fg("muted", text),
        "noMatch": lambda text: theme.fg("muted", text),
    }


def get_editor_theme() -> dict:
    return {
        "borderColor": lambda text: theme.fg("borderMuted", text),
        "selectList": get_select_list_theme(),
    }


def get_settings_list_theme() -> dict:
    return {
        "label": lambda text, selected: theme.fg("accent", text) if selected else text,
        "value": lambda text, selected: theme.fg("accent", text) if selected else theme.fg("muted", text),
        "description": lambda text: theme.fg("dim", text),
        "cursor": theme.fg("accent", "→ "),
        "hint": lambda text: theme.fg("dim", text),
    }
