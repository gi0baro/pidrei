"""Mirror of pi coding-agent src/modes/interactive/theme/theme.ts.

Theme records mirror pi's JSON shape (camelCase color keys). The global
``theme`` export is a proxy over the module-level current theme (pi uses a
globalThis Proxy for the same purpose); ``init_theme()`` must run first.

Deviations:
- Validation is hand-rolled against pi's typebox schema (same required color
  set, same error message layout) instead of a JSON-schema engine.
- The custom-theme watcher polls (utils/fs_watch) instead of node fs.watch;
  its events and the 100 ms debounced reload (a ``_timers.Timeout``) run on
  the TUI's owner task with the file read on the pool. The module lock
  guards the theme globals against `set_theme` callers on other tasks.
- Syntax highlighting is pygments (utils/syntax_highlight), keyed by the same
  scope names pi feeds cli-highlight.
"""

import contextlib
import json
import math
import os
import re
import threading
from collections.abc import Awaitable

import tonio.colored as tonio
from tonio.colored import fs

from pidrei_tui import get_capabilities
from pidrei_tui._timers import Timeout

from ....config import get_custom_themes_dir, get_themes_dir
from ....utils import colors as chalk
from ....utils.fs_watch import close_watcher, watch_with_error_handler
from ....utils.syntax_highlight import highlight, supports_language
from ....utils.text import strip_bom


# ============================================================================
# Types & Schema
# ============================================================================

# The schema that validates the theme document shape lives in `theme_json.py`
# (ColorValue: hex "#ff0000", var ref "primary", empty "", or 256-color index).

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


THEME_BG_KEYS = (
    "selectedBg",
    "scrollbarThumb",
    "searchMatchBg",
    "userMessageBg",
    "customMessageBg",
    "toolPendingBg",
    "toolSuccessBg",
    "toolErrorBg",
)


# ============================================================================
# Color Utilities
# ============================================================================


def _hex_to_rgb(hex_color: str) -> dict:
    cleaned = hex_color.replace("#", "")
    if len(cleaned) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    try:
        r = int(cleaned[0:2], 16)
        g = int(cleaned[2:4], 16)
        b = int(cleaned[4:6], 16)
    except ValueError:
        raise ValueError(f"Invalid hex color: {hex_color}") from None
    return {"r": r, "g": g, "b": b}


# The 6x6x6 color cube channel values (indices 0-5)
_CUBE_VALUES = [0, 95, 135, 175, 215, 255]

# Grayscale ramp values (indices 232-255, 24 grays from 8 to 238)
_GRAY_VALUES = [8 + i * 10 for i in range(24)]


def _find_closest_index(values: list, target: int) -> int:
    min_dist = math.inf
    min_idx = 0
    for i, value in enumerate(values):
        dist = abs(target - value)
        if dist < min_dist:
            min_dist = dist
            min_idx = i
    return min_idx


def _color_distance(r1, g1, b1, r2, g2, b2) -> float:
    # Weighted Euclidean distance (human eye is more sensitive to green)
    dr = r1 - r2
    dg = g1 - g2
    db = b1 - b2
    return dr * dr * 0.299 + dg * dg * 0.587 + db * db * 0.114


def _rgb_to_256(r: int, g: int, b: int) -> int:
    # Find closest color in the 6x6x6 cube
    r_idx = _find_closest_index(_CUBE_VALUES, r)
    g_idx = _find_closest_index(_CUBE_VALUES, g)
    b_idx = _find_closest_index(_CUBE_VALUES, b)
    cube_r = _CUBE_VALUES[r_idx]
    cube_g = _CUBE_VALUES[g_idx]
    cube_b = _CUBE_VALUES[b_idx]
    cube_index = 16 + 36 * r_idx + 6 * g_idx + b_idx
    cube_dist = _color_distance(r, g, b, cube_r, cube_g, cube_b)

    # Find closest grayscale
    gray = round(0.299 * r + 0.587 * g + 0.114 * b)
    gray_idx = _find_closest_index(_GRAY_VALUES, gray)
    gray_value = _GRAY_VALUES[gray_idx]
    gray_index = 232 + gray_idx
    gray_dist = _color_distance(r, g, b, gray_value, gray_value, gray_value)

    # Only consider grayscale if color is nearly neutral (spread < 10)
    # AND grayscale is actually closer
    spread = max(r, g, b) - min(r, g, b)
    if spread < 10 and gray_dist < cube_dist:
        return gray_index

    return cube_index


def _hex_to_256(hex_color: str) -> int:
    rgb = _hex_to_rgb(hex_color)
    return _rgb_to_256(rgb["r"], rgb["g"], rgb["b"])


def _fg_ansi(color, mode: str) -> str:
    if color == "":
        return "\x1b[39m"
    if isinstance(color, int):
        return f"\x1b[38;5;{color}m"
    if color.startswith("#"):
        if mode == "truecolor":
            rgb = _hex_to_rgb(color)
            return f"\x1b[38;2;{rgb['r']};{rgb['g']};{rgb['b']}m"
        return f"\x1b[38;5;{_hex_to_256(color)}m"
    raise ValueError(f"Invalid color value: {color}")


def _bg_ansi(color, mode: str) -> str:
    if color == "":
        return "\x1b[49m"
    if isinstance(color, int):
        return f"\x1b[48;5;{color}m"
    if color.startswith("#"):
        if mode == "truecolor":
            rgb = _hex_to_rgb(color)
            return f"\x1b[48;2;{rgb['r']};{rgb['g']};{rgb['b']}m"
        return f"\x1b[48;5;{_hex_to_256(color)}m"
    raise ValueError(f"Invalid color value: {color}")


def _resolve_var_refs(value, vars_map: dict, visited: set | None = None):
    if isinstance(value, int) or value == "" or value.startswith("#"):
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
    fallback = colors.get("thinkingMax")
    if fallback is None:
        fallback = colors["thinkingXhigh"]
    scrollbar_thumb = colors.get("scrollbarThumb")
    if scrollbar_thumb is None:
        scrollbar_thumb = colors["selectedBg"]
    search_match_bg = colors.get("searchMatchBg")
    if search_match_bg is None:
        search_match_bg = colors["selectedBg"]
    search_match_text = colors.get("searchMatchText")
    if search_match_text is None:
        search_match_text = colors["text"]
    return {
        **colors,
        "thinkingMax": fallback,
        "scrollbarThumb": scrollbar_thumb,
        "searchMatchBg": search_match_bg,
        "searchMatchText": search_match_text,
    }


# ============================================================================
# Theme Class
# ============================================================================


class Theme:
    def __init__(self, fg_colors: dict, bg_colors: dict, mode: str, options: dict | None = None):
        options = options or {}
        self.name = options.get("name")
        self.source_path = options.get("sourcePath")
        self.source_info = options.get("sourceInfo")
        self._mode = mode
        thinking_max = fg_colors.get("thinkingMax")
        if thinking_max is None:
            thinking_max = fg_colors["thinkingXhigh"]
        search_match_text = fg_colors.get("searchMatchText")
        if search_match_text is None:
            search_match_text = fg_colors["text"]
        self._fg_colors = {
            key: _fg_ansi(value, mode)
            for key, value in {**fg_colors, "thinkingMax": thinking_max, "searchMatchText": search_match_text}.items()
        }
        backgrounds = {
            **bg_colors,
            "scrollbarThumb": bg_colors.get("scrollbarThumb") or bg_colors["selectedBg"],
            "searchMatchBg": bg_colors.get("searchMatchBg") or bg_colors["selectedBg"],
        }
        self._bg_colors = {key: _bg_ansi(value, mode) for key, value in backgrounds.items()}

    def fg(self, color: str, text: str) -> str:
        ansi = self._fg_colors.get(color)
        if not ansi:
            raise ValueError(f"Unknown theme color: {color}")
        return f"{ansi}{text}\x1b[39m"  # Reset only foreground color

    def bg(self, color: str, text: str) -> str:
        ansi = self._bg_colors.get(color)
        if not ansi:
            raise ValueError(f"Unknown theme background color: {color}")
        return f"{ansi}{text}\x1b[49m"  # Reset only background color

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
        ansi = self._fg_colors.get(color)
        if not ansi:
            raise ValueError(f"Unknown theme color: {color}")
        return ansi

    def get_bg_ansi(self, color: str) -> str:
        ansi = self._bg_colors.get(color)
        if not ansi:
            raise ValueError(f"Unknown theme background color: {color}")
        return ansi

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
    because it is reached from a sync TUI callback.
    """
    await tonio.spawn_blocking(_get_builtin_themes)


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

    # Built-in themes
    for name in _get_builtin_themes():
        add_theme({"name": name, "path": os.path.join(themes_dir, f"{name}.json")})

    # Custom themes
    for theme_info in await _get_custom_theme_infos():
        add_theme(theme_info)

    for name, registered in _registered_themes.items():
        add_theme({"name": name, "path": registered.source_path})

    return sorted(result, key=lambda info: (info["name"].lower(), info["name"]))


def _scan_custom_theme_dir(custom_themes_dir: str) -> list[str]:
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
    entries = await tonio.spawn_blocking(_scan_custom_theme_dir, custom_themes_dir)
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


def _create_theme(theme_json: dict, mode: str | None = None, source_path: str | None = None) -> Theme:
    color_mode = mode or ("truecolor" if get_capabilities()["trueColor"] else "256color")
    resolved_colors = _resolve_theme_colors(_with_theme_color_fallbacks(theme_json["colors"]), theme_json.get("vars"))
    fg_colors: dict = {}
    bg_colors: dict = {}
    for key, value in resolved_colors.items():
        if key in THEME_BG_KEYS:
            bg_colors[key] = value
        else:
            fg_colors[key] = value
    return Theme(fg_colors, bg_colors, color_mode, {"name": theme_json["name"], "sourcePath": source_path})


def _load_theme_from_path_sync(theme_path: str, mode: str | None = None) -> Theme:
    """Blocking read+parse. Only for callers already off the runtime.

    The theme watcher's reload calls this through `spawn_blocking`.
    """
    with open(theme_path, encoding="utf-8") as f:
        content = f.read()
    theme_json = _parse_theme_json_content(theme_path, content)
    return _create_theme(theme_json, mode, theme_path)


def load_theme_from_path(theme_path: str, mode: str | None = None) -> Awaitable[Theme]:
    return tonio.spawn_blocking(_load_theme_from_path_sync, theme_path, mode)


async def _load_theme(name: str, mode: str | None = None) -> Theme:
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


_INT_PREFIX_RE = re.compile(r"^[+-]?\d+")


def _get_colorfgbg_background_index(colorfgbg: str) -> int | None:
    parts = colorfgbg.split(";")
    for part in reversed(parts):
        # JS parseInt: leading integer prefix, NaN otherwise
        match = _INT_PREFIX_RE.match(part.strip())
        if match is None:
            continue
        bg = int(match.group(0))
        if 0 <= bg <= 255:
            return bg
    return None


def _get_rgb_color_luminance(rgb: dict) -> float:
    def to_linear(channel: float) -> float:
        value = channel / 255
        return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4

    return 0.2126 * to_linear(rgb["r"]) + 0.7152 * to_linear(rgb["g"]) + 0.0722 * to_linear(rgb["b"])


def _get_ansi_color_luminance(index: int) -> float:
    return _get_rgb_color_luminance(_hex_to_rgb(_ansi_256_to_hex(index)))


def get_theme_for_rgb_color(rgb: dict) -> str:
    return "light" if _get_rgb_color_luminance(rgb) >= 0.5 else "dark"


def detect_terminal_background_from_env(options: dict | None = None) -> dict:
    """Detect the terminal theme from environment hints.

    Returns ``{"theme", "source", "detail", "confidence"}``.
    """
    options = options or {}
    env = options.get("env")
    if env is None:
        env = os.environ
    colorfgbg = env.get("COLORFGBG") or ""
    bg = _get_colorfgbg_background_index(colorfgbg)
    if bg is not None:
        return {
            "theme": "light" if _get_ansi_color_luminance(bg) >= 0.5 else "dark",
            "source": "COLORFGBG",
            "detail": f"background color index {bg}",
            "confidence": "high",
        }

    return {
        "theme": "dark",
        "source": "fallback",
        "detail": "no terminal background hint found",
        "confidence": "low",
    }


async def detect_terminal_background_theme(options: dict) -> dict:
    """Detect via an OSC 11 query (``options["ui"]``), falling back to env."""
    try:
        rgb = await options["ui"].query_terminal_background_color(timeout_ms=options["timeoutMs"])
        if rgb:
            return {
                "theme": get_theme_for_rgb_color(rgb),
                "source": "terminal background",
                "detail": f"OSC 11 background rgb({rgb['r']}, {rgb['g']}, {rgb['b']})",
                "confidence": "high",
            }
    except Exception:
        # Fall back to environment-based detection when the terminal query fails.
        pass

    return detect_terminal_background_from_env({"env": options.get("env")})


async def detect_terminal_theme_for_auto(options: dict) -> str:
    # Both probes are started before either is awaited, so an unsupported
    # color-scheme DSR costs its timeout only once instead of serializing ahead
    # of the OSC 11 fallback.
    color_scheme_task = None
    query_color_scheme = getattr(options["ui"], "query_terminal_color_scheme", None)
    if query_color_scheme is not None:
        try:
            color_scheme_task = tonio.spawn(query_color_scheme(timeout_ms=options["timeoutMs"]))
        except Exception:
            # Fall back to OSC 11 / COLORFGBG detection when starting the
            # color-scheme query fails.
            color_scheme_task = None
    background_task = tonio.spawn(detect_terminal_background_theme(options))

    if color_scheme_task is not None:
        try:
            color_scheme = await color_scheme_task
            if color_scheme:
                return color_scheme
        except Exception:
            # Fall back to the concurrently queried OSC 11 / COLORFGBG detection.
            pass
    return (await background_task)["theme"]


def get_default_theme() -> str:
    return detect_terminal_background_from_env()["theme"]


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
_registered_themes: dict = {}


def set_registered_themes(themes: list) -> None:
    _registered_themes.clear()
    for theme_instance in themes:
        if theme_instance.name:
            _assert_theme_name_is_valid(theme_instance.name)
            _registered_themes[theme_instance.name] = theme_instance


def init_theme_sync(theme_name: str | None = None, enable_watcher: bool = False) -> None:
    """Blocking theme init. Only for callers already off the runtime.

    Test fixtures run outside `tonio.run`, which is outside the never-block
    rule by construction; pytest also cannot drive an async autouse
    fixture. Production code uses the async `init_theme`. The watcher needs
    the runtime (timers and the pool), so it cannot be enabled from here.
    """
    global _current_theme_name
    if enable_watcher:
        raise ValueError("the theme watcher needs the runtime; use init_theme")
    name = theme_name if theme_name is not None else get_default_theme()
    try:
        loaded, fallback = _load_theme_sync(name), None
    except Exception as error:
        loaded, fallback = _load_theme_sync("dark"), str(error)
    with _theme_state_lock:
        _current_theme_name = "dark" if fallback else name
        _set_global_theme(loaded)


def _load_theme_sync(name: str, mode: str | None = None) -> Theme:
    registered_theme = _registered_themes.get(name)
    if registered_theme is not None:
        return registered_theme
    builtin_themes = _get_builtin_themes()
    if name in builtin_themes:
        return _create_theme(builtin_themes[name], mode)
    theme_path = os.path.join(get_custom_themes_dir(), f"{name}.json")
    if not os.path.exists(theme_path):
        raise ValueError(f"Theme not found: {name}")
    return _load_theme_from_path_sync(theme_path, mode)


async def init_theme(theme_name: str | None = None, enable_watcher: bool = False) -> None:
    global _current_theme_name
    name = theme_name if theme_name is not None else get_default_theme()
    loaded, fallback = await _load_theme_or_fallback(name)
    with _theme_state_lock:
        _current_theme_name = "dark" if fallback else name
        _set_global_theme(loaded)
    # No watcher for the fallback theme.
    if enable_watcher and not fallback:
        await _start_theme_watcher()


async def _load_theme_or_fallback(name: str) -> tuple[Theme, str | None]:
    """Load `name`, or the dark theme if it is invalid.

    Both reads happen here, deliberately outside `_theme_state_lock`: the lock
    guards in-memory theme state only and must never be held across an await.
    Returns the theme plus the error that forced a fallback (None on success).
    """
    try:
        return await _load_theme(name), None
    except Exception as error:
        return await _load_theme("dark"), str(error)


async def set_theme(name: str, enable_watcher: bool = False) -> dict:
    global _current_theme_name
    loaded, error = await _load_theme_or_fallback(name)
    with _theme_state_lock:
        _current_theme_name = "dark" if error else name
        _set_global_theme(loaded)
        callback = _on_theme_change_callback
    # Outside the lock: the watcher start awaits, the callback re-enters UI code.
    if error:
        return {"success": False, "error": error}
    if enable_watcher:
        await _start_theme_watcher()
    if callback is not None:
        callback()
    return {"success": True}


def set_theme_instance(theme_instance: Theme) -> None:
    global _current_theme_name
    with _theme_state_lock:
        _set_global_theme(theme_instance)
        _current_theme_name = "<in-memory>"
        stop_theme_watcher()  # Can't watch a direct instance
        if _on_theme_change_callback is not None:
            _on_theme_change_callback()


def on_theme_change(callback) -> None:
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
    if not watched_theme_name or watched_theme_name in ("dark", "light"):
        return

    custom_themes_dir = get_custom_themes_dir()
    watched_file_name = f"{watched_theme_name}.json"
    theme_file = os.path.join(custom_themes_dir, watched_file_name)

    # Only watch if the file exists
    if not await tonio.spawn_blocking(os.path.exists, theme_file):
        return

    def _reload_from_disk() -> Theme | None:
        # Keep the last successfully loaded theme active if the file is
        # temporarily missing or in an invalid state while being edited.
        if not os.path.exists(theme_file):
            return None
        try:
            return _load_theme_from_path_sync(theme_file)
        except Exception:
            return None

    async def reload_theme() -> None:
        # On the UI owner (a `_timers.Timeout` fire); the read is a pool hop.
        global _theme_reload_timer
        with _theme_state_lock:
            _theme_reload_timer = None
            # Ignore stale timers after switching themes or stopping the watcher
            if _current_theme_name != watched_theme_name:
                return
        reloaded_theme = await tonio.spawn_blocking(_reload_from_disk)
        if reloaded_theme is None:
            return
        with _theme_state_lock:
            if _current_theme_name != watched_theme_name:
                return
            # Refresh the registry cache and notify (to invalidate UI)
            _registered_themes[watched_theme_name] = reloaded_theme
            _set_global_theme(reloaded_theme)
            callback = _on_theme_change_callback
        if callback is not None:
            callback()

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

# Basic colors (0-15) - approximate common terminal values
_BASIC_COLORS = [
    "#000000",
    "#800000",
    "#008000",
    "#808000",
    "#000080",
    "#800080",
    "#008080",
    "#c0c0c0",
    "#808080",
    "#ff0000",
    "#00ff00",
    "#ffff00",
    "#0000ff",
    "#ff00ff",
    "#00ffff",
    "#ffffff",
]


def _ansi_256_to_hex(index: int) -> str:
    """Convert a 256-color index to hex string.

    Indices 0-15: basic colors (approximate)
    Indices 16-231: 6x6x6 color cube
    Indices 232-255: grayscale ramp
    """
    if index < 16:
        return _BASIC_COLORS[index]

    # Color cube (16-231): 6x6x6 = 216 colors
    if index < 232:
        cube_index = index - 16
        r = cube_index // 36
        g = (cube_index % 36) // 6
        b = cube_index % 6

        def to_hex(n: int) -> str:
            return format(0 if n == 0 else 55 + n * 40, "02x")

        return f"#{to_hex(r)}{to_hex(g)}{to_hex(b)}"

    # Grayscale (232-255): 24 shades
    gray_hex = format(8 + (index - 232) * 10, "02x")
    return f"#{gray_hex}{gray_hex}{gray_hex}"


async def get_resolved_theme_colors(theme_name: str | None = None) -> dict:
    """Get resolved theme colors as CSS-compatible hex strings.

    Used by HTML export to generate CSS custom properties.
    """
    name = theme_name or _current_theme_name or get_default_theme()
    is_light = name == "light"
    theme_json = await _load_theme_json(name)
    resolved = _resolve_theme_colors(_with_theme_color_fallbacks(theme_json["colors"]), theme_json.get("vars"))

    # Default text color for empty values (terminal uses default fg color)
    default_text = "#000000" if is_light else "#e5e5e7"

    css_colors: dict = {}
    for key, value in resolved.items():
        if isinstance(value, int):
            css_colors[key] = _ansi_256_to_hex(value)
        elif value == "":
            # Empty means default terminal color - use sensible fallback for HTML
            css_colors[key] = default_text
        else:
            css_colors[key] = value
    return css_colors


def is_light_theme(theme_name: str | None = None) -> bool:
    """Check if a theme is a "light" theme (for CSS that needs variants)."""
    # Currently just check the name - could be extended to analyze colors
    return theme_name == "light"


async def get_theme_export_colors(theme_name: str | None = None) -> dict:
    """Get explicit export colors from theme JSON, if specified.

    Returns None for each color that isn't explicitly set.
    """
    name = theme_name or _current_theme_name or get_default_theme()
    try:
        theme_json = await _load_theme_json(name)
        export_section = theme_json.get("export")
        if not export_section:
            return {}

        vars_map = theme_json.get("vars") or {}

        def resolve(value):
            if value is None:
                return None
            resolved = _resolve_var_refs(value, vars_map)
            if isinstance(resolved, int):
                return _ansi_256_to_hex(resolved)
            if resolved == "":
                return None
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
