"""Mirror of pi coding-agent test/test-theme-colors.ts (manual CLI tool).

Not collected by pytest; run directly:
    uv run python packages/pidrei/tests/theme_colors_tool.py light|dark
    uv run python packages/pidrei/tests/theme_colors_tool.py contrast 4.5
    uv run python packages/pidrei/tests/theme_colors_tool.py test file.json
"""

import json
import os
import re
import sys

from pidrei.modes.interactive.theme import init_theme, theme


# --- Color utilities ---

_HEX_RE = re.compile(r"^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$", re.IGNORECASE)


def hex_to_rgb(hex_color: str) -> tuple:
    result = _HEX_RE.match(hex_color)
    if not result:
        return (0, 0, 0)
    return (int(result.group(1), 16), int(result.group(2), 16), int(result.group(3), 16))


def rgb_to_hex(r: float, g: float, b: float) -> str:
    return "#" + "".join(format(round(max(0, min(255, x))), "02x") for x in (r, g, b))


def rgb_to_hsl(r: float, g: float, b: float) -> tuple:
    r /= 255
    g /= 255
    b /= 255
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    h = 0.0
    s = 0.0
    lum = (max_c + min_c) / 2
    if max_c != min_c:
        d = max_c - min_c
        s = d / (2 - max_c - min_c) if lum > 0.5 else d / (max_c + min_c)
        if max_c == r:
            h = ((g - b) / d + (6 if g < b else 0)) / 6
        elif max_c == g:
            h = ((b - r) / d + 2) / 6
        else:
            h = ((r - g) / d + 4) / 6
    return (h, s, lum)


def hsl_to_rgb(h: float, s: float, lum: float) -> tuple:
    if s == 0:
        r = g = b = lum
    else:

        def hue2rgb(p: float, q: float, t: float) -> float:
            if t < 0:
                t += 1
            if t > 1:
                t -= 1
            if t < 1 / 6:
                return p + (q - p) * 6 * t
            if t < 1 / 2:
                return q
            if t < 2 / 3:
                return p + (q - p) * (2 / 3 - t) * 6
            return p

        q = lum * (1 + s) if lum < 0.5 else lum + s - lum * s
        p = 2 * lum - q
        r = hue2rgb(p, q, h + 1 / 3)
        g = hue2rgb(p, q, h)
        b = hue2rgb(p, q, h - 1 / 3)
    return (round(r * 255), round(g * 255), round(b * 255))


def get_luminance(r: float, g: float, b: float) -> float:
    def lin(c: float) -> float:
        c = c / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def get_contrast(rgb: tuple, bg_lum: float) -> float:
    fg_lum = get_luminance(*rgb)
    lighter = max(fg_lum, bg_lum)
    darker = min(fg_lum, bg_lum)
    return (lighter + 0.05) / (darker + 0.05)


def adjust_color_to_contrast(hex_color: str, target_contrast: float, against_white: bool) -> str:
    rgb = hex_to_rgb(hex_color)
    h, s, _ = rgb_to_hsl(*rgb)
    bg_lum = 1.0 if against_white else 0.0

    lo = 0.0 if against_white else 0.5
    hi = 0.5 if against_white else 1.0

    for _ in range(50):
        mid = (lo + hi) / 2
        test_rgb = hsl_to_rgb(h, s, mid)
        contrast = get_contrast(test_rgb, bg_lum)

        if against_white:
            if contrast < target_contrast:
                hi = mid
            else:
                lo = mid
        elif contrast < target_contrast:
            lo = mid
        else:
            hi = mid

    final_l = lo if against_white else hi
    return rgb_to_hex(*hsl_to_rgb(h, s, final_l))


def fg_ansi(hex_color: str) -> str:
    rgb = hex_to_rgb(hex_color)
    return f"\x1b[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"


RESET = "\x1b[0m"

# --- Commands ---


def cmd_contrast(target_contrast: float) -> None:
    base_colors = {
        "teal": "#5f8787",
        "blue": "#5f87af",
        "green": "#87af87",
        "yellow": "#d7af5f",
        "red": "#af5f5f",
    }

    print(f"\n=== Colors adjusted to {target_contrast}:1 contrast ===\n")

    print("For LIGHT theme (vs white):")
    for name, hex_color in base_colors.items():
        adjusted = adjust_color_to_contrast(hex_color, target_contrast, True)
        contrast = get_contrast(hex_to_rgb(adjusted), 1.0)
        print(f"  {name.ljust(8)} {fg_ansi(adjusted)}Sample{RESET}  {adjusted}  ({contrast:.2f}:1)")

    print("\nFor DARK theme (vs black):")
    for name, hex_color in base_colors.items():
        adjusted = adjust_color_to_contrast(hex_color, target_contrast, False)
        contrast = get_contrast(hex_to_rgb(adjusted), 0.0)
        print(f"  {name.ljust(8)} {fg_ansi(adjusted)}Sample{RESET}  {adjusted}  ({contrast:.2f}:1)")


def cmd_test(file_path: str) -> None:
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    color_vars = data.get("vars") or data

    print(f"\n=== Testing {file_path} ===\n")

    for name, hex_color in color_vars.items():
        if not isinstance(hex_color, str) or not hex_color.startswith("#"):
            continue
        rgb = hex_to_rgb(hex_color)
        vs_white = get_contrast(rgb, 1.0)
        vs_black = get_contrast(rgb, 0.0)
        pass_w = "AA" if vs_white >= 4.5 else "AA-lg" if vs_white >= 3.0 else "FAIL"
        pass_b = "AA" if vs_black >= 4.5 else "AA-lg" if vs_black >= 3.0 else "FAIL"
        print(
            f"{name.ljust(14)} {fg_ansi(hex_color)}Sample text{RESET}  {hex_color}  "
            f"white: {vs_white:.2f}:1 {pass_w.ljust(5)}  black: {vs_black:.2f}:1 {pass_b}"
        )


def cmd_theme(theme_name: str) -> None:
    os.environ["COLORTERM"] = "truecolor"
    init_theme(theme_name)

    def parse_ansi_rgb(ansi: str) -> tuple | None:
        match = re.search(r"38;2;(\d+);(\d+);(\d+)", ansi)
        if not match:
            return None
        return (int(match.group(1)), int(match.group(2)), int(match.group(3)))

    def contrast_label(color_name: str, bg_lum: float) -> str:
        rgb = parse_ansi_rgb(theme.get_fg_ansi(color_name))
        if not rgb:
            return "(default)"
        ratio = get_contrast(rgb, bg_lum)
        passed = "AA" if ratio >= 4.5 else "AA-lg" if ratio >= 3.0 else "FAIL"
        return f"{ratio:.2f}:1 {passed}"

    def log_color(name: str) -> None:
        sample = theme.fg(name, "Sample text")
        cw = contrast_label(name, 1.0)
        cb = contrast_label(name, 0.0)
        print(f"{name.ljust(20)} {sample}  white: {cw.ljust(12)} black: {cb}")

    print(f"\n=== {theme_name} theme (WCAG AA = 4.5:1) ===")

    print("\n--- Core UI ---")
    for name in ["accent", "border", "borderAccent", "borderMuted", "success", "error", "warning", "muted", "dim"]:
        log_color(name)

    print("\n--- Markdown ---")
    for name in ["mdHeading", "mdLink", "mdCode", "mdCodeBlock", "mdCodeBlockBorder", "mdQuote", "mdListBullet"]:
        log_color(name)

    print("\n--- Diff ---")
    for name in ["toolDiffAdded", "toolDiffRemoved", "toolDiffContext"]:
        log_color(name)

    print("\n--- Thinking ---")
    for name in ["thinkingOff", "thinkingMinimal", "thinkingLow", "thinkingMedium", "thinkingHigh"]:
        log_color(name)

    print("\n--- Backgrounds ---")
    print("userMessageBg:", theme.bg("userMessageBg", " Sample "))
    print("toolPendingBg:", theme.bg("toolPendingBg", " Sample "))
    print("toolSuccessBg:", theme.bg("toolSuccessBg", " Sample "))
    print("toolErrorBg:", theme.bg("toolErrorBg", " Sample "))
    print()


# --- Main ---


def main() -> None:
    args = sys.argv[1:]
    cmd = args[0] if args else None
    arg = args[1] if len(args) > 1 else None

    if cmd == "contrast":
        try:
            target = float(arg) if arg else 0.0
        except ValueError:
            target = 0.0
        cmd_contrast(target or 4.5)
    elif cmd == "test":
        cmd_test(arg or "")
    elif cmd in ("light", "dark"):
        cmd_theme(cmd)
    else:
        print("Usage:")
        print("  python theme_colors_tool.py light|dark     Test built-in theme")
        print("  python theme_colors_tool.py contrast 4.5   Compute colors at ratio")
        print("  python theme_colors_tool.py test file.json Test any JSON file")


if __name__ == "__main__":
    main()
