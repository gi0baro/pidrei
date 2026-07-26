"""Mirror of pi coding-agent src/core/export-html/index.ts."""

import base64
import json
import os
import re

from ...config import APP_NAME, get_export_template_dir
from ...modes.interactive.theme import get_resolved_theme_colors, get_theme_export_colors
from ...utils.paths import normalize_path, resolve_path
from ..session_manager import SessionManager, _entry_to_wire
from .ansi_to_html import ansi_lines_to_html, ansi_to_html
from .tool_renderer import ToolHtmlRenderer, create_tool_html_renderer


__all__ = [
    "ToolHtmlRenderer",
    "ansi_lines_to_html",
    "ansi_to_html",
    "create_tool_html_renderer",
    "export_from_file",
    "export_session_to_html",
]

_HEX_COLOR_RE = re.compile(r"^#([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})$")
_RGB_COLOR_RE = re.compile(r"^rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$")


def _parse_color(color: str) -> dict | None:
    """Parse hex (#RRGGBB) or rgb(r,g,b) colors to RGB values."""
    hex_match = _HEX_COLOR_RE.match(color)
    if hex_match:
        return {
            "r": int(hex_match.group(1), 16),
            "g": int(hex_match.group(2), 16),
            "b": int(hex_match.group(3), 16),
        }
    rgb_match = _RGB_COLOR_RE.match(color)
    if rgb_match:
        return {"r": int(rgb_match.group(1)), "g": int(rgb_match.group(2)), "b": int(rgb_match.group(3))}
    return None


def _get_luminance(r: int, g: int, b: int) -> float:
    """Relative luminance of a color (0-1, higher = lighter)."""

    def to_linear(c: int) -> float:
        s = c / 255
        return s / 12.92 if s <= 0.03928 else ((s + 0.055) / 1.055) ** 2.4

    return 0.2126 * to_linear(r) + 0.7152 * to_linear(g) + 0.0722 * to_linear(b)


def _adjust_brightness(color: str, factor: float) -> str:
    """Adjust color brightness. Factor > 1 lightens, < 1 darkens."""
    parsed = _parse_color(color)
    if parsed is None:
        return color

    def adjust(c: int) -> int:
        return min(255, max(0, round(c * factor)))

    return f"rgb({adjust(parsed['r'])}, {adjust(parsed['g'])}, {adjust(parsed['b'])})"


def _derive_export_colors(base_color: str) -> dict:
    """Derive export background colors from a base color (userMessageBg)."""
    parsed = _parse_color(base_color)
    if parsed is None:
        return {
            "pageBg": "rgb(24, 24, 30)",
            "cardBg": "rgb(30, 30, 36)",
            "infoBg": "rgb(60, 55, 40)",
        }

    luminance = _get_luminance(parsed["r"], parsed["g"], parsed["b"])
    is_light = luminance > 0.5

    if is_light:
        return {
            "pageBg": _adjust_brightness(base_color, 0.96),
            "cardBg": base_color,
            "infoBg": (f"rgb({min(255, parsed['r'] + 10)}, {min(255, parsed['g'] + 5)}, {max(0, parsed['b'] - 20)})"),
        }
    return {
        "pageBg": _adjust_brightness(base_color, 0.7),
        "cardBg": _adjust_brightness(base_color, 0.85),
        "infoBg": f"rgb({min(255, parsed['r'] + 20)}, {min(255, parsed['g'] + 15)}, {parsed['b']})",
    }


def _generate_theme_vars(theme_name: str | None = None) -> str:
    """Generate CSS custom property declarations from theme colors."""
    colors = get_resolved_theme_colors(theme_name)
    lines = [f"--{key}: {value};" for key, value in colors.items()]

    # Use explicit theme export colors if available, otherwise derive from
    # userMessageBg
    theme_export = get_theme_export_colors(theme_name)
    user_message_bg = colors.get("userMessageBg") or "#343541"
    derived_colors = _derive_export_colors(user_message_bg)

    def pick(key: str) -> str:
        value = theme_export.get(key)
        return value if value is not None else derived_colors[key]

    lines.append(f"--exportPageBg: {pick('pageBg')};")
    lines.append(f"--exportCardBg: {pick('cardBg')};")
    lines.append(f"--exportInfoBg: {pick('infoBg')};")

    return "\n      ".join(lines)


def _read_template_file(*parts: str) -> str:
    with open(os.path.join(get_export_template_dir(), *parts), encoding="utf-8") as f:
        return f.read()


def _generate_html(session_data: dict, theme_name: str | None = None) -> str:
    """Core HTML generation logic shared by both export functions."""
    template = _read_template_file("template.html")
    template_css = _read_template_file("template.css")
    template_js = _read_template_file("template.js")
    marked_js = _read_template_file("vendor", "marked.min.js")
    hljs_js = _read_template_file("vendor", "highlight.min.js")

    theme_vars = _generate_theme_vars(theme_name)
    colors = get_resolved_theme_colors(theme_name)
    theme_export = get_theme_export_colors(theme_name)
    derived_export_colors = _derive_export_colors(colors.get("userMessageBg") or "#343541")

    def pick(key: str) -> str:
        value = theme_export.get(key)
        return value if value is not None else derived_export_colors[key]

    body_bg = pick("pageBg")
    container_bg = pick("cardBg")
    info_bg = pick("infoBg")

    # Base64 encode session data to avoid escaping issues
    session_data_base64 = base64.b64encode(json.dumps(session_data).encode("utf-8")).decode("ascii")

    # Build the CSS with theme variables injected
    css = (
        template_css.replace("{{THEME_VARS}}", theme_vars, 1)
        .replace("{{BODY_BG}}", body_bg, 1)
        .replace("{{CONTAINER_BG}}", container_bg, 1)
        .replace("{{INFO_BG}}", info_bg, 1)
    )

    return (
        template.replace("{{CSS}}", css, 1)
        .replace("{{JS}}", template_js, 1)
        .replace("{{SESSION_DATA}}", session_data_base64, 1)
        .replace("{{MARKED_JS}}", marked_js, 1)
        .replace("{{HIGHLIGHT_JS}}", hljs_js, 1)
    )


# Tools rendered directly by the HTML template (not pre-rendered via the
# TUI→ANSI→HTML pipeline)
_TEMPLATE_RENDERED_TOOLS = {"bash", "read", "write", "edit", "ls"}


def _pre_render_custom_tools(entries: list, tool_renderer) -> dict:
    """Pre-render custom tools to HTML using their TUI renderers."""
    rendered_tools: dict = {}

    for entry in entries:
        if entry.get("type") != "message":
            continue
        msg = entry.get("message")
        role = getattr(msg, "role", None)

        # Find tool calls in assistant messages
        if role == "assistant" and isinstance(getattr(msg, "content", None), list):
            for block in msg.content:
                block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
                if block_type == "toolCall":
                    name = block.get("name") if isinstance(block, dict) else block.name
                    if name in _TEMPLATE_RENDERED_TOOLS:
                        continue
                    block_id = block.get("id") if isinstance(block, dict) else block.id
                    arguments = block.get("arguments") if isinstance(block, dict) else block.arguments
                    call_html = tool_renderer.render_call(block_id, name, arguments)
                    if call_html:
                        rendered_tools[block_id] = {"callHtml": call_html}

        # Find tool results
        if role == "toolResult" and getattr(msg, "tool_call_id", None):
            tool_name = getattr(msg, "tool_name", None) or ""
            # Only render if we have a pre-rendered call OR it's not
            # template-rendered
            existing = rendered_tools.get(msg.tool_call_id)
            if existing or tool_name not in _TEMPLATE_RENDERED_TOOLS:
                rendered = tool_renderer.render_result(
                    msg.tool_call_id,
                    tool_name,
                    msg.content,
                    getattr(msg, "details", None),
                    getattr(msg, "is_error", False) or False,
                )
                if rendered:
                    rendered_tools[msg.tool_call_id] = {
                        **(existing or {}),
                        "resultHtmlCollapsed": rendered.get("collapsed"),
                        "resultHtmlExpanded": rendered.get("expanded"),
                    }

    return rendered_tools


def _entries_to_wire(entries: list) -> list:
    return [_entry_to_wire(entry) for entry in entries]


def _normalize_export_options(options) -> dict:
    if isinstance(options, str):
        return {"outputPath": options}
    return options or {}


async def export_session_to_html(sm: SessionManager, state=None, options=None) -> str:
    """Export session to HTML using SessionManager and AgentState.

    Used by the TUI's /export command. ``options`` is an
    ``{"outputPath"?, "themeName"?, "toolRenderer"?}`` record or a plain
    output path string.
    """
    opts = _normalize_export_options(options)

    session_file = sm.get_session_file()
    if not session_file:
        raise Exception("Cannot export in-memory session to HTML")
    if not os.path.exists(session_file):
        raise Exception("Nothing to export yet - start a conversation first")

    entries = sm.get_entries()

    # Pre-render custom tools if a tool renderer is provided
    rendered_tools = None
    if opts.get("toolRenderer") is not None:
        rendered_tools = _pre_render_custom_tools(entries, opts["toolRenderer"])
        # Only include if we actually rendered something
        if not rendered_tools:
            rendered_tools = None

    tools = None
    if state is not None and state.tools is not None:
        tools = [{"name": t.name, "description": t.description, "parameters": t.parameters} for t in state.tools]

    session_data = {
        "header": sm.get_header(),
        "entries": _entries_to_wire(entries),
        "leafId": sm.get_leaf_id(),
        "systemPrompt": state.system_prompt if state is not None else None,
        "tools": tools,
        "renderedTools": rendered_tools,
    }

    html = _generate_html(session_data, opts.get("themeName"))

    output_path = normalize_path(opts["outputPath"]) if opts.get("outputPath") else None
    if not output_path:
        session_basename = os.path.basename(session_file)
        session_basename = session_basename.removesuffix(".jsonl")
        output_path = f"{APP_NAME}-session-{session_basename}.html"

    # One-shot write of the finished document, like pi's sync writeFileSync
    with open(output_path, "w", encoding="utf-8") as f:  # noqa: ASYNC230
        f.write(html)
    return output_path


async def export_from_file(input_path: str, options=None) -> str:
    """Export a session file to HTML (standalone, without AgentState).

    Used by the CLI for exporting arbitrary session files.
    """
    opts = _normalize_export_options(options)
    resolved_input_path = resolve_path(input_path)

    if not os.path.exists(resolved_input_path):
        raise Exception(f"File not found: {resolved_input_path}")

    sm = SessionManager.open(resolved_input_path)

    session_data = {
        "header": sm.get_header(),
        "entries": _entries_to_wire(sm.get_entries()),
        "leafId": sm.get_leaf_id(),
        "systemPrompt": None,
        "tools": None,
    }

    html = _generate_html(session_data, opts.get("themeName"))

    output_path = normalize_path(opts["outputPath"]) if opts.get("outputPath") else None
    if not output_path:
        input_basename = os.path.basename(resolved_input_path).removesuffix(".jsonl")
        output_path = f"{APP_NAME}-session-{input_basename}.html"

    # One-shot write of the finished document, like pi's sync writeFileSync
    with open(output_path, "w", encoding="utf-8") as f:  # noqa: ASYNC230
        f.write(html)
    return output_path
