"""Overlay test.

Validates overlay compositing with inline text inputs. `/overlay-test` shows a
floating overlay with:
- Inline text inputs within menu items
- Edge case tests (wide chars, styled text, emoji)

Start pidrei with this extension:
    pidrei -e ./examples/extensions/overlay_test.py
"""

from pidrei_tui import CURSOR_MARKER, matches_key, visible_width


class OverlayTestComponent:
    width = 70

    def __init__(self, theme, done) -> None:
        self._theme = theme
        self._done = done

        # Focusable interface - set by the TUI when focus changes
        self.focused = False

        self._selected = 0
        self._items = [
            {"label": "Search", "has_input": True, "text": "", "cursor": 0},
            {"label": "Run", "has_input": True, "text": "", "cursor": 0},
            {"label": "Settings", "has_input": False, "text": "", "cursor": 0},
            {"label": "Cancel", "has_input": False, "text": "", "cursor": 0},
        ]

    async def handle_input(self, data: str) -> None:
        if matches_key(data, "escape"):
            self._done(None)
            return

        current = self._items[self._selected]

        if matches_key(data, "return"):
            query = current["text"] if current["has_input"] else None
            self._done({"action": current["label"], "query": query})
            return

        if matches_key(data, "up"):
            self._selected = max(0, self._selected - 1)
        elif matches_key(data, "down"):
            self._selected = min(len(self._items) - 1, self._selected + 1)
        elif current["has_input"]:
            if matches_key(data, "backspace"):
                if current["cursor"] > 0:
                    cursor = current["cursor"]
                    current["text"] = current["text"][: cursor - 1] + current["text"][cursor:]
                    current["cursor"] -= 1
            elif matches_key(data, "left"):
                current["cursor"] = max(0, current["cursor"] - 1)
            elif matches_key(data, "right"):
                current["cursor"] = min(len(current["text"]), current["cursor"] + 1)
            elif len(data) == 1 and ord(data) >= 32:
                cursor = current["cursor"]
                current["text"] = current["text"][:cursor] + data + current["text"][cursor:]
                current["cursor"] += 1

    def render(self, _width: int) -> list[str]:
        w = self.width
        th = self._theme
        inner_w = w - 2
        lines: list[str] = []

        def pad(s: str, length: int) -> str:
            return s + " " * max(0, length - visible_width(s))

        def row(content: str) -> str:
            return th.fg("border", "│") + pad(content, inner_w) + th.fg("border", "│")

        lines.append(th.fg("border", f"╭{'─' * inner_w}╮"))
        lines.append(row(f" {th.fg('accent', '🧪 Overlay Test')}"))
        lines.append(row(""))

        # Edge cases - full width lines to test compositing at boundaries
        lines.append(row(f" {th.fg('dim', '─── Edge Cases (borders should align) ───')}"))
        lines.append(row(f" Wide: {th.fg('warning', '中文日本語한글テスト漢字繁體简体ひらがなカタカナ가나다라마바')}"))
        styled = " ".join(
            [
                th.fg("error", "RED"),
                th.fg("success", "GREEN"),
                th.fg("warning", "YELLOW"),
                th.fg("accent", "ACCENT"),
                th.fg("dim", "DIM"),
                th.fg("error", "more"),
                th.fg("success", "colors"),
            ]
        )
        lines.append(row(f" Styled: {styled}"))
        lines.append(row(" Emoji: 👨‍👩‍👧‍👦 🇯🇵 🚀 💻 🎉 🔥 😀 🎯 🌟 💡 🎨 🔧 📦 🏆 🌈 🎪 🎭 🎬 🎮 🎲"))
        lines.append(row(""))

        # Menu with inline inputs
        lines.append(row(f" {th.fg('dim', '─── Actions ───')}"))

        for i, item in enumerate(self._items):
            is_selected = i == self._selected
            prefix = " ▶ " if is_selected else "   "

            if item["has_input"]:
                label = th.fg("accent" if is_selected else "text", f"{item['label']}:")

                input_display = item["text"]
                if is_selected:
                    cursor = item["cursor"]
                    before = input_display[:cursor]
                    cursor_char = input_display[cursor] if cursor < len(input_display) else " "
                    after = input_display[cursor + 1 :]
                    # Emit hardware cursor marker for IME support when focused
                    marker = CURSOR_MARKER if self.focused else ""
                    input_display = f"{before}{marker}\x1b[7m{cursor_char}\x1b[27m{after}"
                content = f"{prefix}{label} {input_display}"
            else:
                content = prefix + th.fg("accent" if is_selected else "text", item["label"])

            lines.append(row(content))

        lines.append(row(""))
        lines.append(row(f" {th.fg('dim', '↑↓ navigate • type to input • Enter select • Esc cancel')}"))
        lines.append(th.fg("border", f"╰{'─' * inner_w}╯"))

        return lines

    def invalidate(self) -> None:
        pass

    def dispose(self) -> None:
        pass


def extension(pi):
    async def handle(_args: str, ctx) -> None:
        result = await ctx.ui.custom(
            lambda _tui, theme, _keybindings, done: OverlayTestComponent(theme, done),
            {"overlay": True},
        )

        if result:
            msg = '{}: "{}"'.format(result["action"], result["query"]) if result["query"] else result["action"]
            ctx.ui.notify(msg, "info")

    pi.register_command("overlay-test", handler=handle, description="Test overlay rendering with edge cases")
