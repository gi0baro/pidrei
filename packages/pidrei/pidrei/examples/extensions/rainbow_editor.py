"""Rainbow Editor

Highlights "ultrathink" in the editor with an animated shine effect.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/rainbow_editor.py
"""

import re

import tonio.colored as tonio

from pidrei.modes.interactive.components import CustomEditor


# Base colors (coral → yellow → green → teal → blue → purple → pink)
COLORS: list[tuple[int, int, int]] = [
    (233, 137, 115),  # coral
    (228, 186, 103),  # yellow
    (141, 192, 122),  # green
    (102, 194, 179),  # teal
    (121, 157, 207),  # blue
    (157, 134, 195),  # purple
    (206, 130, 172),  # pink
]
RESET = "\x1b[0m"
ULTRATHINK = re.compile(r"ultrathink", re.IGNORECASE)
ANIMATION_INTERVAL_S = 0.06


def brighten(rgb: tuple[int, int, int], factor: float) -> str:
    r, g, b = (round(c + (255 - c) * factor) for c in rgb)
    return f"\x1b[38;2;{r};{g};{b}m"


def colorize(text: str, shine_pos: int) -> str:
    parts = []
    for i, char in enumerate(text):
        base_color = COLORS[i % len(COLORS)]
        # 3-letter shine: center bright, adjacent dimmer
        factor = 0.0
        if shine_pos >= 0:
            dist = abs(i - shine_pos)
            if dist == 0:
                factor = 0.7
            elif dist == 1:
                factor = 0.35
        parts.append(f"{brighten(base_color, factor)}{char}")
    return "".join(parts) + RESET


class RainbowEditor(CustomEditor):
    def __init__(self, tui, theme, keybindings) -> None:
        super().__init__(tui, theme, keybindings)
        self._animation_stop: tonio.Event | None = None
        self._frame = 0

    def _has_ultrathink(self) -> bool:
        return ULTRATHINK.search(self.get_text()) is not None

    def _start_animation(self) -> None:
        if self._animation_stop is not None:
            return

        # pi drives the animation with setInterval; here it is a background
        # task that ends cooperatively through the Event.
        stop = tonio.Event()
        self._animation_stop = stop

        async def animate() -> None:
            while True:
                await stop.wait(ANIMATION_INTERVAL_S)
                if stop.is_set():
                    return
                self._frame += 1
                self._tui.request_render()

        tonio.spawn.without_tracking(animate())

    def _stop_animation(self) -> None:
        if self._animation_stop is not None:
            self._animation_stop.set()
            self._animation_stop = None

    async def handle_input(self, data: str) -> None:
        await super().handle_input(data)
        if self._has_ultrathink():
            self._start_animation()
        else:
            self._stop_animation()

    def render(self, width: int) -> list[str]:
        # Cycle: 10 shine positions + 10 pause frames
        cycle = self._frame % 20
        shine_pos = cycle if cycle < 10 else -1  # -1 means no shine (pause)
        return [ULTRATHINK.sub(lambda m: colorize(m.group(0), shine_pos), line) for line in super().render(width)]


def extension(pi):
    async def on_session_start(_event, ctx) -> None:
        ctx.ui.set_editor_component(lambda tui, theme, keybindings: RainbowEditor(tui, theme, keybindings))

    pi.on("session_start", on_session_start)
