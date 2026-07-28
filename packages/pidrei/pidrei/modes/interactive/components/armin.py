"""Mirror of pi coding-agent src/modes/interactive/components/armin.ts.

Armin says hi! A fun easter egg with animated XBM art.
"""

import math
import random

from pidrei_tui._timers import Interval

from ..theme import theme


# XBM image: 31x36 pixels, LSB first, 1=background, 0=foreground
WIDTH = 31
HEIGHT = 36
BITS = [
    0xFF,
    0xFF,
    0xFF,
    0x7F,
    0xFF,
    0xF0,
    0xFF,
    0x7F,
    0xFF,
    0xED,
    0xFF,
    0x7F,
    0xFF,
    0xDB,
    0xFF,
    0x7F,
    0xFF,
    0xB7,
    0xFF,
    0x7F,
    0xFF,
    0x77,
    0xFE,
    0x7F,
    0x3F,
    0xF8,
    0xFE,
    0x7F,
    0xDF,
    0xFF,
    0xFE,
    0x7F,
    0xDF,
    0x3F,
    0xFC,
    0x7F,
    0x9F,
    0xC3,
    0xFB,
    0x7F,
    0x6F,
    0xFC,
    0xF4,
    0x7F,
    0xF7,
    0x0F,
    0xF7,
    0x7F,
    0xF7,
    0xFF,
    0xF7,
    0x7F,
    0xF7,
    0xFF,
    0xE3,
    0x7F,
    0xF7,
    0x07,
    0xE8,
    0x7F,
    0xEF,
    0xF8,
    0x67,
    0x70,
    0x0F,
    0xFF,
    0xBB,
    0x6F,
    0xF1,
    0x00,
    0xD0,
    0x5B,
    0xFD,
    0x3F,
    0xEC,
    0x53,
    0xC1,
    0xFF,
    0xEF,
    0x57,
    0x9F,
    0xFD,
    0xEE,
    0x5F,
    0x9F,
    0xFC,
    0xAE,
    0x5F,
    0x1F,
    0x78,
    0xAC,
    0x5F,
    0x3F,
    0x00,
    0x50,
    0x6C,
    0x7F,
    0x00,
    0xDC,
    0x77,
    0xFF,
    0xC0,
    0x3F,
    0x78,
    0xFF,
    0x01,
    0xF8,
    0x7F,
    0xFF,
    0x03,
    0x9C,
    0x78,
    0xFF,
    0x07,
    0x8C,
    0x7C,
    0xFF,
    0x0F,
    0xCE,
    0x78,
    0xFF,
    0xFF,
    0xCF,
    0x7F,
    0xFF,
    0xFF,
    0xCF,
    0x78,
    0xFF,
    0xFF,
    0xDF,
    0x78,
    0xFF,
    0xFF,
    0xDF,
    0x7D,
    0xFF,
    0xFF,
    0x3F,
    0x7E,
    0xFF,
    0xFF,
    0xFF,
    0x7F,
]

BYTES_PER_ROW = math.ceil(WIDTH / 8)
DISPLAY_HEIGHT = math.ceil(HEIGHT / 2)  # Half-block rendering

EFFECTS = ["typewriter", "scanline", "rain", "fade", "crt", "glitch", "dissolve"]


def _get_pixel(x: int, y: int) -> bool:
    """Pixel at (x, y): True = foreground, False = background."""
    if y >= HEIGHT:
        return False
    byte_index = y * BYTES_PER_ROW + x // 8
    bit_index = x % 8
    return ((BITS[byte_index] >> bit_index) & 1) == 0


def _get_char(x: int, row: int) -> str:
    """Character for a cell (2 vertical pixels packed)."""
    upper = _get_pixel(x, row * 2)
    lower = _get_pixel(x, row * 2 + 1)
    if upper and lower:
        return "█"
    if upper:
        return "▀"
    if lower:
        return "▄"
    return " "


def _build_final_grid() -> list:
    return [[_get_char(x, row) for x in range(WIDTH)] for row in range(DISPLAY_HEIGHT)]


class ArminComponent:
    def __init__(self, ui) -> None:
        self._ui = ui
        self._interval: Interval | None = None
        self._effect = random.choice(EFFECTS)  # noqa: S311
        self._final_grid = _build_final_grid()
        self._current_grid = self._create_empty_grid()
        self._effect_state: dict = {}
        self._cached_lines: list = []
        self._cached_width = 0
        self._grid_version = 0
        self._cached_version = -1

        self._init_effect()
        self._start_animation()

    def invalidate(self) -> None:
        self._cached_width = 0

    def render(self, width: int) -> list:
        if width == self._cached_width and self._cached_version == self._grid_version:
            return self._cached_lines

        padding = 1
        available_width = width - padding

        self._cached_lines = []
        for row in self._current_grid:
            # Clip row to available width before applying color
            clipped = "".join(row[: max(0, available_width)])
            pad_right = max(0, width - padding - len(clipped))
            self._cached_lines.append(f" {theme.fg('accent', clipped)}{' ' * pad_right}")

        # Add "ARMIN SAYS HI" at the end
        message = "ARMIN SAYS HI"
        msg_pad_right = max(0, width - padding - len(message))
        self._cached_lines.append(f" {theme.fg('accent', message)}{' ' * msg_pad_right}")

        self._cached_width = width
        self._cached_version = self._grid_version

        return self._cached_lines

    def _create_empty_grid(self) -> list:
        return [[" "] * WIDTH for _ in range(DISPLAY_HEIGHT)]

    def _init_effect(self) -> None:
        if self._effect == "typewriter":
            self._effect_state = {"pos": 0}
        elif self._effect == "scanline":
            self._effect_state = {"row": 0}
        elif self._effect == "rain":
            # Track falling position for each column
            self._effect_state = {
                "drops": [
                    {"y": -int(random.random() * DISPLAY_HEIGHT * 2), "settled": 0}  # noqa: S311
                    for _ in range(WIDTH)
                ]
            }
        elif self._effect == "fade":
            positions = [(row, x) for row in range(DISPLAY_HEIGHT) for x in range(WIDTH)]
            random.shuffle(positions)
            self._effect_state = {"positions": positions, "idx": 0}
        elif self._effect == "crt":
            self._effect_state = {"expansion": 0}
        elif self._effect == "glitch":
            self._effect_state = {"phase": 0, "glitchFrames": 8}
        elif self._effect == "dissolve":
            # Start with random noise
            chars = [" ", "░", "▒", "▓", "█", "▀", "▄"]
            self._current_grid = [
                [random.choice(chars) for _ in range(WIDTH)]  # noqa: S311
                for _ in range(DISPLAY_HEIGHT)
            ]
            # Shuffle positions for gradual resolve
            positions = [(row, x) for row in range(DISPLAY_HEIGHT) for x in range(WIDTH)]
            random.shuffle(positions)
            self._effect_state = {"positions": positions, "idx": 0}

    def _start_animation(self) -> None:
        fps = 60 if self._effect == "glitch" else 30

        async def on_tick() -> None:
            done = self._tick_effect()
            self._grid_version += 1
            self._ui.request_render()
            if done:
                self._stop_animation()

        self._interval = Interval(1000 / fps, on_tick)

    def _stop_animation(self) -> None:
        if self._interval is not None:
            self._interval.cancel()
            self._interval = None

    def _tick_effect(self) -> bool:
        tick = {
            "typewriter": self._tick_typewriter,
            "scanline": self._tick_scanline,
            "rain": self._tick_rain,
            "fade": self._tick_fade,
            "crt": self._tick_crt,
            "glitch": self._tick_glitch,
            "dissolve": self._tick_dissolve,
        }.get(self._effect)
        if tick is None:
            return True
        return tick()

    def _tick_typewriter(self) -> bool:
        state = self._effect_state
        pixels_per_frame = 3

        for _ in range(pixels_per_frame):
            row = state["pos"] // WIDTH
            x = state["pos"] % WIDTH
            if row >= DISPLAY_HEIGHT:
                return True
            self._current_grid[row][x] = self._final_grid[row][x]
            state["pos"] += 1
        return False

    def _tick_scanline(self) -> bool:
        state = self._effect_state
        if state["row"] >= DISPLAY_HEIGHT:
            return True

        # Copy row
        self._current_grid[state["row"]] = list(self._final_grid[state["row"]])
        state["row"] += 1
        return False

    def _tick_rain(self) -> bool:
        state = self._effect_state

        all_settled = True
        self._current_grid = self._create_empty_grid()

        for x in range(WIDTH):
            drop = state["drops"][x]

            # Draw settled pixels
            for row in range(DISPLAY_HEIGHT - 1, DISPLAY_HEIGHT - drop["settled"] - 1, -1):
                if row >= 0:
                    self._current_grid[row][x] = self._final_grid[row][x]

            # Check if this column is done
            if drop["settled"] >= DISPLAY_HEIGHT:
                continue

            all_settled = False

            # Find the target row for this column (lowest non-space pixel)
            target_row = -1
            for row in range(DISPLAY_HEIGHT - 1 - drop["settled"], -1, -1):
                if self._final_grid[row][x] != " ":
                    target_row = row
                    break

            # Move drop down
            drop["y"] += 1

            # Draw falling drop
            if 0 <= drop["y"] < DISPLAY_HEIGHT:
                if target_row >= 0 and drop["y"] >= target_row:
                    # Settle
                    drop["settled"] = DISPLAY_HEIGHT - target_row
                    drop["y"] = -int(random.random() * 5) - 1  # noqa: S311
                else:
                    # Still falling
                    self._current_grid[drop["y"]][x] = "▓"

        return all_settled

    def _tick_fade(self) -> bool:
        state = self._effect_state
        pixels_per_frame = 15

        for _ in range(pixels_per_frame):
            if state["idx"] >= len(state["positions"]):
                return True
            row, x = state["positions"][state["idx"]]
            self._current_grid[row][x] = self._final_grid[row][x]
            state["idx"] += 1
        return False

    def _tick_crt(self) -> bool:
        state = self._effect_state
        mid_row = DISPLAY_HEIGHT // 2

        self._current_grid = self._create_empty_grid()

        # Draw from middle expanding outward
        top = mid_row - state["expansion"]
        bottom = mid_row + state["expansion"]

        for row in range(max(0, top), min(DISPLAY_HEIGHT - 1, bottom) + 1):
            self._current_grid[row] = list(self._final_grid[row])

        state["expansion"] += 1
        return state["expansion"] > DISPLAY_HEIGHT

    def _tick_glitch(self) -> bool:
        state = self._effect_state

        if state["phase"] < state["glitchFrames"]:
            # Glitch phase: show corrupted version
            new_grid = []
            for row in self._final_grid:
                offset = int(random.random() * 7) - 3  # noqa: S311
                glitch_row = list(row)

                # Random horizontal offset
                if random.random() < 0.3:  # noqa: S311
                    shifted = glitch_row[offset:] + glitch_row[:offset]
                    new_grid.append(shifted[:WIDTH])
                    continue

                # Random vertical swap
                if random.random() < 0.2:  # noqa: S311
                    swap_row = int(random.random() * DISPLAY_HEIGHT)  # noqa: S311
                    new_grid.append(list(self._final_grid[swap_row]))
                    continue

                new_grid.append(glitch_row)
            self._current_grid = new_grid
            state["phase"] += 1
            return False

        # Final frame: show clean image
        self._current_grid = [list(row) for row in self._final_grid]
        return True

    def _tick_dissolve(self) -> bool:
        state = self._effect_state
        pixels_per_frame = 20

        for _ in range(pixels_per_frame):
            if state["idx"] >= len(state["positions"]):
                return True
            row, x = state["positions"][state["idx"]]
            self._current_grid[row][x] = self._final_grid[row][x]
            state["idx"] += 1
        return False

    def dispose(self) -> None:
        self._stop_animation()
