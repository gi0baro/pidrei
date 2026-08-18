"""Snake game.

Play snake with the /snake command. Shows a full custom component with a game
loop (`Interval`, pidrei's `setInterval` equivalent), cached rendering keyed on
a version counter, and pause/resume persisted across sessions through custom
session entries.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/snake.py
"""

import random

from pidrei_tui import matches_key, visible_width
from pidrei_tui._timers import Interval


GAME_WIDTH = 40
GAME_HEIGHT = 15
TICK_MS = 100

SNAKE_SAVE_TYPE = "snake-save"

# Direction -> (dx, dy). Points are [x, y] lists so the state round-trips
# through the JSON session file unchanged.
_DELTAS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
_OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


def create_initial_state() -> dict:
    start_x = GAME_WIDTH // 2
    start_y = GAME_HEIGHT // 2
    return {
        "snake": [[start_x, start_y], [start_x - 1, start_y], [start_x - 2, start_y]],
        "food": spawn_food([[start_x, start_y]]),
        "direction": "right",
        "nextDirection": "right",
        "score": 0,
        "gameOver": False,
        "highScore": 0,
    }


def spawn_food(snake: list) -> list:
    while True:
        food = [random.randrange(GAME_WIDTH), random.randrange(GAME_HEIGHT)]  # noqa: S311
        if food not in snake:
            return food


class SnakeComponent:
    def __init__(self, tui, on_close, on_save, saved_state: dict | None = None) -> None:
        self._tui = tui
        self._on_close = on_close
        self._on_save = on_save  # async callback
        self._interval: Interval | None = None
        self._cached_lines: list[str] = []
        self._cached_width = 0
        self._version = 0
        self._cached_version = -1

        if saved_state and not saved_state["gameOver"]:
            # Resume from saved state, start paused
            self._state = saved_state
            self._paused = True
        else:
            # New game or saved game was over
            self._state = create_initial_state()
            if saved_state:
                self._state["highScore"] = saved_state["highScore"]
            self._paused = False
            self._start_game()

    def _start_game(self) -> None:
        async def on_tick() -> None:
            if not self._state["gameOver"]:
                self._tick()
                self._version += 1
                self._tui.request_render()

        self._interval = Interval(TICK_MS, on_tick)

    def _tick(self) -> None:
        state = self._state

        # Apply queued direction change
        state["direction"] = state["nextDirection"]

        # Calculate new head position
        head = state["snake"][0]
        dx, dy = _DELTAS[state["direction"]]
        new_head = [head[0] + dx, head[1] + dy]

        # Check wall collision
        if not (0 <= new_head[0] < GAME_WIDTH and 0 <= new_head[1] < GAME_HEIGHT):
            state["gameOver"] = True
            return

        # Check self collision
        if new_head in state["snake"]:
            state["gameOver"] = True
            return

        # Move snake
        state["snake"].insert(0, new_head)

        # Check food collision
        if new_head == state["food"]:
            state["score"] += 10
            state["highScore"] = max(state["highScore"], state["score"])
            state["food"] = spawn_food(state["snake"])
        else:
            state["snake"].pop()

    async def handle_input(self, data: str) -> None:
        state = self._state

        # If paused (resuming), wait for any key
        if self._paused:
            if matches_key(data, "escape") or data in ("q", "Q"):
                # Quit without clearing save
                self.dispose()
                self._on_close()
                return
            # Any other key resumes
            self._paused = False
            self._start_game()
            return

        # ESC to pause and save
        if matches_key(data, "escape"):
            self.dispose()
            await self._on_save(state)
            self._on_close()
            return

        # Q to quit without saving (clears saved state)
        if data in ("q", "Q"):
            self.dispose()
            await self._on_save(None)  # Clear saved state
            self._on_close()
            return

        # Arrow keys or WASD
        wanted = None
        if matches_key(data, "up") or data in ("w", "W"):
            wanted = "up"
        elif matches_key(data, "down") or data in ("s", "S"):
            wanted = "down"
        elif matches_key(data, "right") or data in ("d", "D"):
            wanted = "right"
        elif matches_key(data, "left") or data in ("a", "A"):
            wanted = "left"
        if wanted is not None and state["direction"] != _OPPOSITE[wanted]:
            state["nextDirection"] = wanted

        # Restart on game over
        if state["gameOver"] and data in ("r", "R", " "):
            high_score = state["highScore"]
            self._state = create_initial_state()
            self._state["highScore"] = high_score
            await self._on_save(None)  # Clear saved state on restart
            self._version += 1
            self._tui.request_render()

    def invalidate(self) -> None:
        self._cached_width = 0

    def render(self, width: int) -> list[str]:
        if width == self._cached_width and self._cached_version == self._version:
            return self._cached_lines

        state = self._state
        lines: list[str] = []

        # Each game cell is 2 chars wide to appear square (terminal cells are
        # ~2:1 aspect)
        cell_width = 2
        effective_width = min(GAME_WIDTH, (width - 4) // cell_width)
        effective_height = GAME_HEIGHT

        # Colors - raw ANSI on purpose: this component owns its whole look
        dim = lambda s: f"\x1b[2m{s}\x1b[22m"
        green = lambda s: f"\x1b[32m{s}\x1b[0m"
        red = lambda s: f"\x1b[31m{s}\x1b[0m"
        yellow = lambda s: f"\x1b[33m{s}\x1b[0m"
        bold = lambda s: f"\x1b[1m{s}\x1b[22m"

        box_width = effective_width * cell_width

        # Helper to pad content inside box
        def box_line(content: str) -> str:
            padding = max(0, box_width - visible_width(content))
            return dim(" │") + content + " " * padding + dim("│")

        def pad_line(line: str) -> str:
            return line + " " * max(0, width - visible_width(line))

        # Top border
        lines.append(pad_line(dim(f" ╭{'─' * box_width}╮")))

        # Header with score
        score_text = f"Score: {bold(yellow(str(state['score'])))}"
        high_text = f"High: {bold(yellow(str(state['highScore'])))}"
        title = f"{bold(green('SNAKE'))} │ {score_text} │ {high_text}"
        lines.append(pad_line(box_line(title)))

        # Separator
        lines.append(pad_line(dim(f" ├{'─' * box_width}┤")))

        # Game grid
        head = state["snake"][0]
        body = state["snake"][1:]
        for y in range(effective_height):
            row = ""
            for x in range(effective_width):
                cell = [x, y]
                if cell == head:
                    row += green("██")  # Snake head (2 chars)
                elif cell in body:
                    row += green("▓▓")  # Snake body (2 chars)
                elif cell == state["food"]:
                    row += red("◆ ")  # Food (2 chars)
                else:
                    row += "  "  # Empty cell (2 spaces)
            lines.append(pad_line(dim(" │") + row + dim("│")))

        # Separator
        lines.append(pad_line(dim(f" ├{'─' * box_width}┤")))

        # Footer
        if self._paused:
            footer = f"{yellow(bold('PAUSED'))} Press any key to continue, {bold('Q')} to quit"
        elif state["gameOver"]:
            footer = f"{red(bold('GAME OVER!'))} Press {bold('R')} to restart, {bold('Q')} to quit"
        else:
            footer = f"↑↓←→ or WASD to move, {bold('ESC')} pause, {bold('Q')} quit"
        lines.append(pad_line(box_line(footer)))

        # Bottom border
        lines.append(pad_line(dim(f" ╰{'─' * box_width}╯")))

        self._cached_lines = lines
        self._cached_width = width
        self._cached_version = self._version

        return lines

    def dispose(self) -> None:
        if self._interval is not None:
            self._interval.cancel()
            self._interval = None


def extension(pi):
    async def handle(_args: str, ctx) -> None:
        if ctx.mode != "tui":
            ctx.ui.notify("Snake requires interactive mode", "error")
            return

        # Load saved state from session
        saved_state = None
        for entry in reversed(ctx.session_manager.get_entries()):
            if entry.get("type") == "custom" and entry.get("customType") == SNAKE_SAVE_TYPE:
                saved_state = entry.get("data")
                break

        async def on_save(state: dict | None) -> None:
            # Save or clear state
            await pi.append_entry(SNAKE_SAVE_TYPE, state)

        await ctx.ui.custom(
            lambda tui, _theme, _keybindings, done: SnakeComponent(tui, lambda: done(None), on_save, saved_state)
        )

    pi.register_command("snake", handler=handle, description="Play Snake!")
