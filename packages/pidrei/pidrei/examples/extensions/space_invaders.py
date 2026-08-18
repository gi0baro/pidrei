"""Space Invaders game.

Play with the /invaders command. On top of what `snake.py` shows, this opts
into Kitty keyboard protocol key-release events (`wants_key_release = True` +
`is_key_release`) for smooth held-key movement.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/space_invaders.py
"""

import random

from pidrei_tui import Key, is_key_release, matches_key, visible_width
from pidrei_tui._timers import Interval


GAME_WIDTH = 60
GAME_HEIGHT = 24
TICK_MS = 50
PLAYER_Y = GAME_HEIGHT - 2
ALIEN_ROWS = 5
ALIEN_COLS = 11
ALIEN_START_Y = 2

INVADERS_SAVE_TYPE = "space-invaders-save"


# State is plain dicts and lists so it round-trips through the JSON session
# file unchanged. Bullets: {"x", "y", "direction"} with direction -1 = up
# (player), 1 = down (alien). Aliens: {"x", "y", "type", "alive"}.


def create_shields() -> list[dict]:
    return [
        {
            "x": x,
            # 3x4 grid of destructible segments
            "segments": [
                [True, True, True, True],
                [True, True, True, True],
                [True, False, False, True],
            ],
        }
        for x in (8, 22, 36, 50)
    ]


def create_aliens() -> list[dict]:
    aliens: list[dict] = []
    for row in range(ALIEN_ROWS):
        alien_type = 2 if row == 0 else 1 if row < 3 else 0
        for col in range(ALIEN_COLS):
            aliens.append({"x": 4 + col * 5, "y": ALIEN_START_Y + row * 2, "type": alien_type, "alive": True})
    return aliens


def create_initial_state(high_score: int = 0, level: int = 1) -> dict:
    return {
        "player": {"x": GAME_WIDTH // 2, "lives": 3},
        "aliens": create_aliens(),
        "alienDirection": 1,
        "alienMoveCounter": 0,
        "alienMoveDelay": max(5, 20 - level * 2),
        "alienDropping": False,
        "bullets": [],
        "shields": create_shields(),
        "score": 0,
        "highScore": high_score,
        "level": level,
        "gameOver": False,
        "victory": False,
        "alienShootCounter": 0,
    }


class SpaceInvadersComponent:
    # Opt-in to key release events for smooth movement
    wants_key_release = True

    def __init__(self, tui, on_close, on_save, saved_state: dict | None = None) -> None:
        self._tui = tui
        self._on_close = on_close
        self._on_save = on_save  # async callback
        self._keys = {"left": False, "right": False, "fire": False}
        self._interval: Interval | None = None
        self._cached_lines: list[str] = []
        self._cached_width = 0
        self._version = 0
        self._cached_version = -1
        self._fire_cooldown = 0
        self._player_move_counter = 0

        if saved_state and not saved_state["gameOver"] and not saved_state["victory"]:
            self._state = saved_state
            self._paused = True
        else:
            self._state = create_initial_state(saved_state["highScore"] if saved_state else 0)
            self._paused = False
            self._start_game()

    def _start_game(self) -> None:
        async def on_tick() -> None:
            if not self._state["gameOver"] and not self._state["victory"]:
                self._tick()
                self._version += 1
                self._tui.request_render()

        self._interval = Interval(TICK_MS, on_tick)

    def _tick(self) -> None:
        state = self._state

        # Player movement (smooth, every other tick)
        self._player_move_counter += 1
        if self._player_move_counter >= 2:
            self._player_move_counter = 0
            if self._keys["left"] and state["player"]["x"] > 2:
                state["player"]["x"] -= 1
            if self._keys["right"] and state["player"]["x"] < GAME_WIDTH - 3:
                state["player"]["x"] += 1

        # Fire cooldown
        if self._fire_cooldown > 0:
            self._fire_cooldown -= 1

        # Player shooting
        if self._keys["fire"] and self._fire_cooldown == 0:
            player_bullets = [b for b in state["bullets"] if b["direction"] == -1]
            if len(player_bullets) < 2:
                state["bullets"].append({"x": state["player"]["x"], "y": PLAYER_Y - 1, "direction": -1})
                self._fire_cooldown = 8

        # Move bullets
        moved = []
        for bullet in state["bullets"]:
            bullet["y"] += bullet["direction"]
            if 0 <= bullet["y"] < GAME_HEIGHT:
                moved.append(bullet)
        state["bullets"] = moved

        # Alien movement
        state["alienMoveCounter"] += 1
        if state["alienMoveCounter"] >= state["alienMoveDelay"]:
            state["alienMoveCounter"] = 0
            self._move_aliens()

        # Alien shooting
        state["alienShootCounter"] += 1
        if state["alienShootCounter"] >= 30:
            state["alienShootCounter"] = 0
            self._alien_shoot()

        # Collision detection
        self._check_collisions()

        # Check victory
        if all(not a["alive"] for a in state["aliens"]):
            state["victory"] = True

    def _move_aliens(self) -> None:
        state = self._state
        alive_aliens = [a for a in state["aliens"] if a["alive"]]
        if not alive_aliens:
            return

        if state["alienDropping"]:
            # Drop down
            for alien in alive_aliens:
                alien["y"] += 1
                if alien["y"] >= PLAYER_Y - 1:
                    state["gameOver"] = True
                    return
            state["alienDropping"] = False
        else:
            # Check if we need to change direction
            min_x = min(a["x"] for a in alive_aliens)
            max_x = max(a["x"] for a in alive_aliens)

            if (state["alienDirection"] == 1 and max_x >= GAME_WIDTH - 3) or (
                state["alienDirection"] == -1 and min_x <= 2
            ):
                state["alienDirection"] *= -1
                state["alienDropping"] = True
            else:
                # Move horizontally
                for alien in alive_aliens:
                    alien["x"] += state["alienDirection"]

        # Speed up as fewer aliens remain
        alive_count = len(alive_aliens)
        if alive_count <= 5:
            state["alienMoveDelay"] = 1
        elif alive_count <= 10:
            state["alienMoveDelay"] = 2
        elif alive_count <= 20:
            state["alienMoveDelay"] = 3

    def _alien_shoot(self) -> None:
        state = self._state
        alive_aliens = [a for a in state["aliens"] if a["alive"]]
        if not alive_aliens:
            return

        # Find bottom-most alien in each column
        columns: dict[int, dict] = {}
        for alien in alive_aliens:
            existing = columns.get(alien["x"])
            if existing is None or alien["y"] > existing["y"]:
                columns[alien["x"]] = alien

        # Random column shoots
        shooters = list(columns.values())
        alien_bullets = [b for b in state["bullets"] if b["direction"] == 1]
        if shooters and len(alien_bullets) < 3:
            shooter = random.choice(shooters)  # noqa: S311
            state["bullets"].append({"x": shooter["x"], "y": shooter["y"] + 1, "direction": 1})

    def _check_collisions(self) -> None:
        state = self._state
        bullets_to_remove: list[dict] = []

        for bullet in state["bullets"]:
            # Player bullets hitting aliens
            if bullet["direction"] == -1:
                for alien in state["aliens"]:
                    if alien["alive"] and abs(bullet["x"] - alien["x"]) <= 1 and bullet["y"] == alien["y"]:
                        alien["alive"] = False
                        bullets_to_remove.append(bullet)
                        state["score"] += (10, 20, 30)[alien["type"]]
                        state["highScore"] = max(state["highScore"], state["score"])
                        break

            # Alien bullets hitting player
            if bullet["direction"] == 1 and abs(bullet["x"] - state["player"]["x"]) <= 1 and bullet["y"] == PLAYER_Y:
                bullets_to_remove.append(bullet)
                state["player"]["lives"] -= 1
                if state["player"]["lives"] <= 0:
                    state["gameOver"] = True

            # Bullets hitting shields
            for shield in state["shields"]:
                rel_x = bullet["x"] - shield["x"]
                rel_y = bullet["y"] - (PLAYER_Y - 5)
                if 0 <= rel_x < 4 and 0 <= rel_y < 3 and shield["segments"][rel_y][rel_x]:
                    shield["segments"][rel_y][rel_x] = False
                    bullets_to_remove.append(bullet)

        state["bullets"] = [b for b in state["bullets"] if not any(b is r for r in bullets_to_remove)]

    async def handle_input(self, data: str) -> None:
        state = self._state
        released = is_key_release(data)

        # Pause handling
        if self._paused and not released:
            if matches_key(data, Key.escape) or data in ("q", "Q"):
                self.dispose()
                self._on_close()
                return
            self._paused = False
            self._start_game()
            return

        # ESC to pause and save
        if not released and matches_key(data, Key.escape):
            self.dispose()
            await self._on_save(state)
            self._on_close()
            return

        # Q to quit without saving
        if not released and data in ("q", "Q"):
            self.dispose()
            await self._on_save(None)
            self._on_close()
            return

        # Movement keys (track press/release state; matches_key also covers
        # the Kitty-protocol sequences the raw comparisons miss)
        if matches_key(data, Key.left) or data in ("a", "A") or matches_key(data, "a"):
            self._keys["left"] = not released
        if matches_key(data, Key.right) or data in ("d", "D") or matches_key(data, "d"):
            self._keys["right"] = not released

        # Fire key
        if matches_key(data, Key.space) or data in (" ", "f", "F") or matches_key(data, "f"):
            self._keys["fire"] = not released

        # Restart on game over or victory
        if not released and (state["gameOver"] or state["victory"]) and data in ("r", "R", " "):
            high_score = state["highScore"]
            next_level = state["level"] + 1 if state["victory"] else 1
            self._state = create_initial_state(high_score, next_level)
            self._keys = {"left": False, "right": False, "fire": False}
            await self._on_save(None)
            self._version += 1
            self._tui.request_render()

    def invalidate(self) -> None:
        self._cached_width = 0

    def render(self, width: int) -> list[str]:
        if width == self._cached_width and self._cached_version == self._version:
            return self._cached_lines

        state = self._state
        lines: list[str] = []

        # Colors - raw ANSI on purpose: this component owns its whole look
        dim = lambda s: f"\x1b[2m{s}\x1b[22m"
        green = lambda s: f"\x1b[32m{s}\x1b[0m"
        red = lambda s: f"\x1b[31m{s}\x1b[0m"
        yellow = lambda s: f"\x1b[33m{s}\x1b[0m"
        cyan = lambda s: f"\x1b[36m{s}\x1b[0m"
        magenta = lambda s: f"\x1b[35m{s}\x1b[0m"
        white = lambda s: f"\x1b[97m{s}\x1b[0m"
        bold = lambda s: f"\x1b[1m{s}\x1b[22m"

        box_width = GAME_WIDTH

        def box_line(content: str) -> str:
            padding = max(0, box_width - visible_width(content))
            return dim(" │") + content + " " * padding + dim("│")

        def pad_line(line: str) -> str:
            return line + " " * max(0, width - visible_width(line))

        # Top border
        lines.append(pad_line(dim(f" ╭{'─' * box_width}╮")))

        # Header
        title = bold(green("SPACE INVADERS"))
        score_text = f"Score: {bold(yellow(str(state['score'])))}"
        high_text = f"Hi: {bold(yellow(str(state['highScore'])))}"
        level_text = f"Lv: {bold(cyan(str(state['level'])))}"
        lives_text = red("♥" * state["player"]["lives"])
        header = f"{title} │ {score_text} │ {high_text} │ {level_text} │ {lives_text}"
        lines.append(pad_line(box_line(header)))

        # Separator
        lines.append(pad_line(dim(f" ├{'─' * box_width}┤")))

        # Game grid
        alien_sprites = [("╲", "▼", "╱"), ("╱", "◆", "╲"), ("◄", "☆", "►")]
        alien_colors = [green, cyan, magenta]
        for y in range(GAME_HEIGHT):
            row = ""
            for x in range(GAME_WIDTH):
                char = None

                # Check aliens (each is 3 cells wide)
                for alien in state["aliens"]:
                    if alien["alive"] and alien["y"] == y and abs(alien["x"] - x) <= 1:
                        sprite = alien_sprites[alien["type"]][x - alien["x"] + 1]
                        char = alien_colors[alien["type"]](sprite)
                        break

                # Check shields
                if char is None:
                    for shield in state["shields"]:
                        rel_x = x - shield["x"]
                        rel_y = y - (PLAYER_Y - 5)
                        if 0 <= rel_x < 4 and 0 <= rel_y < 3:
                            if shield["segments"][rel_y][rel_x]:
                                char = dim("█")
                            break

                # Check player
                if char is None and y == PLAYER_Y and abs(x - state["player"]["x"]) <= 1:
                    char = white("▲" if x == state["player"]["x"] else "═")

                # Check bullets
                if char is None:
                    for bullet in state["bullets"]:
                        if bullet["x"] == x and bullet["y"] == y:
                            char = yellow("│") if bullet["direction"] == -1 else red("│")
                            break

                row += char if char is not None else " "
            lines.append(pad_line(dim(" │") + row + dim("│")))

        # Separator
        lines.append(pad_line(dim(f" ├{'─' * box_width}┤")))

        # Footer
        if self._paused:
            footer = f"{yellow(bold('PAUSED'))} Press any key to continue, {bold('Q')} to quit"
        elif state["gameOver"]:
            footer = f"{red(bold('GAME OVER!'))} Press {bold('R')} to restart, {bold('Q')} to quit"
        elif state["victory"]:
            footer = f"{green(bold('VICTORY!'))} Press {bold('R')} for level {state['level'] + 1}, {bold('Q')} to quit"
        else:
            footer = f"←→ or AD to move, {bold('SPACE')}/F to fire, {bold('ESC')} pause, {bold('Q')} quit"
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
            ctx.ui.notify("Space Invaders requires interactive mode", "error")
            return

        # Load saved state from session
        saved_state = None
        for entry in reversed(ctx.session_manager.get_entries()):
            if entry.get("type") == "custom" and entry.get("customType") == INVADERS_SAVE_TYPE:
                saved_state = entry.get("data")
                break

        async def on_save(state: dict | None) -> None:
            await pi.append_entry(INVADERS_SAVE_TYPE, state)

        await ctx.ui.custom(
            lambda tui, _theme, _keybindings, done: SpaceInvadersComponent(
                tui, lambda: done(None), on_save, saved_state
            )
        )

    pi.register_command("invaders", handler=handle, description="Play Space Invaders!")
