"""Tic-tac-toe — demonstrates `execution_mode="sequential"` on tools.

The user plays via /tic-tac-toe (arrow keys + Enter). The agent plays via a
single tool `tic_tac_toe` that takes ONE atomic action per call. To play at
(r, c) from its cursor (r0, c0) the agent must emit the required move_* and a
final `play` as SEPARATE tool_use blocks inside ONE assistant response.

Move actions share the agent cursor and have a 300ms delay. Under the default
parallel tool-execution mode this races: `play` can resolve before the earlier
`move_*` calls finish and O lands on the wrong cell. With
`execution_mode="sequential"` the runner serializes the sibling calls and O
lands on the intended cell.

The user cursor (TUI-only) and the agent cursor (tool-only) are stored in
separate fields. Only the agent cursor is ever exposed to the agent.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/tic_tac_toe.py
"""

import tonio.colored as tonio

from pidrei.core.extensions.types import ToolDefinition
from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent
from pidrei_tui import Text, matches_key, truncate_to_width, visible_width


class TicTacToeError(Exception):
    """Raised from the tool on illegal actions. The agent runtime surfaces
    raised errors as tool errors (isError) without resetting any of our
    state."""


# ---------------------------------------------------------------------------
# Game logic
# ---------------------------------------------------------------------------

# Agent cursor home: where the cursor is reset to after a SUCCESSFUL play.
# Pinned at (0,0) so every non-origin play requires at least one move, which
# guarantees multiple tool calls per turn and makes the parallel-vs-sequential
# behavior observable in the demo. The cursor is NOT reset when the user plays
# nor on a failed `play` (cell taken), so the agent can retry without
# starting over.
AGENT_CURSOR_HOME_ROW = 0
AGENT_CURSOR_HOME_COL = 0

# State keys stay camelCase so details round-trip through the JSON session
# file unchanged (same convention as the snake example).


def create_initial_state() -> dict:
    return {
        "board": [[" ", " ", " "], [" ", " ", " "], [" ", " ", " "]],
        # User cursor (TUI-only, never exposed to the agent).
        "userCursorRow": 1,
        "userCursorCol": 1,
        # Agent cursor (manipulated by the tool, shown during O's turn).
        "agentCursorRow": AGENT_CURSOR_HOME_ROW,
        "agentCursorCol": AGENT_CURSOR_HOME_COL,
        "status": "playing",  # "playing" | "win_X" | "win_O" | "draw"
        "userMark": "X",
        "agentMark": "O",
        "currentTurn": "X",
    }


_LINES = [
    [(0, 0), (0, 1), (0, 2)],
    [(1, 0), (1, 1), (1, 2)],
    [(2, 0), (2, 1), (2, 2)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 1), (1, 1), (2, 1)],
    [(0, 2), (1, 2), (2, 2)],
    [(0, 0), (1, 1), (2, 2)],
    [(0, 2), (1, 1), (2, 0)],
]


def get_win_line(board: list[list[str]]) -> list[tuple[int, int]] | None:
    for line in _LINES:
        vals = [board[r][c] for r, c in line]
        if vals[0] != " " and vals[0] == vals[1] == vals[2]:
            return line
    return None


def check_win(board: list[list[str]]) -> str:
    win_line = get_win_line(board)
    if win_line:
        r, c = win_line[0]
        return "win_X" if board[r][c] == "X" else "win_O"
    if all(cell != " " for row in board for cell in row):
        return "draw"
    return "playing"


def board_to_ascii(board: list[list[str]], agent_cursor_row: int, agent_cursor_col: int) -> str:
    """Plain grid with coordinates for empty cells, marking the agent cursor
    position with angle brackets. The user cursor is NEVER included: it is a
    TUI-only concept and must not leak to the agent."""

    def cell(r: int, c: int) -> str:
        mark = board[r][c]
        on_cursor = r == agent_cursor_row and c == agent_cursor_col
        if mark == " ":
            return f"<[{r},{c}]>" if on_cursor else f" [{r},{c}] "
        return f"   <{mark}>   " if on_cursor else f"    {mark}    "

    rows = ["|".join(cell(r, c) for c in range(3)) for r in range(3)]
    separator = "---------+---------+---------"
    return f"\n{separator}\n".join(rows)


# ---------------------------------------------------------------------------
# Visual board rendering (ANSI).
# - Cells have NO background fill. Only the centered glyph is drawn.
# - Played cells color their glyph AND their surrounding borders in the
#   player's color, so each mark reads as a colored boxed region.
# - Cursor is indicated with colored borders around the cursor cell.
# ---------------------------------------------------------------------------

CELL_WIDTH = 7
CELL_HEIGHT = 3

# Player colors (SGR fg codes). Also used for the borders of played cells.
FG_CODE_X = "34"  # blue
FG_CODE_O = "33"  # yellow
FG_CODE_WIN = "32"  # green (overrides on the winning line)

# Single-character glyphs, picked for maximum visual size without emoji.
# - ╳ (BOX DRAWINGS LIGHT DIAGONAL CROSS) for X
# - ◯ (LARGE CIRCLE) for O
GLYPH_X = "╳"
GLYPH_O = "◯"

RESET = "\x1b[0m"


def _dim(s: str) -> str:
    return f"\x1b[2m{s}\x1b[22m"


def center_pad(content: str, width: int) -> str:
    content_len = visible_width(content)
    if content_len >= width:
        return truncate_to_width(content, width)
    pad = width - content_len
    left = pad // 2
    return " " * left + content + " " * (pad - left)


def cell_fg_code(cell: str, is_win: bool) -> str | None:
    """Fg color for a played cell's glyph and its surrounding borders. None
    for empty cells."""
    if cell == " ":
        return None
    if is_win:
        return FG_CODE_WIN
    return FG_CODE_X if cell == "X" else FG_CODE_O


def build_cell_content(mark: str, line_idx: int, is_win: bool) -> str:
    empty = " " * CELL_WIDTH
    if mark == " ":
        return empty

    if line_idx != CELL_HEIGHT // 2:
        return empty

    glyph = GLYPH_X if mark == "X" else GLYPH_O
    fg = cell_fg_code(mark, is_win)
    pad_len = CELL_WIDTH - visible_width(glyph)
    left_pad = pad_len // 2
    right_pad = pad_len - left_pad
    return f"{' ' * left_pad}\x1b[{fg};1m{glyph}{RESET}{' ' * right_pad}"


def border_fg_code(adjacent: list[tuple[str, bool]]) -> str | None:
    """Fg color for a border char based on its adjacent cells. None when no
    adjacent cell is played or when adjacent plays disagree (border stays dim
    to show the separation)."""
    fgs = [fg for cell, is_win in adjacent if (fg := cell_fg_code(cell, is_win)) is not None]
    if not fgs:
        return None
    return fgs[0] if all(f == fgs[0] for f in fgs) else None


_CORNERS = {
    (0, 0): "┌",
    (0, 3): "┐",
    (3, 0): "└",
    (3, 3): "┘",
}


def _corner_char(grid_r: int, grid_c: int) -> str:
    if (grid_r, grid_c) in _CORNERS:
        return _CORNERS[(grid_r, grid_c)]
    if grid_r == 0:
        return "┬"
    if grid_r == 3:
        return "┴"
    if grid_c == 0:
        return "├"
    if grid_c == 3:
        return "┤"
    return "┼"


def render_board(board: list[list[str]], max_width: int, cursor: dict | None = None) -> list[str]:
    """Render the board. `cursor` is an optional overlay dict with "row",
    "col" and "owner" ("user" | "agent"); omit it to render a static snapshot
    (used in tool results, move messages, and the game-over banner)."""
    show_cursor = cursor is not None
    cr = cursor["row"] if cursor else -1
    cc = cursor["col"] if cursor else -1

    # Green for user cursor, yellow for agent cursor.
    cursor_sgr = "\x1b[33;1m" if cursor and cursor["owner"] == "agent" else "\x1b[32;1m"

    win_line = get_win_line(board)
    win_cells = {(r, c) for r, c in (win_line or [])}

    def cell_at(r: int, c: int) -> tuple[str, bool]:
        return board[r][c], (r, c) in win_cells

    def is_cursor_corner(grid_r: int, grid_c: int) -> bool:
        return show_cursor and grid_r in (cr, cr + 1) and grid_c in (cc, cc + 1)

    def is_cursor_h_segment(grid_r: int, c: int) -> bool:
        return show_cursor and c == cc and grid_r in (cr, cr + 1)

    def is_cursor_v_border(r: int, grid_c: int) -> bool:
        return show_cursor and r == cr and grid_c in (cc, cc + 1)

    def paint_border(ch: str, highlighted: bool, fg_code: str | None) -> str:
        if highlighted:
            return f"{cursor_sgr}{ch}{RESET}"
        if fg_code:
            return f"\x1b[{fg_code};1m{ch}{RESET}"
        return _dim(ch)

    def corner_adjacent(grid_r: int, grid_c: int) -> list[tuple[str, bool]]:
        out = []
        for dr, dc in ((-1, -1), (-1, 0), (0, -1), (0, 0)):
            r, c = grid_r + dr, grid_c + dc
            if 0 <= r < 3 and 0 <= c < 3:
                out.append(cell_at(r, c))
        return out

    lines: list[str] = []

    for grid_r in range(4):
        # Horizontal border row.
        row = ""
        for grid_c in range(4):
            corner_color = border_fg_code(corner_adjacent(grid_r, grid_c))
            row += paint_border(_corner_char(grid_r, grid_c), is_cursor_corner(grid_r, grid_c), corner_color)
            if grid_c < 3:
                adj = []
                if grid_r > 0:
                    adj.append(cell_at(grid_r - 1, grid_c))
                if grid_r < 3:
                    adj.append(cell_at(grid_r, grid_c))
                seg_color = border_fg_code(adj)
                row += paint_border("─" * CELL_WIDTH, is_cursor_h_segment(grid_r, grid_c), seg_color)
        lines.append(center_pad(row, max_width))

        if grid_r == 3:
            break

        for line_idx in range(CELL_HEIGHT):
            content_row = ""
            for grid_c in range(4):
                adj = []
                if grid_c > 0:
                    adj.append(cell_at(grid_r, grid_c - 1))
                if grid_c < 3:
                    adj.append(cell_at(grid_r, grid_c))
                v_color = border_fg_code(adj)
                content_row += paint_border("│", is_cursor_v_border(grid_r, grid_c), v_color)
                if grid_c < 3:
                    content_row += build_cell_content(board[grid_r][grid_c], line_idx, (grid_r, grid_c) in win_cells)
            lines.append(center_pad(content_row, max_width))

    return lines


def render_visual_board(state: dict, max_width: int) -> list[str]:
    """Full TUI board with the right cursor overlayed for the current turn."""
    is_user_turn = state["currentTurn"] == state["userMark"]
    cursor = None
    if state["status"] == "playing":
        cursor = {
            "row": state["userCursorRow"] if is_user_turn else state["agentCursorRow"],
            "col": state["userCursorCol"] if is_user_turn else state["agentCursorCol"],
            "owner": "user" if is_user_turn else "agent",
        }
    return render_board(state["board"], max_width, cursor)


# ---------------------------------------------------------------------------
# TUI component
# ---------------------------------------------------------------------------


class TicTacToeComponent:
    def __init__(self, tui, on_close, on_user_play, state: dict) -> None:
        self._tui = tui
        self._on_close = on_close
        self._on_user_play = on_user_play  # async (row, col) -> None
        self._state = state
        self._cached_lines: list[str] = []
        self._cached_width = 0
        self._version = 0
        self._cached_version = -1

    def update_state(self, state: dict) -> None:
        self._state = state
        self._version += 1
        self._tui.request_render()

    async def handle_input(self, data: str) -> None:
        state = self._state
        if matches_key(data, "escape") or data in ("q", "Q"):
            self._on_close()
            return
        if state["status"] != "playing":
            if data in ("r", "R"):
                self._on_close()
            return
        if state["currentTurn"] != state["userMark"]:
            return

        if matches_key(data, "up") and state["userCursorRow"] > 0:
            state["userCursorRow"] -= 1
        elif matches_key(data, "down") and state["userCursorRow"] < 2:
            state["userCursorRow"] += 1
        elif matches_key(data, "left") and state["userCursorCol"] > 0:
            state["userCursorCol"] -= 1
        elif matches_key(data, "right") and state["userCursorCol"] < 2:
            state["userCursorCol"] += 1
        elif matches_key(data, "return") or data == " ":
            row, col = state["userCursorRow"], state["userCursorCol"]
            if state["board"][row][col] == " ":
                await self._on_user_play(row, col)
            return
        else:
            return
        self._version += 1
        self._tui.request_render()

    def invalidate(self) -> None:
        self._cached_width = 0

    def render(self, width: int) -> list[str]:
        if width == self._cached_width and self._cached_version == self._version:
            return self._cached_lines

        state = self._state

        # Raw ANSI on purpose: this component owns its whole look.
        def bold(s: str) -> str:
            return f"\x1b[1m{s}\x1b[0m"

        def blue(s: str) -> str:
            return f"\x1b[34m{s}\x1b[0m"

        def yellow(s: str) -> str:
            return f"\x1b[33m{s}\x1b[0m"

        def green(s: str) -> str:
            return f"\x1b[32m{s}\x1b[0m"

        lines: list[str] = []

        # Top title banner, full width.
        title_text = " Tic-Tac-Toe "
        border_len = max(0, width - visible_width(title_text))
        left_border = border_len // 2
        lines.append(_dim("─" * left_border) + bold(blue(title_text)) + _dim("─" * (border_len - left_border)))

        lines.append("")

        # Status line.
        if state["status"] != "playing":
            status_text = {
                "draw": bold(yellow("Draw!")),
                "win_X": bold(green("X wins!")),
            }.get(state["status"], bold(yellow("O wins!")))
            lines.append(center_pad(status_text, width))
        elif state["currentTurn"] == "X":
            lines.append(center_pad(f"Turn: {bold(blue('X'))} (You)  {_dim('|')}  {bold(yellow('O'))} (Agent)", width))
        else:
            lines.append(center_pad(f"{blue('X')} (You)  {_dim('|')}  Turn: {bold(yellow('O'))} (Agent)", width))

        lines.extend(["", ""])
        lines.extend(render_visual_board(state, width))
        lines.extend(["", ""])

        # Footer.
        if state["status"] != "playing":
            footer = f"{bold('R')} restart  {_dim('|')}  {bold('Q')}/{bold('ESC')} quit"
        elif state["currentTurn"] != state["userMark"]:
            footer = _dim("Agent is thinking...")
        else:
            footer = f"{bold('←↑↓→')} move  {_dim('|')}  {bold('ENTER')} play  {_dim('|')}  {bold('ESC')} quit"
        lines.append(center_pad(footer, width))

        # Bottom separator between the component and the editor below.
        lines.append("")
        lines.append(_dim("─" * width))

        self._cached_lines = lines
        self._cached_width = width
        self._cached_version = self._version
        return lines


# ---------------------------------------------------------------------------
# Message renderer components
# ---------------------------------------------------------------------------


class BannerMessageComponent:
    """Full-width banner with an optional board snapshot underneath."""

    def __init__(self, title: str, details: dict | None, expanded: bool, theme) -> None:
        self._title = title
        self._details = details
        self._expanded = expanded
        self._theme = theme

    def invalidate(self) -> None:
        pass

    def render(self, width: int) -> list[str]:
        dim = lambda s: self._theme.fg("dim", s)
        fill_len = max(0, width - visible_width(self._title) - 2)
        left_fill = fill_len // 2
        lines = [f"{dim('─' * left_fill)} {self._title} {dim('─' * (fill_len - left_fill))}"]

        if self._expanded and self._details:
            lines.append("")
            lines.extend(render_board(self._details["board"], width))

        return lines


class GameOverMessageComponent:
    """End-of-game banner: two dim hrs, a big colored title line, and the
    final board with the winning line highlighted."""

    def __init__(self, status: str, details: dict | None, theme) -> None:
        self._status = status
        self._details = details
        self._theme = theme

    def invalidate(self) -> None:
        pass

    def render(self, width: int) -> list[str]:
        theme = self._theme
        dim = lambda s: theme.fg("dim", s)
        bold = theme.bold

        hr = dim("─" * width)
        lines = [hr, ""]

        if self._status == "win_X":
            title = bold(theme.fg("accent", "★ Player X wins ★"))
            sub = "You beat the agent."
        elif self._status == "win_O":
            title = bold(theme.fg("warning", "★ Player O wins ★"))
            sub = "The agent beat you."
        elif self._status == "draw":
            title = bold(theme.fg("muted", "— Draw —"))
            sub = "No winner."
        else:
            title = bold("Game over")
            sub = ""

        for line in (title, dim(sub)):
            pad = max(0, width - visible_width(line))
            lines.append(f"{' ' * (pad // 2)}{line}")

        lines.append("")
        if self._details:
            lines.extend(render_board(self._details["board"], width))
            lines.append("")
        lines.append(hr)

        return lines


# ---------------------------------------------------------------------------
# Extension
# ---------------------------------------------------------------------------

SAVE_TYPE = "tic-tac-toe-save"
MOVE_MESSAGE_TYPE = "tic-tac-toe-move"
GAME_OVER_MESSAGE_TYPE = "tic-tac-toe-game-over"

ACTION_DELAYS_S = {"move_up": 0.3, "move_down": 0.3, "move_left": 0.3, "move_right": 0.3, "play": 0.0}

GAME_INSTRUCTIONS = f"""

## Tic-Tac-Toe (you are Player O)

A tic-tac-toe game is in progress. The human is Player X. You are Player O.
The human plays through a TUI; you play through the `tic_tac_toe` tool.

### Turn protocol

When the human plays, you receive a message that contains the cell X marked,
the full board, and YOUR cursor position (Player O's cursor). The message is
the source of truth for the board.

Player O's cursor persists between O turns. It is reset to (row={AGENT_CURSOR_HOME_ROW}, col={AGENT_CURSOR_HOME_COL})
only after a successful `play`. If a `play` fails (cell already taken), the
cursor stays where it was, so you can move and retry.

You may also call `tic_tac_toe_see_board` if you want the current board and
your cursor position restated at any point. The user's cursor is private and
is never shown to you.

### The tool

`tic_tac_toe` takes ONE action per call:
- `move_up` / `move_down` / `move_left` / `move_right`: move YOUR cursor one cell (clamped at edges)
- `play`: place O on the cell under YOUR cursor. Errors if the cell is not empty.

There is no batched form. One call = one action.

### CRITICAL: emit the whole turn in a single response

To play at (r, c) from your cursor (r0, c0) emit, in order:
- `move_down` (r - r0) times (or `move_up` (r0 - r) times if r < r0)
- `move_right` (c - c0) times (or `move_left` (c0 - c) times if c < c0)
- one call of `play`

All of these tool calls MUST be emitted in the SAME assistant response, as
separate tool_use blocks, before you stop. Do not:
- split the sequence across multiple assistant responses,
- wait for a move result before emitting the next move or `play`,
- write any explanation or text between the tool calls,
- call any other tool during your turn (except `tic_tac_toe_see_board` when you
  explicitly need the state restated).

Decide the target cell first, then dump every action for the turn in one go.

### Examples (cursor starts at ({AGENT_CURSOR_HOME_ROW}, {AGENT_CURSOR_HOME_COL}))

- Target (0,0): one call, `play`.
- Target (0,2): `move_right`, `move_right`, `play`. Three calls, one response.
- Target (1,1): `move_down`, `move_right`, `play`. Three calls, one response.
- Target (2,2): `move_down`, `move_down`, `move_right`, `move_right`, `play`. Five calls, one response.

### Strategy

1. If you have two O's in a line with the third cell empty, win by playing there.
2. Otherwise, if X has two in a line with the third cell empty, block there.
3. Otherwise, prefer center, then corners, then edges.
"""


class TicTacToe:
    """pi keeps all of this in one factory closure, which is the JS idiom;
    methods on an object read better in Python at this size."""

    def __init__(self, pi) -> None:
        self.pi = pi
        self.state = create_initial_state()
        self.component: TicTacToeComponent | None = None
        self.game_active = False

    def wire(self) -> None:
        self.pi.on("session_start", self.on_session_event)
        self.pi.on("session_tree", self.on_session_event)
        self.pi.on("before_agent_start", self.on_before_agent_start)
        self.pi.register_message_renderer(MOVE_MESSAGE_TYPE, self.render_move_message)
        self.pi.register_message_renderer(GAME_OVER_MESSAGE_TYPE, self.render_game_over_message)
        self.pi.register_command("tic-tac-toe", handler=self.command, description="Play tic-tac-toe against the agent")
        self.register_tools()

    # -- state ---------------------------------------------------------------

    def reconstruct_state(self, ctx) -> None:
        self.state = create_initial_state()
        self.game_active = False

        for entry in ctx.session_manager.get_branch():
            if entry.get("type") != "message":
                continue
            msg = entry.get("message")
            if getattr(msg, "role", None) != "toolResult":
                continue
            if msg.tool_name not in ("tic_tac_toe", "tic_tac_toe_see_board"):
                continue

            details = msg.details
            if details:
                self.state["board"] = [list(row) for row in details["board"]]
                self.state["agentCursorRow"] = details["agentCursorRow"]
                self.state["agentCursorCol"] = details["agentCursorCol"]
                self.state["status"] = details["status"]
                self.state["currentTurn"] = details["currentTurn"]

    async def on_session_event(self, _event, ctx) -> None:
        self.reconstruct_state(ctx)

    def get_board_details(self) -> dict:
        """Persisted with each toolResult for state reconstruction AND sent to
        the agent as `details`. Only the agent cursor is included: the user
        cursor is private to the TUI."""
        return {
            "board": [list(row) for row in self.state["board"]],
            "agentCursorRow": self.state["agentCursorRow"],
            "agentCursorCol": self.state["agentCursorCol"],
            "status": self.state["status"],
            "currentTurn": self.state["currentTurn"],
        }

    def emit_game_over_message(self) -> None:
        """Sent once per game at end-of-game. The custom renderer paints the
        banner; `content` is a plain-text fallback for any non-TUI consumer
        and for the LLM (in case the message ends up in future context)."""
        label = {
            "win_X": "Player X (human) wins",
            "win_O": "Player O (agent) wins",
            "draw": "Draw",
        }.get(self.state["status"], "Game over")
        self.pi.send_message(
            {
                "customType": GAME_OVER_MESSAGE_TYPE,
                "content": f"Game over: {label}.",
                "display": True,
                "details": self.get_board_details(),
            }
        )

    # -- message renderers ---------------------------------------------------

    def render_move_message(self, message, options, theme):
        details = message.details
        if details and details.get("currentTurn") == "O":
            turn_label = f"{theme.fg('warning', theme.bold('O'))} (Agent)"
        else:
            turn_label = f"{theme.fg('accent', theme.bold('X'))} (You)"
        title = f"{theme.fg('accent', theme.bold('Player X played'))}  {theme.fg('dim', '→')}  next: {turn_label}"
        return BannerMessageComponent(title, details, options.get("expanded", False), theme)

    def render_game_over_message(self, message, _options, theme):
        details = message.details
        status = details.get("status", "draw") if details else "draw"
        return GameOverMessageComponent(status, details, theme)

    # -- events --------------------------------------------------------------

    async def on_before_agent_start(self, event, _ctx):
        """Inject the game instructions each turn while a game is active."""
        if not self.game_active:
            return None
        return {"systemPrompt": event["systemPrompt"] + GAME_INSTRUCTIONS}

    # -- /tic-tac-toe command ------------------------------------------------

    async def command(self, _args: str, ctx) -> None:
        if ctx.mode != "tui":
            ctx.ui.notify("Tic-tac-toe requires interactive mode", "error")
            return

        self.reconstruct_state(ctx)
        if self.state["status"] != "playing":
            self.state = create_initial_state()
        self.game_active = True
        await self.pi.set_session_name("Tic-Tac-Toe")

        async def on_user_play(row: int, col: int) -> None:
            state = self.state
            state["board"][row][col] = state["userMark"]
            state["status"] = check_win(state["board"])
            if state["status"] == "playing":
                state["currentTurn"] = state["agentMark"]
            if self.component is not None:
                self.component.update_state(state)
            await self.pi.append_entry(SAVE_TYPE, self.get_board_details())

            if state["status"] == "playing":
                # IMPORTANT: user play does NOT touch the agent cursor.
                # The agent cursor is only reset after a successful agent play.
                board_ascii = board_to_ascii(state["board"], state["agentCursorRow"], state["agentCursorCol"])
                self.pi.send_message(
                    {
                        "customType": MOVE_MESSAGE_TYPE,
                        "content": (
                            f"Player X played at (row={row}, col={col}). It is now Player O's turn.\n\n"
                            f"Board (your cursor marked with <>):\n{board_ascii}\n\n"
                            f"Your cursor is at (row={state['agentCursorRow']}, col={state['agentCursorCol']}). "
                            "Decide your target cell, then emit every move_* and the final play "
                            "as separate tic_tac_toe tool calls in THIS response."
                        ),
                        "display": True,
                        "details": self.get_board_details(),
                    },
                    {"triggerTurn": True},
                )
            else:
                self.emit_game_over_message()
                self.game_active = False

        def factory(tui, _theme, _kb, done):
            def close() -> None:
                self.component = None
                self.game_active = False
                done(None)

            self.component = TicTacToeComponent(tui, close, on_user_play, self.state)
            return self.component

        await ctx.ui.custom(factory)

    # -- tools ---------------------------------------------------------------

    async def execute_action(self, _tool_call_id, params, _cancel=None, _on_update=None, _ctx=None):
        action = params["action"]
        delay = ACTION_DELAYS_S[action]
        if delay > 0:
            await tonio.time.sleep(delay)

        state = self.state

        if action == "move_up":
            if state["agentCursorRow"] > 0:
                state["agentCursorRow"] -= 1
            result = f"Moved up. Cursor: ({state['agentCursorRow']}, {state['agentCursorCol']})"
        elif action == "move_down":
            if state["agentCursorRow"] < 2:
                state["agentCursorRow"] += 1
            result = f"Moved down. Cursor: ({state['agentCursorRow']}, {state['agentCursorCol']})"
        elif action == "move_left":
            if state["agentCursorCol"] > 0:
                state["agentCursorCol"] -= 1
            result = f"Moved left. Cursor: ({state['agentCursorRow']}, {state['agentCursorCol']})"
        elif action == "move_right":
            if state["agentCursorCol"] < 2:
                state["agentCursorCol"] += 1
            result = f"Moved right. Cursor: ({state['agentCursorRow']}, {state['agentCursorCol']})"
        else:  # play
            if state["status"] != "playing":
                raise TicTacToeError(f"Game is over ({state['status']}).")
            if state["currentTurn"] != state["agentMark"]:
                raise TicTacToeError("It is not your turn.")
            r, c = state["agentCursorRow"], state["agentCursorCol"]
            if state["board"][r][c] != " ":
                # Do NOT reset the cursor on failure. The agent can retry
                # from the cursor's current position.
                if self.component is not None:
                    self.component.update_state(state)
                await self.pi.append_entry(SAVE_TYPE, self.get_board_details())
                raise TicTacToeError(
                    f"Cell ({r},{c}) is already {state['board'][r][c]}. Your cursor is still at "
                    f"({r},{c}). Move to an empty cell and retry play."
                )
            state["board"][r][c] = state["agentMark"]
            state["status"] = check_win(state["board"])
            # Reset agent cursor to home ONLY on successful play.
            state["agentCursorRow"] = AGENT_CURSOR_HOME_ROW
            state["agentCursorCol"] = AGENT_CURSOR_HOME_COL
            if state["status"] == "playing":
                state["currentTurn"] = state["userMark"]
                result = (
                    f"Placed O at ({r},{c}). Cursor reset to "
                    f"({AGENT_CURSOR_HOME_ROW},{AGENT_CURSOR_HOME_COL}). Your turn, X!"
                )
            elif state["status"] == "win_O":
                result = f"Placed O at ({r},{c}). Player O wins!"
                self.game_active = False
                self.emit_game_over_message()
            elif state["status"] == "draw":
                result = f"Placed O at ({r},{c}). It's a draw!"
                self.game_active = False
                self.emit_game_over_message()
            else:
                result = f"Placed O at ({r},{c})."

        if self.component is not None:
            self.component.update_state(state)
        await self.pi.append_entry(SAVE_TYPE, self.get_board_details())

        return AgentToolResult(content=[TextContent(text=result)], details=self.get_board_details())

    async def execute_see_board(self, _tool_call_id, _params, _cancel=None, _on_update=None, _ctx=None):
        state = self.state
        board_ascii = board_to_ascii(state["board"], state["agentCursorRow"], state["agentCursorCol"])
        turn = "Player O (you)" if state["currentTurn"] == state["agentMark"] else "Player X"
        text = (
            f"Board (your cursor marked with <>):\n{board_ascii}\n\n"
            f"Your cursor: (row={state['agentCursorRow']}, col={state['agentCursorCol']})\n"
            f"Status: {state['status']}\n"
            f"Turn: {turn}"
        )
        return AgentToolResult(content=[TextContent(text=text)], details=self.get_board_details())

    def register_tools(self) -> None:
        def render_action_call(args, theme, _context):
            action = args.get("action") if isinstance(args, dict) else ""
            return Text(theme.fg("toolTitle", theme.bold("tic_tac_toe ")) + theme.fg("muted", action or ""), 0, 0)

        def render_action_result(result, options, theme, context):
            details = result["details"] if isinstance(result, dict) else result.details
            content = result["content"] if isinstance(result, dict) else result.content
            first = content[0] if content else None
            msg = getattr(first, "text", "") or ""
            is_error = bool(context.get("isError")) if context else False
            prefix = theme.fg("error", "✗ ") if is_error else theme.fg("success", "✓ ")
            summary = prefix + theme.fg("muted", msg)

            if options.get("expanded") and details:
                return BannerMessageComponent(summary, details, True, theme)
            return Text(summary, 0, 0)

        self.pi.register_tool(
            ToolDefinition(
                name="tic_tac_toe",
                label="Tic-Tac-Toe",
                description=(
                    "Execute ONE tic-tac-toe action as Player O. `action` is exactly one of: move_up, "
                    "move_down, move_left, move_right (move YOUR cursor one cell, clamped at edges), or "
                    "play (place O under YOUR cursor; errors if the cell is not empty). There is no "
                    "batched form. To play at (r, c) from your current cursor (r0, c0), emit the required "
                    "move_down/move_up and move_right/move_left calls, then play, all as separate "
                    "tool_use blocks in the SAME assistant response. Do not split the sequence across "
                    "responses and do not wait for a result before emitting the next call. Your cursor "
                    "position persists between turns and is reset to (0,0) only after a successful play."
                ),
                prompt_snippet="Play a tic-tac-toe action (move_up/down/left/right or play) as Player O",
                prompt_guidelines=[
                    (
                        "When it is your tic-tac-toe turn, decide the target cell first, then emit every "
                        "move_* plus the final play as separate tic_tac_toe tool calls in a SINGLE assistant "
                        "response. Never split them across responses or wait for intermediate results."
                    ),
                    (
                        "Never ask the user for the board. The board and your cursor position are included "
                        "in the user's move message; use tic_tac_toe_see_board if you need them restated."
                    ),
                ],
                parameters={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["move_up", "move_down", "move_left", "move_right", "play"],
                            "description": (
                                "The single action to perform this call. Emit multiple tic_tac_toe calls "
                                "in one response to string actions together."
                            ),
                        }
                    },
                    "required": ["action"],
                },
                execution_mode="sequential",
                execute=self.execute_action,
                render_call=render_action_call,
                render_result=render_action_result,
            )
        )

        def render_see_board_call(_args, theme, _context):
            return Text(theme.fg("toolTitle", theme.bold("tic_tac_toe_see_board")), 0, 0)

        def render_see_board_result(result, options, theme, _context):
            details = result["details"] if isinstance(result, dict) else result.details
            row = details.get("agentCursorRow", 0) if details else 0
            col = details.get("agentCursorCol", 0) if details else 0
            summary = theme.fg("success", "✓ ") + theme.fg("muted", f"cursor ({row},{col})")
            if options.get("expanded") and details:
                return BannerMessageComponent(summary, details, True, theme)
            return Text(summary, 0, 0)

        self.pi.register_tool(
            ToolDefinition(
                name="tic_tac_toe_see_board",
                label="See Board",
                description=(
                    "Return the current tic-tac-toe board state and YOUR cursor position (Player O). "
                    "Takes no arguments. Use this if you need the current state restated mid-turn (for "
                    "example after a failed play). The user's cursor is never exposed."
                ),
                prompt_snippet="Inspect the tic-tac-toe board and your cursor",
                parameters={"type": "object", "properties": {}},
                execute=self.execute_see_board,
                render_call=render_see_board_call,
                render_result=render_see_board_result,
            )
        )


def extension(pi):
    TicTacToe(pi).wire()
