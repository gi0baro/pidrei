"""Mirror of pi coding-agent src/modes/interactive/components/pi-logo.ts."""

from pidrei_tui import background_ansi, foreground_ansi, is_apple_terminal_session, rgb_color

from ..theme import theme


_CORAL = rgb_color(228, 138, 122)
_BLUE = rgb_color(79, 142, 179)
_YELLOW = rgb_color(234, 182, 93)
_RESET = "\x1b[0m"


def pi_logo_lines() -> tuple[str, str]:
    """The pidrei logo: pi's logo followed by a 3 colored like it (coral top
    bar and right pixel below it, blue below), 8 cells wide and 2 lines tall.
    Each cell shows two square pixels with half blocks::

        coral coral coral .      .  coral coral coral
        blue  .     coral .      .  .     blue  coral
        blue  blue  .     yellow .  .     .     blue
        blue  .     .     yellow .  blue  blue  blue

    The brand colors stay fixed across themes; they follow the terminal's
    color mode. The 3 is pidrei-only; pi's logo stops at the fourth cell.
    """
    mode = theme.get_color_mode()

    def fg(color) -> str:
        return foreground_ansi(color, mode)

    # The fourth cell of the top line is empty, so it is padded to the same
    # width as the bottom line.
    top = f"{fg(_CORAL)}{background_ansi(_BLUE, mode)}▀{_RESET}{fg(_CORAL)}▀█{_RESET} "
    bottom = f"{fg(_BLUE)}█▀{_RESET} {fg(_YELLOW)}█{_RESET}"
    three_top = f"{fg(_CORAL)}▀{background_ansi(_BLUE, mode)}▀{_RESET}{fg(_CORAL)}█{_RESET}"
    return f"{top} {three_top}", f"{bottom} {fg(_BLUE)}▄▄█{_RESET}"


def supports_pi_logo() -> bool:
    """Whether the terminal renders the half-block logo correctly. Apple
    Terminal draws gaps between rows and misaligns the half blocks, so it gets
    the text wordmark instead."""
    return not is_apple_terminal_session()


def pi_wordmark() -> str:
    """Text fallback for the logo: "PiDrei" in the logo's colors, "Pi" coral
    and yellow like pi's wordmark and "Drei" blue like the logo's 3."""
    mode = theme.get_color_mode()
    return (
        f"{foreground_ansi(_CORAL, mode)}P{_RESET}{foreground_ansi(_YELLOW, mode)}i{_RESET}"
        f"{foreground_ansi(_BLUE, mode)}Drei{_RESET}"
    )
