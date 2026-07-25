"""Minimal port of chalk (level 3) for test themes.

Mirrors the pieces pi's tests rely on: SGR open/close pairs per style,
chainable compound styles (``chalk.bold.cyan``), empty input passthrough, and
chalk's nested-close replacement (occurrences of a style's close code inside
the content are replaced with its open code so outer styles stay active).
"""

__all__ = ["chalk"]

_STYLES = {
    "reset": (0, 0),
    "bold": (1, 22),
    "dim": (2, 22),
    "italic": (3, 23),
    "underline": (4, 24),
    "inverse": (7, 27),
    "hidden": (8, 28),
    "strikethrough": (9, 29),
    "black": (30, 39),
    "red": (31, 39),
    "green": (32, 39),
    "yellow": (33, 39),
    "blue": (34, 39),
    "magenta": (35, 39),
    "cyan": (36, 39),
    "white": (37, 39),
    "gray": (90, 39),
    "bgBlack": (40, 49),
    "bgRed": (41, 49),
    "bgGreen": (42, 49),
    "bgYellow": (43, 49),
    "bgBlue": (44, 49),
    "bgMagenta": (45, 49),
    "bgCyan": (46, 49),
    "bgWhite": (47, 49),
}


class Chalk:
    __slots__ = ("_parts",)

    def __init__(self, parts: tuple = ()) -> None:
        self._parts = parts

    def __getattr__(self, name: str) -> Chalk:
        try:
            pair = _STYLES[name]
        except KeyError:
            raise AttributeError(name) from None
        return Chalk((*self._parts, pair))

    def __call__(self, text: str) -> str:
        if not text:
            return text
        result = text
        if "\x1b" in result:
            for open_code, close_code in reversed(self._parts):
                result = result.replace(f"\x1b[{close_code}m", f"\x1b[{open_code}m")
        opens = "".join(f"\x1b[{open_code}m" for open_code, _ in self._parts)
        closes = "".join(f"\x1b[{close_code}m" for _, close_code in reversed(self._parts))
        return opens + result + closes


chalk = Chalk()
