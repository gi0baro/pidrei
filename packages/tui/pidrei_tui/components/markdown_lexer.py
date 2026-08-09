"""Public access to the bundled Markdown lexer.

pi's ``tui/src/index.ts`` re-exports ``Marked`` (plus its ``Token``/``Tokens``
types) from the bundled ``marked`` dependency so extensions can parse Markdown
the same way the renderer does, without adding a dependency of their own.

pidrei's renderer parses through :mod:`._marked`, a markdown-it-py adapter that
emits marked-shaped token dicts. There is no ``Marked`` class to hand out — the
adapter is a single function — so the exported surface is that function.
Tokens are plain dicts keyed exactly like marked's (``type``, ``raw``,
``text``, ``tokens``, ...), which is what a transformer or a custom renderer
needs to walk them.
"""

from ._marked import lex


__all__ = ["lex_markdown"]


def lex_markdown(text: str) -> list[dict]:
    """Tokenize ``text`` into the marked-shaped token stream the renderer uses."""
    return lex(text)
