"""Mirror of pi coding-agent src/utils/syntax-highlight.ts.

pi highlights through highlight.js and re-renders the HTML spans with
scope-keyed formatters. pidrei uses pygments instead: tokens map straight to
the same scope names pi's themes key on (keyword, string, comment, ...), so
theme dicts translate 1:1. Tokenization differences between the two engines
mean colors can differ per language, but the theme surface is identical.
"""

from pygments import lex
from pygments.lexers import get_lexer_by_name
from pygments.token import Token
from pygments.util import ClassNotFound


# Most-specific-first mapping from pygments token types to the highlight.js
# scope names pi's CliHighlightTheme uses.
_TOKEN_SCOPES = [
    (Token.Comment.Preproc, "meta"),
    (Token.Comment.PreprocFile, "meta"),
    (Token.Comment, "comment"),
    (Token.Keyword.Type, "type"),
    (Token.Keyword.Constant, "literal"),
    (Token.Keyword, "keyword"),
    (Token.Operator.Word, "keyword"),
    (Token.Operator, "operator"),
    (Token.Punctuation, "punctuation"),
    (Token.Name.Function, "function"),
    (Token.Name.Class, "class"),
    (Token.Name.Builtin, "built_in"),
    (Token.Name.Decorator, "meta"),
    (Token.Name.Tag, "tag"),
    (Token.Name.Attribute, "attr"),
    (Token.Name.Variable, "variable"),
    (Token.Name.Constant, "variable"),
    (Token.String.Regex, "regexp"),
    (Token.String, "string"),
    (Token.Number, "number"),
    (Token.Generic.Inserted, "addition"),
    (Token.Generic.Deleted, "deletion"),
    (Token.Generic.Emph, "emphasis"),
    (Token.Generic.Strong, "strong"),
    (Token.Generic.Heading, "section"),
    (Token.Generic.Subheading, "section"),
]

_lexer_cache: dict = {}


def _get_lexer(name: str):
    if name not in _lexer_cache:
        try:
            _lexer_cache[name] = get_lexer_by_name(name, stripnl=False, ensurenl=False)
        except ClassNotFound:
            _lexer_cache[name] = None
    return _lexer_cache[name]


def _scope_for_token(token_type) -> str | None:
    for candidate, scope in _TOKEN_SCOPES:
        if token_type in candidate:
            return scope
    return None


def _get_scope_formatter(scope: str, theme: dict):
    exact = theme.get(scope)
    if exact:
        return exact

    dot_index = scope.find(".")
    if dot_index != -1:
        prefix_formatter = theme.get(scope[:dot_index])
        if prefix_formatter:
            return prefix_formatter

    dash_index = scope.find("-")
    if dash_index != -1:
        prefix_formatter = theme.get(scope[:dash_index])
        if prefix_formatter:
            return prefix_formatter

    return None


def _get_active_formatter(scope: str | None, theme: dict):
    if scope is not None:
        formatter = _get_scope_formatter(scope, theme)
        if formatter:
            return formatter
    return theme.get("default")


def highlight(
    code: str,
    *,
    language: str | None = None,
    ignore_illegals: bool = True,
    theme: dict | None = None,
) -> str:
    """Highlight ``code`` and return one ANSI-styled string.

    Unlike pi there is no auto-detection fallback: callers always validate the
    language first (pi's callers do too, to avoid misdetection artifacts).
    """
    lexer = _get_lexer(language) if language else None
    if lexer is None:
        return code
    theme = theme or {}
    output = []
    for token_type, text in lex(code, lexer):
        if not text:
            continue
        formatter = _get_active_formatter(_scope_for_token(token_type), theme)
        output.append(formatter(text) if formatter else text)
    return "".join(output)


def supports_language(name: str) -> bool:
    return _get_lexer(name) is not None
