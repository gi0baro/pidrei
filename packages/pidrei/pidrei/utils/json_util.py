"""Mirror of pi coding-agent src/utils/json.ts."""

import re


_STRING_OR_LINE_COMMENT = re.compile(r'"(?:\\.|[^"\\])*"|//[^\n]*')
_STRING_OR_TRAILING_COMMA = re.compile(r'"(?:\\.|[^"\\])*"|,(\s*[}\]])')


def strip_json_comments(input: str) -> str:
    """Strip `//` line comments and trailing commas from JSON, leaving string literals untouched."""
    without_comments = _STRING_OR_LINE_COMMENT.sub(lambda m: m.group(0) if m.group(0)[0] == '"' else "", input)
    return _STRING_OR_TRAILING_COMMA.sub(
        lambda m: m.group(1) if m.group(1) is not None else (m.group(0) if m.group(0)[0] == '"' else ""),
        without_comments,
    )
