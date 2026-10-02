"""Mirror of pi codemode src/identifier.ts."""

import keyword
import re


_VALID_FIRST = re.compile(r"[A-Za-z_]")
_VALID_REST = re.compile(r"[A-Za-z0-9_]")


def to_codemode_identifier(name: str) -> str:
    """The identifier a script uses for a tool: characters that are not valid in
    a Python identifier become `_`, and a keyword gets a trailing `_`.
    `mcp__docs__search` stays as is, `my-tool` becomes `my_tool`, `class`
    becomes `class_`."""
    identifier = ""
    for char in name:
        valid = _VALID_FIRST if identifier == "" else _VALID_REST
        identifier += char if valid.fullmatch(char) else "_"
    if identifier == "":
        return "_"
    return f"{identifier}_" if keyword.iskeyword(identifier) else identifier


def is_identifier(name: str) -> bool:
    """Whether `name` can be written as is in a script: a plain identifier that
    is not a keyword."""
    return name.isidentifier() and name.isascii() and not keyword.iskeyword(name)
