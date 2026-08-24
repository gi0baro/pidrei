"""Mirror of pi coding-agent src/utils/text.ts."""


def split_bom(content: str) -> tuple[str, str]:
    """Split a leading UTF-8 byte order mark from decoded text.

    pi returns `{ bom, text }`; here it is the same pair as a tuple.
    """
    return ("﻿", content[1:]) if content.startswith("﻿") else ("", content)


def strip_bom(content: str) -> str:
    """Remove a leading UTF-8 byte order mark from decoded text."""
    return split_bom(content)[1]
