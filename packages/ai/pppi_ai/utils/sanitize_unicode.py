"""Port of pi's surrogate sanitizer (packages/ai/src/utils/sanitize-unicode.ts).

Unpaired surrogate code points (typically produced by lone `\\uD800`-range
escapes in provider JSON, or corrupted text processing) break JSON
serialization at many API providers. Properly paired astral characters are a
single code point in Python and are never affected.
"""

import re


_UNPAIRED_SURROGATES = re.compile(
    "[\ud800-\udbff](?![\udc00-\udfff])|(?<![\ud800-\udbff])[\udc00-\udfff]",
)


def sanitize_surrogates(text: str) -> str:
    """Remove unpaired Unicode surrogate characters from a string."""
    return _UNPAIRED_SURROGATES.sub("", text)
