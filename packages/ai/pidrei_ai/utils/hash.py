"""Port of pi's short string hash (packages/ai/src/utils/hash.ts).

Bit-exact with the JS original: iterates UTF-16 code units (charCodeAt) and
mirrors Math.imul / `>>>` semantics on 32-bit patterns.
"""

_MASK32 = 0xFFFFFFFF
_BASE36_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"


def _imul(a: int, b: int) -> int:
    return (a * b) & _MASK32


def _to_base36(value: int) -> str:
    if value == 0:
        return "0"
    digits: list[str] = []
    while value:
        value, remainder = divmod(value, 36)
        digits.append(_BASE36_ALPHABET[remainder])
    return "".join(reversed(digits))


def short_hash(text: str) -> str:
    """Fast deterministic hash to shorten long strings."""
    h1 = 0xDEADBEEF
    h2 = 0x41C6CE57

    data = text.encode("utf-16-le", "surrogatepass")
    for index in range(0, len(data), 2):
        unit = data[index] | (data[index + 1] << 8)
        h1 = _imul(h1 ^ unit, 2654435761)
        h2 = _imul(h2 ^ unit, 1597334677)

    mixed1 = (_imul(h1 ^ (h1 >> 16), 2246822507) ^ _imul(h2 ^ (h2 >> 13), 3266489909)) & _MASK32
    mixed2 = (_imul(h2 ^ (h2 >> 16), 2246822507) ^ _imul(mixed1 ^ (mixed1 >> 13), 3266489909)) & _MASK32
    return _to_base36(mixed2) + _to_base36(mixed1)
