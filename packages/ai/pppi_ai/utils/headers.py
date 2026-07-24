"""Header helpers ported from pi (packages/ai/src/utils/headers.ts, models.ts)."""

from collections.abc import Mapping


def provider_headers_to_record(headers: Mapping[str, str | None] | None) -> dict[str, str] | None:
    """Port of `providerHeadersToRecord`: drop null values, None when empty."""
    if not headers:
        return None
    result = {key: value for key, value in headers.items() if value is not None}
    return result or None


def merge_headers(
    base: Mapping[str, str | None] | None,
    override: Mapping[str, str | None] | None,
) -> dict[str, str | None] | None:
    """Port of `mergeHeaders` (models.ts): case-insensitive override.

    An override replaces any base entry with the same lowercased name, keeps
    the override's casing, and lands at the end of the mapping (insertion
    order mirrors the JS delete-then-set behavior).
    """
    if base is None and override is None:
        return None
    merged: dict[str, str | None] = dict(base or {})
    for name, value in (override or {}).items():
        lower_name = name.lower()
        for existing_name in list(merged):
            if existing_name.lower() == lower_name:
                del merged[existing_name]
        merged[name] = value
    return merged
