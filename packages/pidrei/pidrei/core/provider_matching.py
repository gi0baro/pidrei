"""Port of pi's core/provider-matching.ts.

pi matches any object with `id` and `name`. Here the records differ by caller
(interactive mode's login options are dicts, the auth commands match
`Provider` objects), so the caller says how to read the two names.
"""

from collections.abc import Callable, Sequence


def find_matching_providers[T](
    providers: Sequence[T], provider_ref: str, names: Callable[[T], tuple[str, str]]
) -> list[T]:
    """Match provider IDs and display names exactly, ignoring case and surrounding whitespace in the reference.
    `names` returns a provider's `(id, name)`."""
    normalized = provider_ref.strip().lower()
    if not normalized:
        return []
    matches: list[T] = []
    for provider in providers:
        provider_id, name = names(provider)
        if provider_id.lower() == normalized or name.lower() == normalized:
            matches.append(provider)
    return matches
