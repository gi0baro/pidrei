"""Mirror of pi coding-agent src/modes/interactive/components/session-selector-search.ts.

Sort modes: "threaded" | "recent" | "relevance". Name filters: "all" |
"named". Parsed queries are ``{"mode", "tokens", "regex", "error"?}``
records with ``{"kind", "value"}`` tokens.
"""

import re
from dataclasses import dataclass

from pidrei_tui import fuzzy_match


_WS_RE = re.compile(r"\s+")


def _normalize_whitespace_lower(text: str) -> str:
    return _WS_RE.sub(" ", text.lower()).strip()


# Match tiers, best first. Each token is scored by the best field it matches; tiers dominate
# fuzzy scores so an exact hit anywhere beats a scattered subsequence in a short field.
_TIER_NAME = 0  # query text appears in the session name
_TIER_TEXT = 1  # query text appears in the session id, messages, or cwd
_TIER_FUZZY = 2  # query characters appear scattered in the session name or cwd
_TIER_WEIGHT = 1_000_000


@dataclass(slots=True)
class _LowercaseText:
    source: str
    lower: str
    normalized: str | None = None


class MessageTextCache:
    """Lowercased message text per session.

    `all_messages_text` holds every user/assistant message and can be megabytes, so it is
    lowercased once instead of on every keystroke. pi keeps one module-wide `WeakMap` keyed by
    the session object; here the session list owns the cache (its filtering runs under the UI
    state lock), keyed by session file. An entry is reused only while the session still holds
    the same text object, so a reloaded session is lowercased again.
    """

    def __init__(self) -> None:
        self._by_path: dict[str, _LowercaseText] = {}

    def get(self, session) -> _LowercaseText:
        cached = self._by_path.get(session.path)
        if cached is None or cached.source is not session.all_messages_text:
            cached = _LowercaseText(session.all_messages_text, session.all_messages_text.lower())
            self._by_path[session.path] = cached
        return cached

    def get_normalized(self, session) -> str:
        cached = self.get(session)
        if cached.normalized is None:
            cached.normalized = _WS_RE.sub(" ", cached.lower).strip()
        return cached.normalized


def _tier_score(tier: int, score: float = 0) -> float:
    return tier * _TIER_WEIGHT + score


def _match_literal_token(session, value: str, messages: MessageTextCache) -> float | None:
    """Exact (case-insensitive) text match. Messages and ids are only matched exactly: scattered matches in them are noise."""
    name = (session.name or "").lower()
    if value in name:
        return _tier_score(_TIER_NAME, fuzzy_match(value, name)["score"])
    if value in session.id.lower():
        return _tier_score(_TIER_TEXT)
    if value in messages.get(session).lower:
        return _tier_score(_TIER_TEXT)
    if value in session.cwd.lower():
        return _tier_score(_TIER_TEXT)
    return None


def _match_fuzzy_token(session, value: str, messages: MessageTextCache) -> float | None:
    literal = _match_literal_token(session, value.lower(), messages)
    if literal is not None:
        return literal

    best: float | None = None
    for field in (session.name or "", session.cwd):
        m = fuzzy_match(value, field)
        if m["matches"] and (best is None or m["score"] < best):
            best = m["score"]
    return None if best is None else _tier_score(_TIER_FUZZY, best)


def _match_phrase_token(session, phrase: str, messages: MessageTextCache) -> float | None:
    name = _normalize_whitespace_lower(session.name or "")
    if phrase in name:
        return _tier_score(_TIER_NAME, fuzzy_match(phrase, name)["score"])
    if phrase in session.id.lower():
        return _tier_score(_TIER_TEXT)
    if phrase in messages.get_normalized(session):
        return _tier_score(_TIER_TEXT)
    if phrase in _normalize_whitespace_lower(session.cwd):
        return _tier_score(_TIER_TEXT)
    return None


def _match_regex(session, regex: re.Pattern) -> float | None:
    if session.name and regex.search(session.name):
        return _tier_score(_TIER_NAME)
    if regex.search(session.id) or regex.search(session.all_messages_text) or regex.search(session.cwd):
        return _tier_score(_TIER_TEXT)
    return None


def has_session_name(session) -> bool:
    return bool(session.name and session.name.strip())


def _matches_name_filter(session, name_filter: str) -> bool:
    if name_filter == "all":
        return True
    return has_session_name(session)


def parse_search_query(query: str) -> dict:
    trimmed = query.strip()
    if not trimmed:
        return {"mode": "tokens", "tokens": [], "regex": None}

    # Regex mode: re:<pattern>
    if trimmed.startswith("re:"):
        pattern = trimmed[3:].strip()
        if not pattern:
            return {"mode": "regex", "tokens": [], "regex": None, "error": "Empty regex"}
        try:
            return {"mode": "regex", "tokens": [], "regex": re.compile(pattern, re.IGNORECASE)}
        except re.error as err:
            return {"mode": "regex", "tokens": [], "regex": None, "error": str(err)}

    # Token mode with quote support.
    # Example: foo "node cve" bar
    tokens: list = []
    buf = ""
    in_quote = False
    had_unclosed_quote = False

    def flush(kind: str) -> None:
        nonlocal buf
        value = buf.strip()
        buf = ""
        if not value:
            return
        tokens.append({"kind": kind, "value": value})

    for ch in trimmed:
        if ch == '"':
            if in_quote:
                flush("phrase")
                in_quote = False
            else:
                flush("fuzzy")
                in_quote = True
            continue

        if not in_quote and ch.isspace():
            flush("fuzzy")
            continue

        buf += ch

    if in_quote:
        had_unclosed_quote = True

    # If quotes were unbalanced, fall back to plain whitespace tokenization.
    if had_unclosed_quote:
        return {
            "mode": "tokens",
            "tokens": [{"kind": "fuzzy", "value": t} for t in trimmed.split() if t],
            "regex": None,
        }

    flush("phrase" if in_quote else "fuzzy")

    return {"mode": "tokens", "tokens": tokens, "regex": None}


def match_session(session, parsed: dict, messages: MessageTextCache | None = None) -> dict:
    """Returns ``{"matches", "score"}``; lower score is better.

    Each field is matched separately, so a token never matches across field boundaries.
    Different tokens may match different fields.
    """
    if parsed["mode"] == "regex":
        if parsed["regex"] is None:
            return {"matches": False, "score": 0}
        score = _match_regex(session, parsed["regex"])
        return {"matches": False, "score": 0} if score is None else {"matches": True, "score": score}

    if messages is None:
        messages = MessageTextCache()
    total_score = 0.0

    for token in parsed["tokens"]:
        if token["kind"] == "phrase":
            phrase = _normalize_whitespace_lower(token["value"])
            if not phrase:
                continue
            score = _match_phrase_token(session, phrase, messages)
        else:
            score = _match_fuzzy_token(session, token["value"], messages)
        if score is None:
            return {"matches": False, "score": 0}
        total_score += score

    return {"matches": True, "score": total_score}


def filter_and_sort_sessions(
    sessions: list, query: str, sort_mode: str, name_filter: str = "all", messages: MessageTextCache | None = None
) -> list:
    """`messages` is the caller's lowercased message text; without one, each call lowercases afresh."""
    if name_filter == "all":
        name_filtered = sessions
    else:
        name_filtered = [session for session in sessions if _matches_name_filter(session, name_filter)]
    trimmed = query.strip()
    if not trimmed:
        return name_filtered

    parsed = parse_search_query(query)
    if parsed.get("error"):
        return []
    if messages is None:
        messages = MessageTextCache()

    # Recent mode: filter only, keep incoming order.
    if sort_mode == "recent":
        return [s for s in name_filtered if match_session(s, parsed, messages)["matches"]]

    # Relevance mode: sort by score, tie-break by modified desc.
    scored = []
    for s in name_filtered:
        res = match_session(s, parsed, messages)
        if not res["matches"]:
            continue
        scored.append((s, res["score"]))

    scored.sort(key=lambda entry: (entry[1], -entry[0].modified.timestamp()))

    return [entry[0] for entry in scored]
