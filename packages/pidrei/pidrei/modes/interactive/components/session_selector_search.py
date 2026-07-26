"""Mirror of pi coding-agent src/modes/interactive/components/session-selector-search.ts.

Sort modes: "threaded" | "recent" | "relevance". Name filters: "all" |
"named". Parsed queries are ``{"mode", "tokens", "regex", "error"?}``
records with ``{"kind", "value"}`` tokens.
"""

import re

from pidrei_tui import fuzzy_match


_WS_RE = re.compile(r"\s+")


def _normalize_whitespace_lower(text: str) -> str:
    return _WS_RE.sub(" ", text.lower()).strip()


def _get_session_search_text(session) -> str:
    return f"{session.id} {session.name or ''} {session.all_messages_text} {session.cwd}"


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


def match_session(session, parsed: dict) -> dict:
    """Returns ``{"matches", "score"}``; lower score is better."""
    text = _get_session_search_text(session)

    if parsed["mode"] == "regex":
        if parsed["regex"] is None:
            return {"matches": False, "score": 0}
        match = parsed["regex"].search(text)
        if match is None:
            return {"matches": False, "score": 0}
        return {"matches": True, "score": match.start() * 0.1}

    if not parsed["tokens"]:
        return {"matches": True, "score": 0}

    total_score = 0.0
    normalized_text: str | None = None

    for token in parsed["tokens"]:
        if token["kind"] == "phrase":
            if normalized_text is None:
                normalized_text = _normalize_whitespace_lower(text)
            phrase = _normalize_whitespace_lower(token["value"])
            if not phrase:
                continue
            idx = normalized_text.find(phrase)
            if idx < 0:
                return {"matches": False, "score": 0}
            total_score += idx * 0.1
            continue

        m = fuzzy_match(token["value"], text)
        if not m["matches"]:
            return {"matches": False, "score": 0}
        total_score += m["score"]

    return {"matches": True, "score": total_score}


def filter_and_sort_sessions(sessions: list, query: str, sort_mode: str, name_filter: str = "all") -> list:
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

    # Recent mode: filter only, keep incoming order.
    if sort_mode == "recent":
        return [s for s in name_filtered if match_session(s, parsed)["matches"]]

    # Relevance mode: sort by score, tie-break by modified desc.
    scored = []
    for s in name_filtered:
        res = match_session(s, parsed)
        if not res["matches"]:
            continue
        scored.append((s, res["score"]))

    scored.sort(key=lambda entry: (entry[1], -entry[0].modified.timestamp()))

    return [entry[0] for entry in scored]
