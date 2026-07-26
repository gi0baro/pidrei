"""Mirror of pi coding-agent src/utils/git.ts.

Git sources are ``{"type", "repo", "host", "path", "ref"?, "pinned"}``
records. Deviation: pi resolves hosted shorthands through the
hosted-git-info package; this port recognizes the common hosted prefixes
(github:, gitlab:, bitbucket:, and implicit github user/repo) and otherwise
uses the same generic URL parser.
"""

import re
import urllib.parse


_SCP_LIKE_RE = re.compile(r"^git@([^:]+):(.+)$")
_PROTOCOL_RE = re.compile(r"^(https?|ssh|git)://", re.IGNORECASE)

_HOSTED_DOMAINS = {
    "github": "github.com",
    "gitlab": "gitlab.com",
    "bitbucket": "bitbucket.org",
    "sourcehut": "git.sr.ht",
}


def _split_ref(url: str) -> dict:
    scp_like_match = _SCP_LIKE_RE.match(url)
    if scp_like_match:
        path_with_maybe_ref = scp_like_match.group(2) or ""
        ref_separator = path_with_maybe_ref.find("@")
        if ref_separator < 0:
            return {"repo": url, "ref": None}
        repo_path = path_with_maybe_ref[:ref_separator]
        ref = path_with_maybe_ref[ref_separator + 1 :]
        if not repo_path or not ref:
            return {"repo": url, "ref": None}
        return {"repo": f"git@{scp_like_match.group(1) or ''}:{repo_path}", "ref": ref}

    if "://" in url:
        try:
            parsed = urllib.parse.urlsplit(url)
            path_with_maybe_ref = parsed.path.lstrip("/")
            ref_separator = path_with_maybe_ref.find("@")
            if ref_separator < 0:
                return {"repo": url, "ref": None}
            repo_path = path_with_maybe_ref[:ref_separator]
            ref = path_with_maybe_ref[ref_separator + 1 :]
            if not repo_path or not ref:
                return {"repo": url, "ref": None}
            rebuilt = urllib.parse.urlunsplit(
                (parsed.scheme, parsed.netloc, f"/{repo_path}", parsed.query, parsed.fragment)
            )
            return {"repo": rebuilt.rstrip("/"), "ref": ref}
        except ValueError:
            return {"repo": url, "ref": None}

    slash_index = url.find("/")
    if slash_index < 0:
        return {"repo": url, "ref": None}
    host = url[:slash_index]
    path_with_maybe_ref = url[slash_index + 1 :]
    ref_separator = path_with_maybe_ref.find("@")
    if ref_separator < 0:
        return {"repo": url, "ref": None}
    repo_path = path_with_maybe_ref[:ref_separator]
    ref = path_with_maybe_ref[ref_separator + 1 :]
    if not repo_path or not ref:
        return {"repo": url, "ref": None}
    return {"repo": f"{host}/{repo_path}", "ref": ref}


def _decode_for_validation(value: str) -> str | None:
    try:
        return urllib.parse.unquote(value, errors="strict")
    except UnicodeDecodeError, ValueError:
        return None


def _has_unsafe_git_install_part(value: str, allow_slash: bool) -> bool:
    decoded = _decode_for_validation(value)
    if decoded is None:
        return True
    for candidate in (value, decoded):
        if "\0" in candidate or "\\" in candidate or candidate.startswith("/"):
            return True
        if not allow_slash and "/" in candidate:
            return True
        if ".." in candidate.split("/"):
            return True
    return False


def _build_git_source(*, repo: str, host: str, path: str, ref: str | None) -> dict | None:
    if path.startswith("/"):
        return None
    normalized_path = re.sub(r"\.git$", "", path).lstrip("/")
    if not host or not normalized_path or len(normalized_path.split("/")) < 2:
        return None
    if _has_unsafe_git_install_part(host, False) or _has_unsafe_git_install_part(normalized_path, True):
        return None

    return {
        "type": "git",
        "repo": repo,
        "host": host,
        "path": normalized_path,
        "ref": ref,
        "pinned": bool(ref),
    }


def _parse_generic_git_url(url: str) -> dict | None:
    split = _split_ref(url)
    repo_without_ref = split["repo"]
    ref = split["ref"]
    repo = repo_without_ref
    host = ""
    path = ""

    scp_like_match = _SCP_LIKE_RE.match(repo_without_ref)
    if scp_like_match:
        host = scp_like_match.group(1) or ""
        path = scp_like_match.group(2) or ""
    elif repo_without_ref.startswith(("https://", "http://", "ssh://", "git://")):
        try:
            parsed = urllib.parse.urlsplit(repo_without_ref)
            host = parsed.hostname or ""
            path = parsed.path.lstrip("/")
        except ValueError:
            return None
    else:
        slash_index = repo_without_ref.find("/")
        if slash_index < 0:
            return None
        host = repo_without_ref[:slash_index]
        path = repo_without_ref[slash_index + 1 :]
        if "." not in host and host != "localhost":
            return None
        repo = f"https://{repo_without_ref}"

    return _build_git_source(repo=repo, host=host, path=path, ref=ref)


def _parse_hosted_shorthand(url: str) -> dict | None:
    """Resolve hosted-git-info style shorthands (github:user/repo, user/repo)."""
    shorthand = url
    domain = None
    for prefix, mapped_domain in _HOSTED_DOMAINS.items():
        if shorthand.lower().startswith(f"{prefix}:"):
            shorthand = shorthand[len(prefix) + 1 :]
            domain = mapped_domain
            break

    if _PROTOCOL_RE.match(shorthand) or _SCP_LIKE_RE.match(shorthand) or "://" in shorthand:
        return None

    # Split a #committish (hosted-git-info) or @ref (pi shorthand)
    ref = None
    for separator in ("#", "@"):
        separator_index = shorthand.find(separator)
        if separator_index >= 0:
            ref = shorthand[separator_index + 1 :] or None
            shorthand = shorthand[:separator_index]
            break

    parts = [part for part in shorthand.split("/") if part]
    if len(parts) != 2:
        return None
    if domain is None:
        # Implicit github only when the first segment is not a domain
        if "." in parts[0]:
            return None
        domain = "github.com"

    user, project = parts
    return _build_git_source(repo=f"https://{domain}/{user}/{project}", host=domain, path=f"{user}/{project}", ref=ref)


def parse_git_url(source: str) -> dict | None:
    """Parse a git source into a GitSource record.

    Rules:
    - With git: prefix, accept all historical shorthand forms.
    - Without git: prefix, only accept explicit protocol URLs.
    """
    trimmed = source.strip()
    has_git_prefix = trimmed.startswith("git:") and not trimmed.startswith("git://")
    url = trimmed[4:].strip() if has_git_prefix else trimmed

    if not has_git_prefix and not re.match(r"^(https?|ssh|git)://", url, re.IGNORECASE):
        return None

    if has_git_prefix:
        hosted = _parse_hosted_shorthand(url)
        if hosted is not None:
            return hosted

    return _parse_generic_git_url(url)
