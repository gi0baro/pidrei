"""GitHub Issue Autocomplete

Requires the GitHub CLI (`gh`) and a GitHub repository checkout. Preloads the
latest open issues once per session, then filters them locally for fast `#...`
completion. The provider wraps the built-in one and defers to it whenever the
cursor is not on a `#` token.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/github_issue_autocomplete.py
"""

import json
import re

import tonio.colored as tonio

from pidrei_tui import fuzzy_filter


MAX_ISSUES = 100
MAX_SUGGESTIONS = 20

_ISSUE_TOKEN_RE = re.compile(r"(?:^|[ \t])#([^\s#]*)$")
_SSH_REMOTE_RE = re.compile(r"^git@github\.com:([^/]+/[^/]+?)(?:\.git)?$")
_HTTPS_REMOTE_RE = re.compile(r"^https?://github\.com/([^/]+/[^/]+?)(?:\.git)?$")


def extract_issue_token(text_before_cursor: str) -> str | None:
    match = _ISSUE_TOKEN_RE.search(text_before_cursor)
    return match.group(1) if match is not None else None


def parse_github_repo(remote_url: str) -> str | None:
    for pattern in (_SSH_REMOTE_RE, _HTTPS_REMOTE_RE):
        match = pattern.match(remote_url)
        if match is not None:
            return match.group(1)
    return None


async def resolve_github_repo(pi, cwd: str) -> tuple[str | None, str | None]:
    """Returns (repo, None) on success, (None, error message) otherwise."""
    result = await pi.exec("git", ["remote", "-v"], cwd=cwd, timeout=5_000)
    if result.code != 0:
        return None, "github-issue-autocomplete: cwd is not a git repository"

    for line in result.stdout.split("\n"):
        columns = line.strip().split()
        if len(columns) < 2:
            continue
        repo = parse_github_repo(columns[1])
        if repo is not None:
            return repo, None

    return None, "github-issue-autocomplete: cwd is not a GitHub repository"


def format_issue_item(issue: dict) -> dict:
    return {
        "value": f"#{issue['number']}",
        "label": f"#{issue['number']}",
        "description": f"[{issue['state'].lower()}] {issue['title']}",
    }


def filter_issues(issues: list[dict], query: str) -> list[dict]:
    if not query.strip():
        return [format_issue_item(issue) for issue in issues[:MAX_SUGGESTIONS]]

    if query.isdigit():
        numeric_matches = [issue for issue in issues if str(issue["number"]).startswith(query)]
        if numeric_matches:
            return [format_issue_item(issue) for issue in numeric_matches[:MAX_SUGGESTIONS]]

    matches = fuzzy_filter(issues, query, lambda issue: f"{issue['number']} {issue['title']}")
    return [format_issue_item(issue) for issue in matches[:MAX_SUGGESTIONS]]


class IssueAutocompleteProvider:
    """Wraps the built-in provider: `#` tokens complete against open GitHub
    issues, everything else falls through to the wrapped provider."""

    def __init__(self, current, get_issues):
        self._current = current
        self._get_issues = get_issues

    async def get_suggestions(self, lines, cursor_line, cursor_col, options):
        current_line = lines[cursor_line] if cursor_line < len(lines) else ""
        token = extract_issue_token(current_line[:cursor_col])
        if token is None:
            return await self._current.get_suggestions(lines, cursor_line, cursor_col, options)

        issues = await self._get_issues()
        if options["signal"].cancelled or not issues:
            return await self._current.get_suggestions(lines, cursor_line, cursor_col, options)

        suggestions = filter_issues(issues, token)
        if not suggestions:
            return await self._current.get_suggestions(lines, cursor_line, cursor_col, options)

        return {"items": suggestions, "prefix": f"#{token}"}

    def apply_completion(self, lines, cursor_line, cursor_col, item, prefix):
        return self._current.apply_completion(lines, cursor_line, cursor_col, item, prefix)

    def should_trigger_file_completion(self, lines, cursor_line, cursor_col):
        should = getattr(self._current, "should_trigger_file_completion", None)
        return should(lines, cursor_line, cursor_col) if should is not None else True


def extension(pi):
    async def on_session_start(_event, ctx) -> None:
        repo, error = await resolve_github_repo(pi, ctx.cwd)
        if repo is None:
            ctx.ui.notify(error, "error")
            return

        # pi memoizes the load promise; here a single load task sets an Event
        # every caller waits on, so the `gh` call runs at most once.
        state = {"issues": None, "error_shown": False}
        loaded = tonio.Event()

        async def load_issues() -> None:
            result = await pi.exec(
                "gh",
                [
                    "issue",
                    "list",
                    "--repo",
                    repo,
                    "--state",
                    "open",
                    "--limit",
                    str(MAX_ISSUES),
                    "--json",
                    "number,title,state",
                ],
                cwd=ctx.cwd,
                timeout=5_000,
            )
            if result.code != 0:
                if not state["error_shown"]:
                    state["error_shown"] = True
                    details = result.stderr.strip() or f"exit code {result.code}"
                    ctx.ui.notify(f"github-issue-autocomplete: failed to load issues: {details}", "error")
            else:
                try:
                    state["issues"] = json.loads(result.stdout)
                except ValueError:
                    if not state["error_shown"]:
                        state["error_shown"] = True
                        ctx.ui.notify("github-issue-autocomplete: failed to parse gh issue list output", "error")
            loaded.set()

        async def get_issues() -> list[dict] | None:
            await loaded.wait()
            return state["issues"]

        tonio.spawn.without_tracking(load_issues())
        ctx.ui.add_autocomplete_provider(lambda current: IssueAutocompleteProvider(current, get_issues))

    pi.on("session_start", on_session_start)
