"""Pure utility functions for plan mode. Extracted for testability."""

import re
from dataclasses import dataclass


# Destructive commands blocked in plan mode.
DESTRUCTIVE_PATTERNS = [
    re.compile(pattern, flags)
    for pattern, flags in (
        (r"\brm\b", re.IGNORECASE),
        (r"\brmdir\b", re.IGNORECASE),
        (r"\bmv\b", re.IGNORECASE),
        (r"\bcp\b", re.IGNORECASE),
        (r"\bmkdir\b", re.IGNORECASE),
        (r"\btouch\b", re.IGNORECASE),
        (r"\bchmod\b", re.IGNORECASE),
        (r"\bchown\b", re.IGNORECASE),
        (r"\bchgrp\b", re.IGNORECASE),
        (r"\bln\b", re.IGNORECASE),
        (r"\btee\b", re.IGNORECASE),
        (r"\btruncate\b", re.IGNORECASE),
        (r"\bdd\b", re.IGNORECASE),
        (r"\bshred\b", re.IGNORECASE),
        (r"(^|[^<])>(?!>)", 0),
        (r">>", 0),
        (r"\bnpm\s+(install|uninstall|update|ci|link|publish)", re.IGNORECASE),
        (r"\byarn\s+(add|remove|install|publish)", re.IGNORECASE),
        (r"\bpnpm\s+(add|remove|install|publish)", re.IGNORECASE),
        (r"\bpip\s+(install|uninstall)", re.IGNORECASE),
        (r"\bapt(-get)?\s+(install|remove|purge|update|upgrade)", re.IGNORECASE),
        (r"\bbrew\s+(install|uninstall|upgrade)", re.IGNORECASE),
        (
            (
                r"\bgit\s+(add|commit|push|pull|merge|rebase|reset|checkout|branch\s+-[dD]"
                r"|stash|cherry-pick|revert|tag|init|clone)"
            ),
            re.IGNORECASE,
        ),
        (r"\bsudo\b", re.IGNORECASE),
        (r"\bsu\b", re.IGNORECASE),
        (r"\bkill\b", re.IGNORECASE),
        (r"\bpkill\b", re.IGNORECASE),
        (r"\bkillall\b", re.IGNORECASE),
        (r"\breboot\b", re.IGNORECASE),
        (r"\bshutdown\b", re.IGNORECASE),
        (r"\bsystemctl\s+(start|stop|restart|enable|disable)", re.IGNORECASE),
        (r"\bservice\s+\S+\s+(start|stop|restart)", re.IGNORECASE),
        (r"\b(vim?|nano|emacs|code|subl)\b", re.IGNORECASE),
    )
]

# Safe read-only commands allowed in plan mode.
SAFE_PATTERNS = [
    re.compile(pattern, flags)
    for pattern, flags in (
        (r"^\s*cat\b", 0),
        (r"^\s*head\b", 0),
        (r"^\s*tail\b", 0),
        (r"^\s*less\b", 0),
        (r"^\s*more\b", 0),
        (r"^\s*grep\b", 0),
        (r"^\s*find\b", 0),
        (r"^\s*ls\b", 0),
        (r"^\s*pwd\b", 0),
        (r"^\s*echo\b", 0),
        (r"^\s*printf\b", 0),
        (r"^\s*wc\b", 0),
        (r"^\s*sort\b", 0),
        (r"^\s*uniq\b", 0),
        (r"^\s*diff\b", 0),
        (r"^\s*file\b", 0),
        (r"^\s*stat\b", 0),
        (r"^\s*du\b", 0),
        (r"^\s*df\b", 0),
        (r"^\s*tree\b", 0),
        (r"^\s*which\b", 0),
        (r"^\s*whereis\b", 0),
        (r"^\s*type\b", 0),
        (r"^\s*env\b", 0),
        (r"^\s*printenv\b", 0),
        (r"^\s*uname\b", 0),
        (r"^\s*whoami\b", 0),
        (r"^\s*id\b", 0),
        (r"^\s*date\b", 0),
        (r"^\s*cal\b", 0),
        (r"^\s*uptime\b", 0),
        (r"^\s*ps\b", 0),
        (r"^\s*top\b", 0),
        (r"^\s*htop\b", 0),
        (r"^\s*free\b", 0),
        (r"^\s*git\s+(status|log|diff|show|branch|remote|config\s+--get)", re.IGNORECASE),
        (r"^\s*git\s+ls-", re.IGNORECASE),
        (r"^\s*npm\s+(list|ls|view|info|search|outdated|audit)", re.IGNORECASE),
        (r"^\s*yarn\s+(list|info|why|audit)", re.IGNORECASE),
        (r"^\s*node\s+--version", re.IGNORECASE),
        (r"^\s*python\s+--version", re.IGNORECASE),
        (r"^\s*curl\s", re.IGNORECASE),
        (r"^\s*wget\s+-O\s*-", re.IGNORECASE),
        (r"^\s*jq\b", 0),
        (r"^\s*sed\s+-n", re.IGNORECASE),
        (r"^\s*awk\b", 0),
        (r"^\s*rg\b", 0),
        (r"^\s*fd\b", 0),
        (r"^\s*bat\b", 0),
        (r"^\s*eza\b", 0),
    )
]


def is_safe_command(command: str) -> bool:
    is_destructive = any(pattern.search(command) for pattern in DESTRUCTIVE_PATTERNS)
    is_safe = any(pattern.search(command) for pattern in SAFE_PATTERNS)
    return not is_destructive and is_safe


@dataclass(slots=True)
class TodoItem:
    step: int
    text: str
    completed: bool = False


_EMPHASIS = re.compile(r"\*{1,2}([^*]+)\*{1,2}")
_CODE = re.compile(r"`([^`]+)`")
_LEADING_VERB = re.compile(
    r"^(Use|Run|Execute|Create|Write|Read|Check|Verify|Update|Modify|Add|Remove|Delete|Install)\s+(the\s+)?",
    re.IGNORECASE,
)
_WHITESPACE = re.compile(r"\s+")
_PLAN_HEADER = re.compile(r"\*{0,2}Plan:\*{0,2}\s*\n", re.IGNORECASE)
_NUMBERED = re.compile(r"^\s*(\d+)[.)]\s+\*{0,2}([^*\n]+)", re.MULTILINE)
_TRAILING_EMPHASIS = re.compile(r"\*{1,2}$")
_DONE = re.compile(r"\[DONE:(\d+)\]", re.IGNORECASE)


def clean_step_text(text: str) -> str:
    cleaned = _EMPHASIS.sub(r"\1", text)
    cleaned = _CODE.sub(r"\1", cleaned)
    cleaned = _LEADING_VERB.sub("", cleaned)
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()

    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    if len(cleaned) > 50:
        cleaned = f"{cleaned[:47]}..."
    return cleaned


def extract_todo_items(message: str) -> list[TodoItem]:
    items: list[TodoItem] = []
    header = _PLAN_HEADER.search(message)
    if header is None:
        return items

    plan_section = message[message.index(header.group(0)) + len(header.group(0)) :]

    for match in _NUMBERED.finditer(plan_section):
        text = _TRAILING_EMPHASIS.sub("", match.group(2).strip()).strip()
        if len(text) > 5 and not text.startswith(("`", "/", "-")):
            cleaned = clean_step_text(text)
            if len(cleaned) > 3:
                items.append(TodoItem(step=len(items) + 1, text=cleaned))
    return items


def extract_done_steps(message: str) -> list[int]:
    return [int(match.group(1)) for match in _DONE.finditer(message)]


def mark_completed_steps(text: str, items: list[TodoItem]) -> int:
    done_steps = extract_done_steps(text)
    for step in done_steps:
        item = next((candidate for candidate in items if candidate.step == step), None)
        if item is not None:
            item.completed = True
    return len(done_steps)
