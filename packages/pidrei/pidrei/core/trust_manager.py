"""Mirror of pi coding-agent src/core/trust-manager.ts."""

import json
import os
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any

import tonio.colored as tonio
from tonio.colored import fs

from ..config import CONFIG_DIR_NAME
from ..utils.lockfile import FileLock
from ..utils.paths import canonicalize_path_blocking, resolve_path
from ..utils.text import strip_bom


type ProjectTrustDecision = bool | None


@dataclass(slots=True)
class ProjectTrustStoreEntry:
    path: str
    decision: bool


@dataclass(slots=True)
class ProjectTrustUpdate:
    path: str
    decision: ProjectTrustDecision


@dataclass(slots=True)
class ProjectTrustOption:
    label: str
    trusted: bool
    updates: list[ProjectTrustUpdate]
    saved_path: str | None = field(default=None)


TRUST_REQUIRING_PROJECT_CONFIG_RESOURCES = (
    "settings.json",
    "mcp.json",
    "extensions",
    "skills",
    "prompts",
    "themes",
    "SYSTEM.md",
    "APPEND_SYSTEM.md",
)


def _normalize_cwd_blocking(cwd: str) -> str:
    return canonicalize_path_blocking(resolve_path(cwd))


def _find_nearest_trust_entry_blocking(data: dict[str, Any], cwd: str) -> ProjectTrustStoreEntry | None:
    current_dir = _normalize_cwd_blocking(cwd)
    while True:
        value = data.get(current_dir)
        if value is True or value is False:
            return ProjectTrustStoreEntry(current_dir, value)

        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            return None
        current_dir = parent_dir


def get_project_trust_parent_path_blocking(cwd: str) -> str | None:
    trust_path = _normalize_cwd_blocking(cwd)
    parent_dir = os.path.dirname(trust_path)
    return None if parent_dir == trust_path else parent_dir


def get_project_trust_options_blocking(cwd: str, *, include_session_only: bool = False) -> list[ProjectTrustOption]:
    trust_path = _normalize_cwd_blocking(cwd)
    trust_options = [
        ProjectTrustOption("Trust", True, [ProjectTrustUpdate(trust_path, True)], saved_path=trust_path),
    ]
    parent_path = get_project_trust_parent_path_blocking(cwd)
    if parent_path is not None:
        trust_options.append(
            ProjectTrustOption(
                f"Trust parent folder ({parent_path})",
                True,
                [ProjectTrustUpdate(parent_path, True), ProjectTrustUpdate(trust_path, None)],
                saved_path=parent_path,
            )
        )
    if include_session_only:
        trust_options.append(ProjectTrustOption("Trust (this session only)", True, []))
    trust_options.append(
        ProjectTrustOption("Do not trust", False, [ProjectTrustUpdate(trust_path, False)], saved_path=trust_path)
    )
    if include_session_only:
        trust_options.append(ProjectTrustOption("Do not trust (this session only)", False, []))
    return trust_options


def _read_trust_file_blocking(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}

    try:
        with open(path, encoding="utf-8") as f:
            parsed = json.loads(strip_bom(f.read()))
    except Exception as error:
        raise Exception(f"Failed to read trust store {path}: {error}")

    if not isinstance(parsed, dict):
        raise Exception(f"Invalid trust store {path}: expected an object")  # noqa: TRY004

    data: dict[str, Any] = {}
    for key, value in parsed.items():
        if value is not True and value is not False and value is not None:
            raise Exception(f"Invalid trust store {path}: value for {json.dumps(key)} must be true, false, or null")
        data[key] = value
    return data


def _write_trust_file_blocking(path: str, data: dict[str, Any]) -> None:
    sorted_data: dict[str, Any] = {}
    for key in sorted(data.keys()):
        value = data[key]
        if value is True or value is False or value is None:
            sorted_data[key] = value
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{json.dumps(sorted_data, indent=2)}\n")


async def _with_trust_file_lock(path: str, fn: Any) -> Any:
    trust_dir = os.path.dirname(path)
    await fs.Path(trust_dir).mkdir(parents=True, exist_ok=True)
    async with FileLock(trust_dir, lockfile_path=f"{path}.lock"):
        return await tonio.spawn_blocking(fn)


def has_trust_requiring_project_resources_blocking(cwd: str) -> bool:
    """Returns True when cwd has project-local resources that must be gated by
    project trust: trust-requiring entries under cwd/.pidrei, or .agents/skills
    in cwd or one of its ancestors. Returns False when no such project resources
    exist. The user/global ~/.agents/skills directory is always treated as a
    trusted user resource and is ignored here, even when cwd is $HOME.
    """
    home_dir = canonicalize_path_blocking(resolve_path(os.environ.get("HOME") or os.path.expanduser("~")))
    user_agents_skills_dir = os.path.join(home_dir, ".agents", "skills")
    current_dir = canonicalize_path_blocking(resolve_path(cwd))

    config_dir = os.path.join(current_dir, CONFIG_DIR_NAME)
    if any(os.path.exists(os.path.join(config_dir, entry)) for entry in TRUST_REQUIRING_PROJECT_CONFIG_RESOURCES):
        return True

    while True:
        agents_skills_dir = os.path.join(current_dir, ".agents", "skills")
        if agents_skills_dir != user_agents_skills_dir and os.path.exists(agents_skills_dir):
            return True

        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            return False
        current_dir = parent_dir


class ProjectTrustStore:
    """pi's store is fully synchronous (`get`/`set`/`setMany` all block) —
    `trust-manager.ts` has no write queue and no promises at all.

    So do we, behaviourally: every operation completes before it returns, and a
    write error reaches the caller. What changes is only how the waiting is
    spelled: the file lock is awaited (`FileLock`), and the read-modify-write
    under it is one blocking unit handed to the pool and `await`ed, rather than
    run on a runtime worker.

    An earlier port queued writes through the same `Event` chain as
    `SettingsManager`. That was a mis-applied analogy: pi genuinely defers in
    settings (`settings-manager.ts:286`) and genuinely does not here. The queue
    made `set()` return before the decision was durable and made write failures
    unreachable — including by `_maybe_save_implicit_project_trust_after_reload`,
    whose `except` clause was dead code as a result.
    """

    def __init__(self, agent_dir: str):
        self._trust_path = os.path.join(resolve_path(agent_dir), "trust.json")

    async def get(self, cwd: str) -> ProjectTrustDecision:
        entry = await self.get_entry(cwd)
        return entry.decision if entry is not None else None

    def get_entry(self, cwd: str) -> Awaitable[ProjectTrustStoreEntry | None]:
        def read_blocking() -> ProjectTrustStoreEntry | None:
            data = _read_trust_file_blocking(self._trust_path)
            return _find_nearest_trust_entry_blocking(data, cwd)

        return _with_trust_file_lock(self._trust_path, read_blocking)

    def set(self, cwd: str, decision: ProjectTrustDecision) -> Awaitable[None]:
        return self.set_many([ProjectTrustUpdate(cwd, decision)])

    def set_many(self, decisions: list[ProjectTrustUpdate]) -> Awaitable[None]:
        def write_blocking() -> None:
            data = _read_trust_file_blocking(self._trust_path)
            for update in decisions:
                key = _normalize_cwd_blocking(update.path)
                if update.decision is None:
                    data.pop(key, None)
                else:
                    data[key] = update.decision
            _write_trust_file_blocking(self._trust_path, data)

        return _with_trust_file_lock(self._trust_path, write_blocking)
