"""Agent discovery and configuration.

Everything here is synchronous filesystem code; the extension runs it through
`tonio.spawn_blocking` so discovery stays off the runtime.
"""

import os
from dataclasses import dataclass

from pidrei.config import CONFIG_DIR_NAME, get_agent_dir
from pidrei.utils.frontmatter import parse_frontmatter


# Scope values: "user" | "project" | "both"


@dataclass(slots=True)
class AgentConfig:
    name: str
    description: str
    system_prompt: str
    source: str  # "user" | "project"
    file_path: str
    tools: list[str] | None = None
    model: str | None = None


@dataclass(slots=True)
class AgentDiscoveryResult:
    agents: list[AgentConfig]
    project_agents_dir: str | None


def parse_tool_list(value) -> list[str] | None:
    """Normalize a frontmatter `tools` value to a list of tool names.

    Both spellings are valid YAML and both are in use:

        tools: read, bash        # string
        tools: [read, bash]      # array

    so accept either. Anything else (a number, a map, a nested list) yields no
    tools rather than raising: this runs inside agent discovery, where a single
    bad file must not take down every other agent in the same directory.
    """
    raw = value if isinstance(value, list) else value.split(",") if isinstance(value, str) else []
    tools = [t.strip() for t in raw if isinstance(t, str) and t.strip()]
    return tools or None


def load_agents_from_dir(dir: str, source: str) -> list[AgentConfig]:
    agents: list[AgentConfig] = []

    if not os.path.exists(dir):
        return agents

    try:
        entries = sorted(os.scandir(dir), key=lambda entry: entry.name)
    except OSError:
        return agents

    for entry in entries:
        if not entry.name.endswith(".md"):
            continue
        file_path = os.path.join(dir, entry.name)
        try:
            if not os.path.isfile(file_path):
                continue  # Broken symlink or directory, skip it
            with open(file_path, encoding="utf-8") as handle:
                content = handle.read()
        except OSError:
            continue

        try:
            frontmatter, body = parse_frontmatter(content)
        except Exception:  # noqa: S112 - one bad file must not hide the rest
            continue

        if not isinstance(frontmatter.get("name"), str) or not isinstance(frontmatter.get("description"), str):
            continue

        agents.append(
            AgentConfig(
                name=frontmatter["name"],
                description=frontmatter["description"],
                tools=parse_tool_list(frontmatter.get("tools")),
                model=frontmatter["model"] if isinstance(frontmatter.get("model"), str) else None,
                system_prompt=body,
                source=source,
                file_path=file_path,
            )
        )

    return agents


def find_nearest_project_agents_dir(cwd: str) -> str | None:
    current_dir = cwd
    while True:
        candidate = os.path.join(current_dir, CONFIG_DIR_NAME, "agents")
        if os.path.isdir(candidate):
            return candidate

        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            return None
        current_dir = parent_dir


def discover_agents(cwd: str, scope: str) -> AgentDiscoveryResult:
    user_dir = os.path.join(get_agent_dir(), "agents")
    project_agents_dir = find_nearest_project_agents_dir(cwd)

    user_agents = [] if scope == "project" else load_agents_from_dir(user_dir, "user")
    project_agents = (
        [] if scope == "user" or project_agents_dir is None else load_agents_from_dir(project_agents_dir, "project")
    )

    # Project agents override user agents with the same name under "both".
    agent_map: dict[str, AgentConfig] = {}
    for agent in [*user_agents, *project_agents]:
        agent_map[agent.name] = agent

    return AgentDiscoveryResult(agents=list(agent_map.values()), project_agents_dir=project_agents_dir)
