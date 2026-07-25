"""Mirror of pi coding-agent src/config.ts — paths and app-config sections.

Not ported (Node-ecosystem machinery with no pidrei equivalent):
- Bun binary / install-method detection (npm/pnpm/yarn/bun) and the
  self-update command builders. pidrei is distributed as a Python package;
  a uv/pip-appropriate update story can be added later if needed.
- package.json `piConfig` renaming indirection: the app identity is fixed.
"""

import importlib.metadata
import os
from pathlib import Path

from .utils.paths import normalize_path


PACKAGE_NAME = "pidrei"
APP_NAME = "pidrei"
APP_TITLE = "pidrei"
CONFIG_DIR_NAME = ".pidrei"

try:
    VERSION = importlib.metadata.version("pidrei")
except importlib.metadata.PackageNotFoundError:
    VERSION = "0.0.0"

# e.g., PIDREI_CODING_AGENT_DIR
ENV_AGENT_DIR = f"{APP_NAME.upper()}_CODING_AGENT_DIR"
ENV_SESSION_DIR = f"{APP_NAME.upper()}_CODING_AGENT_SESSION_DIR"


def expand_tilde_path(path: str) -> str:
    return normalize_path(path)


def get_package_dir() -> str:
    """Base directory for resolving package assets shipped with pidrei."""
    env_dir = os.environ.get("PIDREI_PACKAGE_DIR")
    if env_dir:
        return normalize_path(env_dir)
    return str(Path(__file__).resolve().parent)


def get_docs_path() -> str:
    return os.path.join(get_package_dir(), "docs")


# =============================================================================
# User Config Paths (~/.pidrei/agent/*)
# =============================================================================


def get_agent_dir() -> str:
    """Get the agent config directory (e.g., ~/.pidrei/agent/)."""
    env_dir = os.environ.get(ENV_AGENT_DIR)
    if env_dir:
        return expand_tilde_path(env_dir)
    return os.path.join(os.path.expanduser("~"), CONFIG_DIR_NAME, "agent")


def get_custom_themes_dir() -> str:
    return os.path.join(get_agent_dir(), "themes")


def get_models_path() -> str:
    return os.path.join(get_agent_dir(), "models.json")


def get_auth_path() -> str:
    return os.path.join(get_agent_dir(), "auth.json")


def get_settings_path() -> str:
    return os.path.join(get_agent_dir(), "settings.json")


def get_tools_dir() -> str:
    return os.path.join(get_agent_dir(), "tools")


def get_bin_dir() -> str:
    """Get path to managed binaries directory (fd, rg)."""
    return os.path.join(get_agent_dir(), "bin")


def get_prompts_dir() -> str:
    return os.path.join(get_agent_dir(), "prompts")


def get_sessions_dir() -> str:
    return os.path.join(get_agent_dir(), "sessions")


def get_debug_log_path() -> str:
    return os.path.join(get_agent_dir(), f"{APP_NAME}-debug.log")
