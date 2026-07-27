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


# `Path(__file__).resolve()` costs one lstat per path component, and it was the
# single largest source of filesystem calls on a tonio runtime worker (2,536 of
# them across the test suite). `__file__` cannot change during the process, so
# it is resolved once here, at import time — which is outside the never-block
# rule by construction. The env override is deliberately *not* folded in: it is
# a documented user-facing setting (see `cli/args.py`, for Nix/Guix store
# paths), so it stays a live lookup on every call, exactly as before.
_RESOLVED_PACKAGE_DIR = str(Path(__file__).resolve().parent)


def get_package_dir() -> str:
    """Base directory for resolving package assets shipped with pidrei."""
    env_dir = os.environ.get("PIDREI_PACKAGE_DIR")
    if env_dir:
        return normalize_path(env_dir)
    return _RESOLVED_PACKAGE_DIR


def get_docs_path() -> str:
    return os.path.join(get_package_dir(), "docs")


def get_readme_path() -> str:
    return os.path.join(get_package_dir(), "README.md")


def get_examples_path() -> str:
    return os.path.join(get_package_dir(), "examples")


def get_themes_dir() -> str:
    """Get path to built-in themes directory (shipped with the package)."""
    return os.path.join(get_package_dir(), "modes", "interactive", "theme")


def get_interactive_assets_dir() -> str:
    return os.path.join(get_package_dir(), "modes", "interactive", "assets")


def get_bundled_interactive_asset_path(name: str) -> str:
    """Get path to a bundled interactive asset."""
    return os.path.join(get_interactive_assets_dir(), name)


def get_share_viewer_url(gist_id: str) -> str | None:
    """URL of a viewer that renders a shared session gist, if one is configured.

    pi defaults this to its own hosted viewer (`pi.dev/session/`). pidrei ships
    no viewer, and pointing our users at pi's would send them somewhere that
    cannot read our gists — so there is no default and `/share` prints the gist
    URL alone, which is complete on its own. The override stays so anyone
    running a viewer can point at it without a code change.
    """
    base_url = os.environ.get("PIDREI_SHARE_VIEWER_URL")
    return f"{base_url}#{gist_id}" if base_url else None


def get_changelog_path() -> str:
    return os.path.abspath(os.path.join(get_package_dir(), "CHANGELOG.md"))


def get_export_template_dir() -> str:
    """Get path to the HTML export template directory (shipped with the package)."""
    return os.path.join(get_package_dir(), "core", "export_html")


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
