"""Mirror of pi coding-agent test/config-selector.test.ts.

The resources are pidrei extension modules (`.py`, pi's are `.ts`) from a
`git:` package source (pi's is `npm:`); both are only labels here.
"""

import os

import pytest

from pidrei.core.keybindings import KeybindingsManager
from pidrei.core.package_manager import ResolvedPaths, ResolvedResource
from pidrei.core.settings_manager import SettingsManager
from pidrei.core.source_info import PathMetadata
from pidrei.modes.interactive.components.config_selector import (
    ConfigSelectorComponent,
    build_canonical_path_map_blocking,
)
from pidrei.modes.interactive.theme import init_theme
from pidrei.utils.ansi import strip_ansi
from pidrei_tui import set_keybindings


@pytest.fixture(autouse=True)
async def _setup():
    await init_theme("dark")
    set_keybindings(KeybindingsManager.in_memory())


# config selector search


@pytest.mark.tonio
async def test_keeps_the_selection_when_a_cursor_key_leaves_the_query_unchanged(tmp_path):
    temp_dir = str(tmp_path)
    package_root = os.path.join(temp_dir, "pkg")
    resolved_paths = ResolvedPaths(
        extensions=[
            ResolvedResource(
                path=os.path.join(package_root, "extensions", name),
                enabled=True,
                metadata=PathMetadata(
                    source="git:github.com/example/pidrei-tools", scope="user", origin="package", base_dir=package_root
                ),
            )
            for name in ["tool-a.py", "tool-b.py", "other.py"]
        ],
    )
    both_scopes = {"global": resolved_paths, "project": resolved_paths}
    selector = ConfigSelectorComponent(
        both_scopes,
        SettingsManager.in_memory(project_trusted=True),
        temp_dir,
        os.path.join(temp_dir, "agent"),
        lambda: None,
        lambda: None,
        lambda: None,
        canonical_by_path=build_canonical_path_map_blocking(both_scopes),
    )
    resource_list = selector.get_resource_list()

    def selected_row() -> str | None:
        rows = strip_ansi("\n".join(resource_list.render(120))).split("\n")
        return next((line for line in rows if line.startswith(">") and ".py" in line), None)

    for char in "tool":
        resource_list.handle_input(char)
    resource_list.handle_input("\x1b[B")
    assert "tool-b" in selected_row()
    resource_list.handle_input("\x1b[D")  # Left
    resource_list.handle_input("\x1b[H")  # Home

    assert "tool-b" in selected_row()
