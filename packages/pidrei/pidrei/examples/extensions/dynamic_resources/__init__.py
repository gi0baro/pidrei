"""Dynamic Resources

Contributes a skill, a prompt template, and a theme from the extension's own
directory via the `resources_discover` event, instead of installing them under
`~/.pidrei`. The paths are re-collected on startup and on `/reload`.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/dynamic_resources
"""

import os


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def extension(pi):
    async def on_resources_discover(_event, _ctx):
        return {
            "skillPaths": [os.path.join(_BASE_DIR, "SKILL.md")],
            "promptPaths": [os.path.join(_BASE_DIR, "dynamic.md")],
            "themePaths": [os.path.join(_BASE_DIR, "dynamic.json")],
        }

    pi.on("resources_discover", on_resources_discover)
