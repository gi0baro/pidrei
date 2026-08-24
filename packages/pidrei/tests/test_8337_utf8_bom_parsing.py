"""Mirror of pi coding-agent test/suite/regressions/8337-utf8-bom-parsing.test.ts."""

import json

import pytest

from pidrei.core.settings_manager import SettingsManager
from pidrei.utils.frontmatter import parse_frontmatter
from pidrei.utils.text import split_bom


BOM = "﻿"


@pytest.mark.tonio
async def test_loads_frontmatter_and_settings_with_a_leading_bom(tmp_dir):
    assert split_bom(f"{BOM}content") == (BOM, "content")
    document = "---\nname: demo\ndescription: Test\n---\nBody"
    parsed = parse_frontmatter(f"{BOM}{document}")
    assert parsed.frontmatter == {"name": "demo", "description": "Test"}
    assert parsed.body == "Body"

    agent_dir = tmp_dir / "agent"
    project_dir = tmp_dir / "project"
    (project_dir / ".pidrei").mkdir(parents=True)
    agent_dir.mkdir(parents=True)
    global_settings_path = agent_dir / "settings.json"
    global_settings_path.write_text(BOM + json.dumps({"defaultModel": "global-model"}), encoding="utf-8")
    (project_dir / ".pidrei" / "settings.json").write_text(
        BOM + json.dumps({"defaultProvider": "project-provider"}), encoding="utf-8"
    )

    settings = await SettingsManager.create(str(project_dir), str(agent_dir))
    assert settings.get_default_model() == "global-model"
    assert settings.get_default_provider() == "project-provider"

    settings.set_theme("dark")
    await settings.flush()
    assert not global_settings_path.read_text(encoding="utf-8").startswith(BOM)
