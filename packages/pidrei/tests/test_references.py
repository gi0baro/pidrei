"""Mirror of pi coding-agent test/references.test.ts."""

import importlib.util
from pathlib import Path


PACKAGE_DIRECTORY = Path(__file__).parents[1] / "pidrei"


def _load_generate_references():
    """Import the generator script (not an installed module)."""
    path = Path(__file__).parents[1] / "scripts" / "generate_references.py"
    spec = importlib.util.spec_from_file_location("pidrei_scripts_generate_references", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Regression for #10790: models.json fields lost their reference documentation.
def test_matches_the_committed_pages():
    rendered = _load_generate_references().render_references()
    assert list(rendered) == ["docs/models-configuration.md"]
    for relative_path, expected in rendered.items():
        committed = (PACKAGE_DIRECTORY / relative_path).read_text(encoding="utf-8")
        assert committed == expected, f"{relative_path} is stale; run make references"
