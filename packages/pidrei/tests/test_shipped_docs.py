"""Every path the system prompt names must exist and ship.

pidrei-only. There is no pi suite behind this: pi's docs and README are simply
present in its repo, so nothing there can drift. pidrei's are shipped *inside*
the installed package and reached through `config.get_*_path()`, which makes two
new failure modes possible — a doc the prompt names but nobody wrote, and a file
that exists in the checkout but is left out of the wheel.

The system prompt is model-visible, so a path in it that does not resolve is a
parity defect: it points the agent at a file it cannot read. This suite is the
check that keeps it honest.
"""

import os

import pytest

from pidrei.config import (
    get_docs_path,
    get_examples_path,
    get_package_dir,
    get_readme_path,
)
from pidrei.core.system_prompt import BuildSystemPromptOptions, build_system_prompt


#: Every doc named in the system prompt (`system_prompt.py`), plus providers.md
#: and models.md, which `auth_guidance.py` and `interactive_mode.py` print at
#: users as "read this".
NAMED_DOCS = (
    "extensions.md",
    "themes.md",
    "skills.md",
    "prompt-templates.md",
    "tui.md",
    "keybindings.md",
    "sdk.md",
    "custom-provider.md",
    "models.md",
    "packages.md",
    "environment-variables.md",
    "providers.md",
)


def test_readme_ships():
    assert os.path.isfile(get_readme_path())


def test_docs_directory_ships():
    assert os.path.isdir(get_docs_path())


def test_examples_directory_ships():
    assert os.path.isdir(get_examples_path())


def test_examples_live_inside_the_package():
    """They started outside it, where the wheel would not have carried them."""
    assert get_examples_path().startswith(get_package_dir())


@pytest.mark.parametrize("name", NAMED_DOCS)
def test_named_doc_exists(name):
    path = os.path.join(get_docs_path(), name)
    assert os.path.isfile(path), f"the system prompt names docs/{name}, which does not exist"


@pytest.mark.parametrize("name", NAMED_DOCS)
def test_named_doc_is_not_a_stub(name):
    with open(os.path.join(get_docs_path(), name), encoding="utf-8") as handle:
        content = handle.read()
    assert content.startswith("# "), f"docs/{name} should open with a heading"
    assert len(content.splitlines()) > 20, f"docs/{name} is a stub"


def test_examples_referenced_by_the_docs_exist():
    examples = os.path.join(get_examples_path(), "extensions")
    for name in ("trigger_compact.py", "input_transform_streaming.py", "git_merge_and_resolve.py"):
        assert os.path.isfile(os.path.join(examples, name))
    assert os.path.isfile(os.path.join(examples, "plan_mode", "__init__.py"))


def test_every_path_in_the_system_prompt_resolves():
    """The actual contract: what the prompt tells the model to read is readable."""
    prompt = build_system_prompt(BuildSystemPromptOptions(cwd=os.getcwd()))

    for path in (get_readme_path(), get_docs_path(), get_examples_path()):
        assert path in prompt, f"{path} is no longer named in the system prompt"
        assert os.path.exists(path), f"the system prompt names {path}, which does not exist"
