"""Mirror of pi coding-agent test/tool-system-prompt-contributions.test.ts."""

import pytest

from pidrei.core.tools.bash import BASH_TOOL_SYSTEM_PROMPT_CONTRIBUTION, create_bash_tool_definition
from pidrei.core.tools.edit import EDIT_TOOL_SYSTEM_PROMPT_CONTRIBUTION, create_edit_tool_definition
from pidrei.core.tools.find import FIND_TOOL_SYSTEM_PROMPT_CONTRIBUTION, create_find_tool_definition
from pidrei.core.tools.grep import GREP_TOOL_SYSTEM_PROMPT_CONTRIBUTION, create_grep_tool_definition
from pidrei.core.tools.ls import LS_TOOL_SYSTEM_PROMPT_CONTRIBUTION, create_ls_tool_definition
from pidrei.core.tools.read import READ_TOOL_SYSTEM_PROMPT_CONTRIBUTION, create_read_tool_definition
from pidrei.core.tools.write import WRITE_TOOL_SYSTEM_PROMPT_CONTRIBUTION, create_write_tool_definition


CASES = [
    ("read", READ_TOOL_SYSTEM_PROMPT_CONTRIBUTION, create_read_tool_definition),
    ("bash", BASH_TOOL_SYSTEM_PROMPT_CONTRIBUTION, create_bash_tool_definition),
    ("edit", EDIT_TOOL_SYSTEM_PROMPT_CONTRIBUTION, create_edit_tool_definition),
    ("write", WRITE_TOOL_SYSTEM_PROMPT_CONTRIBUTION, create_write_tool_definition),
    ("grep", GREP_TOOL_SYSTEM_PROMPT_CONTRIBUTION, create_grep_tool_definition),
    ("find", FIND_TOOL_SYSTEM_PROMPT_CONTRIBUTION, create_find_tool_definition),
    ("ls", LS_TOOL_SYSTEM_PROMPT_CONTRIBUTION, create_ls_tool_definition),
]


@pytest.mark.parametrize(("name", "contribution", "create_definition"), CASES, ids=[case[0] for case in CASES])
def test_keeps_tool_definition_aligned_with_its_contribution(name, contribution, create_definition):
    definition = create_definition("/workspace")

    assert definition.prompt_snippet == contribution["snippet"]
    assert (definition.prompt_guidelines or []) == list(contribution["guidelines"])


def test_keeps_bash_session_environment_guidance_conditional():
    definition = create_bash_tool_definition("/workspace", expose_session_environment=False)

    assert definition.prompt_guidelines is None
