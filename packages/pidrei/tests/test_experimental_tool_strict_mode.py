"""Mirror of pi coding-agent test/experimental-tool-strict-mode.test.ts."""

import os

from pidrei.core.tools import (
    create_bash_tool_definition,
    create_edit_tool_definition,
    create_read_tool_definition,
    create_write_tool_definition,
)
from pidrei_ai.types import JsonSchemaConstrainedSampling


def _create_built_in_tools():
    cwd = os.getcwd()
    return [
        create_read_tool_definition(cwd),
        create_bash_tool_definition(cwd),
        create_edit_tool_definition(cwd),
        create_write_tool_definition(cwd),
    ]


def test_only_enables_strict_prefer_sampling_in_experimental_mode(monkeypatch):
    monkeypatch.delenv("PIDREI_EXPERIMENTAL", raising=False)
    normal_tools = _create_built_in_tools()
    monkeypatch.setenv("PIDREI_EXPERIMENTAL", "1")
    experimental_tools = _create_built_in_tools()

    for normal, experimental in zip(normal_tools, experimental_tools, strict=True):
        assert experimental.constrained_sampling == JsonSchemaConstrainedSampling(strict="prefer")
        assert experimental.parameters == normal.parameters
        assert normal.constrained_sampling is None
