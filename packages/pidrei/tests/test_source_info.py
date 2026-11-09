"""Mirror of pi coding-agent test/source-info.test.ts.

pi's case simulates win32 to check that file path casing folds; PiDrei is
POSIX-only, so only its synthetic-identifier half is mirrored.
"""

from pidrei.core.source_info import get_resource_path_key_blocking


def test_keeps_synthetic_identifiers_unchanged():
    assert get_resource_path_key_blocking("builtin:MCP") == "builtin:MCP"
    assert get_resource_path_key_blocking("builtin:MCP") != get_resource_path_key_blocking("builtin:mcp")
    assert get_resource_path_key_blocking("<inline:Name>") == "<inline:Name>"
