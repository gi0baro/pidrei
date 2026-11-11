"""Mirror of pi mcp test/naming.test.ts."""

import re

from pidrei_mcp import create_mcp_tool_name, mcp_namespace


def test_joins_server_and_tool_and_replaces_characters_outside_a_z_0_9_underscore():
    assert create_mcp_tool_name("docs", "search") == "mcp__docs__search"
    assert create_mcp_tool_name("my-server", "get.item/v2") == "mcp__my_server__get_item_v2"


def test_shortens_names_over_64_characters_with_a_hash_of_server_and_tool():
    long = create_mcp_tool_name("server", "x" * 100)
    assert len(long) == 64
    assert re.fullmatch(r"mcp__server__x+_[0-9a-f]{8}", long)
    # Names that share their first 55 characters still differ.
    assert create_mcp_tool_name("server", f"{'x' * 100}y") != long
    # The hash covers the unsanitized names, and is stable.
    assert create_mcp_tool_name("server", "x" * 100) == long
    assert create_mcp_tool_name("ser-ver", "x" * 100) != create_mcp_tool_name("ser_ver", "x" * 100)


def test_adds_the_hash_suffix_when_the_sanitized_name_is_taken():
    taken = create_mcp_tool_name("s", "a_b")
    assert taken == "mcp__s__a_b"
    second = create_mcp_tool_name("s", "a-b", lambda name: name == taken)
    assert re.fullmatch(r"mcp__s__a_b_[0-9a-f]{8}", second)
    assert create_mcp_tool_name("s", "a_b", lambda name: name == taken) != second


def test_mcp_namespace_is_mcp_server_with_dashes_replaced_the_prefix_of_the_servers_tool_names():
    assert mcp_namespace("docs") == "mcp__docs"
    assert mcp_namespace("my-server") == "mcp__my_server"
    assert mcp_namespace("my-server") == mcp_namespace("my_server")
    assert create_mcp_tool_name("my-server", "search").startswith(f"{mcp_namespace('my-server')}__")
