"""Mirror of pi coding-agent src/extensions/tool-search/index.ts.

The `tool_search` tool as an extension. The CLI loads it as a built-in
extension; SDK users add `create_tool_search_extension()` to their extension
factories.

`tool_search` is registered inactive. Activate it with `--tools`, the
`defaultTools` setting, or `set_active_tools()`.
"""

from typing import Any

from .tool import ToolSearchToolOptions, create_tool_search_tool_definition


def create_tool_search_extension() -> Any:
    async def extension(pi: Any) -> None:
        definition = create_tool_search_tool_definition(ToolSearchToolOptions(tools=pi))
        definition.default_active = False
        pi.register_tool(definition)

    return extension


extension = create_tool_search_extension()

__all__ = ["create_tool_search_extension", "extension"]
