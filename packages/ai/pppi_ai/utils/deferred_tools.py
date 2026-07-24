"""Port of pi's deferred-tools split (packages/ai/src/utils/deferred-tools.ts)."""

from collections.abc import Callable

from pppi_ai.types import Context, Tool


def split_deferred_tools(
    context: Context,
    enabled: bool,
    normalize_name: Callable[[str], str] = lambda name: name,
) -> tuple[list[Tool], dict[str, Tool]]:
    """Split current tools into prefix (immediate) and transcript-loaded (deferred)."""
    unique_tools: dict[str, Tool] = {}
    for tool in context.tools or []:
        unique_tools[normalize_name(tool.name)] = tool
    if not enabled:
        return list(unique_tools.values()), {}

    deferred_names: set[str] = set()
    used_names: set[str] = set()
    for message in context.messages:
        if message.role == "assistant":
            for block in message.content:
                if block.type == "toolCall":
                    used_names.add(normalize_name(block.name))
        elif message.role == "toolResult":
            for name in message.added_tool_names or []:
                normalized_name = normalize_name(name)
                if normalized_name not in used_names:
                    deferred_names.add(normalized_name)

    immediate: list[Tool] = []
    deferred: dict[str, Tool] = {}
    for name, tool in unique_tools.items():
        if name in deferred_names:
            deferred[name] = tool
        else:
            immediate.append(tool)
    return immediate, deferred
