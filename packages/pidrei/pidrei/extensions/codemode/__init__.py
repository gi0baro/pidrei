"""Mirror of pi coding-agent src/extensions/codemode/index.ts: the `codemode`
tool as an extension. The CLI loads it as a built-in extension; SDK users add
`create_codemode_extension()` to their extension factories and must call
`session.bind_extensions()`, which emits the `session_start` that opens the
extension's pool.

`codemode` is registered inactive. Activate it with `--tools`, the
`defaultTools` setting, or `set_active_tools()`; the MCP extension activates it
when MCP tools are only reachable from scripts.

Scripts run in a Monty pool, one per extension instance: opened by the
`session_start` handler, closed by the `session_shutdown` handler. Every
reload, new session, resume or fork builds new extension instances, so the
reference is written once and never replaced. A script run before the start or
after the shutdown fails as a sandbox error.

pi loads the executor lazily (`execute.lazy.ts`); here it is imported with the
extension, since importing `pidrei_codemode` primes Monty's runtime and has to
happen at program start.
"""

import math
from typing import Any

from pidrei_codemode import CodemodePool

from ...core.settings_manager import CodemodeMode
from . import execute as _execute  # noqa: F401 - imported at program start, see the module docstring
from .tool import (
    CODEMODE_DOCS_PATH,
    CODEMODE_SCHEMA,
    CODEMODE_STORE_ENTRY_TYPE,
    CODEMODE_TOOL_NAME,
    DEFAULT_CODEMODE_INLINE_BUDGET,
    CodemodeNestedCall,
    CodemodeToolDetails,
    CodemodeToolOptions,
    create_codemode_description,
    create_codemode_tool,
    create_codemode_tool_definition,
    is_codemode_tool,
)


__all__ = [
    "CODEMODE_DOCS_PATH",
    "CODEMODE_SCHEMA",
    "CODEMODE_STORE_ENTRY_TYPE",
    "CODEMODE_TOOL_NAME",
    "DEFAULT_CODEMODE_INLINE_BUDGET",
    "CodemodeNestedCall",
    "CodemodeToolDetails",
    "CodemodeToolOptions",
    "create_codemode_description",
    "create_codemode_extension",
    "create_codemode_tool",
    "create_codemode_tool_definition",
    "extension",
    "is_codemode_tool",
]


def _codemode_settings(pi: Any) -> dict[str, Any]:
    settings = pi.get_settings().get("codemode")
    return settings if isinstance(settings, dict) else {}


def _read_mode(pi: Any) -> CodemodeMode:
    return "only" if _codemode_settings(pi).get("mode") == "only" else "on"


def _read_inline_budget(pi: Any) -> float | None:
    budget = _codemode_settings(pi).get("inlineBudget")
    is_number = isinstance(budget, (int, float)) and not isinstance(budget, bool)
    return budget if is_number and math.isfinite(budget) and budget >= 0 else None


def _read_type_check(pi: Any) -> bool:
    return _codemode_settings(pi).get("typeCheck") is not False


def create_codemode_extension(
    *,
    mode: CodemodeMode | None = None,
    inline_budget: float | None = None,
    models: bool = True,
    type_check: bool | None = None,
) -> Any:
    """The codemode extension factory. `mode`, `inline_budget` and `type_check`
    override the `codemode.mode`, `codemode.inlineBudget` and
    `codemode.typeCheck` settings; `models` exposes the model catalog and
    classifiers to scripts as `models` (default: True)."""

    async def extension(pi: Any) -> None:
        pool: CodemodePool | None = None

        async def on_session_start(_event: dict[str, Any], _ctx: Any) -> None:
            nonlocal pool
            pool = await CodemodePool()

        async def on_session_shutdown(_event: dict[str, Any], _ctx: Any) -> None:
            if pool is not None:
                await pool.close()

        def get_tool_namespace(name: str) -> Any:
            return next((tool.namespace for tool in pi.get_all_tools() if tool.name == name), None)

        pi.on("session_start", on_session_start)
        pi.on("session_shutdown", on_session_shutdown)
        options = CodemodeToolOptions(
            get_pool=lambda: pool,
            get_tool_namespace=get_tool_namespace,
            models=models,
            append_entry=pi.append_entry,
            get_mode=lambda: mode if mode is not None else _read_mode(pi),
            get_inline_budget=lambda: inline_budget if inline_budget is not None else _read_inline_budget(pi),
            get_type_check=lambda: type_check if type_check is not None else _read_type_check(pi),
        )
        definition = create_codemode_tool_definition(options)
        definition.default_active = False
        pi.register_tool(definition)

    return extension


extension = create_codemode_extension()
