"""Mirror of pi codemode src/index.ts: sandboxed Python execution where the only
capability is calling injected tools.

Importing the package primes Monty's runtime (see `runtime/pool.py`): import it
at program start.
"""

from .declarations import (
    BUILTIN_STUBS,
    DEFAULT_INPUT_SCHEMA_MAX_CHARS,
    MCP_PYTHON_PREAMBLE,
    CodemodeParameter,
    RenderedTool,
    RenderedType,
    input_parameters,
    mcp_structured_content_schema,
    render_stubs,
    render_tools,
    reserved_type_names,
    schema_to_type,
)
from .identifier import to_codemode_identifier
from .runtime.host import CodemodeSandbox
from .runtime.pool import CodemodePool
from .runtime.prelude import MAX_OUTPUT_CHARS, MAX_OUTPUT_ITEMS, MAX_STORE_TOTAL_CHARS, MAX_STORE_VALUE_CHARS
from .source import (
    CODEMODE_OPTIONS_PREFIX,
    CODEMODE_SOURCE_GRAMMAR,
    CodemodeSourceError,
    CodemodeSourceOptions,
    ParsedCodemodeSource,
    parse_codemode_source,
)
from .types import (
    CodemodeCall,
    CodemodeCallStatus,
    CodemodeError,
    CodemodeErrorKind,
    CodemodeGlobal,
    CodemodeImageItem,
    CodemodeJsonSchema,
    CodemodeOutputItem,
    CodemodeResult,
    CodemodeStoreWrites,
    CodemodeTextItem,
    CodemodeTool,
)


__all__ = [
    "BUILTIN_STUBS",
    "CODEMODE_OPTIONS_PREFIX",
    "CODEMODE_SOURCE_GRAMMAR",
    "DEFAULT_INPUT_SCHEMA_MAX_CHARS",
    "MAX_OUTPUT_CHARS",
    "MAX_OUTPUT_ITEMS",
    "MAX_STORE_TOTAL_CHARS",
    "MAX_STORE_VALUE_CHARS",
    "MCP_PYTHON_PREAMBLE",
    "CodemodeCall",
    "CodemodeCallStatus",
    "CodemodeError",
    "CodemodeErrorKind",
    "CodemodeGlobal",
    "CodemodeImageItem",
    "CodemodeJsonSchema",
    "CodemodeOutputItem",
    "CodemodeParameter",
    "CodemodePool",
    "CodemodeResult",
    "CodemodeSandbox",
    "CodemodeSourceError",
    "CodemodeSourceOptions",
    "CodemodeStoreWrites",
    "CodemodeTextItem",
    "CodemodeTool",
    "ParsedCodemodeSource",
    "RenderedTool",
    "RenderedType",
    "input_parameters",
    "mcp_structured_content_schema",
    "parse_codemode_source",
    "render_stubs",
    "render_tools",
    "reserved_type_names",
    "schema_to_type",
    "to_codemode_identifier",
]
