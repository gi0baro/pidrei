"""Run modes for the coding agent.

pi's modes/index.ts also exports InteractiveMode; that lands with the
Phase 4 TUI slice.
"""

from .print_mode import PrintModeOptions, run_print_mode
from .rpc.jsonl import JsonlLineDecoder, serialize_json_line
from .rpc.rpc_client import RpcClient, RpcClientOptions
from .rpc.rpc_mode import run_rpc_mode


__all__ = [
    "JsonlLineDecoder",
    "PrintModeOptions",
    "RpcClient",
    "RpcClientOptions",
    "run_print_mode",
    "run_rpc_mode",
    "serialize_json_line",
]
