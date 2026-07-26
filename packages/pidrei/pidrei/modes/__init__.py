"""Run modes for the coding agent (mirror of pi coding-agent src/modes/).

Re-exports are lazy (PEP 562): core/tools renderer modules import
modes.interactive theme/components, and an eager print_mode/rpc import here
would close an import cycle back through core.agent_session (pi's ESM live
bindings tolerate that cycle; Python module init does not).
"""

import importlib


_EXPORTS = {
    "InteractiveMode": ("interactive.interactive_mode", "InteractiveMode"),
    "JsonlLineDecoder": ("rpc.jsonl", "JsonlLineDecoder"),
    "PrintModeOptions": ("print_mode", "PrintModeOptions"),
    "RpcClient": ("rpc.rpc_client", "RpcClient"),
    "RpcClientOptions": ("rpc.rpc_client", "RpcClientOptions"),
    "run_print_mode": ("print_mode", "run_print_mode"),
    "run_rpc_mode": ("rpc.rpc_mode", "run_rpc_mode"),
    "serialize_json_line": ("rpc.jsonl", "serialize_json_line"),
}

__all__ = [
    "InteractiveMode",
    "JsonlLineDecoder",
    "PrintModeOptions",
    "RpcClient",
    "RpcClientOptions",
    "run_print_mode",
    "run_rpc_mode",
    "serialize_json_line",
]


def __getattr__(name: str):
    try:
        module_name, attr = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    module = importlib.import_module(f".{module_name}", __name__)
    return getattr(module, attr)
