"""Mirror of pi server src/index.ts (public re-exports).

pi's radius presence exports are absent: `radius.ts` integrates with
`radius.pi.dev`, a pi-specific service, and Phase 7 step 1 dropped it.
"""

from .config import (
    VERSION,
    get_auth_path,
    get_instances_path,
    get_machine_path,
    get_server_dir,
    get_socket_path,
)
from .handler import handle_ipc_request, open_rpc_stream, to_instance_summary
from .ipc.client import send_ipc_request
from .ipc.protocol import encode_message, parse_request_line, parse_response_line
from .ipc.server import IpcRequestHandler, IpcServer, start_ipc_server
from .rpc_process import RpcProcessInstance, RpcProcessOptions, create_rpc_process_instance
from .serve import serve
from .storage import (
    delete_machine,
    get_instance,
    load_instances,
    load_machine,
    remove_instance,
    save_instances,
    save_machine,
    upsert_instance,
)
from .supervisor import ServerSupervisor, supervisor
from .types import InstanceRecord, InstanceStatus, MachineRecord, timestamp


__all__ = [
    "VERSION",
    "InstanceRecord",
    "InstanceStatus",
    "IpcRequestHandler",
    "IpcServer",
    "MachineRecord",
    "RpcProcessInstance",
    "RpcProcessOptions",
    "ServerSupervisor",
    "create_rpc_process_instance",
    "delete_machine",
    "encode_message",
    "get_auth_path",
    "get_instance",
    "get_instances_path",
    "get_machine_path",
    "get_server_dir",
    "get_socket_path",
    "handle_ipc_request",
    "load_instances",
    "load_machine",
    "open_rpc_stream",
    "parse_request_line",
    "parse_response_line",
    "remove_instance",
    "save_instances",
    "save_machine",
    "send_ipc_request",
    "serve",
    "start_ipc_server",
    "supervisor",
    "timestamp",
    "to_instance_summary",
    "upsert_instance",
]
