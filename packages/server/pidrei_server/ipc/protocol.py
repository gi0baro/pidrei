"""Mirror of pi server src/ipc/protocol.ts.

Newline-framed JSON over the unix socket. pi declares the request/response
unions as TypeScript types over plain JSON objects; the wire format is
identical here, so messages stay plain camelCase dicts and this module
documents the protocol.

Requests (client -> server, one JSON object per line):
  spawn {cwd, label?, provider?, model?}, list, stop {instanceId},
  status {instanceId}, rpc {instanceId, command}, rpc_stream {instanceId}.

Responses (server -> client; every response carries ok: bool and error?):
  spawn_result {instance?}, list_result {instances?}, stop_result
  {instanceId?}, status_result {instance?}, rpc_result {response},
  rpc_ready {instance?}, error {ok: false, error}.

InstanceSummary: {id, status, cwd, label?, sessionId?, sessionFile?}.
(pi also carries radiusPiId; the radius integration was dropped.)

After rpc_ready on an rpc_stream connection, the socket carries RpcCommand
or extension_ui_response lines client -> server, and RpcResponse /
AgentSessionEvent / extension_ui_request / error lines server -> client.
"""

import json
from typing import Any


def encode_message(message: dict[str, Any]) -> str:
    return json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"


def parse_request_line(line: str) -> dict[str, Any]:
    return json.loads(line)


def parse_response_line(line: str) -> dict[str, Any]:
    return json.loads(line)
