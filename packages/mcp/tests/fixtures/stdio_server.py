"""Mirror of pi mcp test/fixtures/stdio-server.mjs: a newline-delimited MCP
server answering initialize, tools/list, tools/call and ping."""

import json
import sys


sys.stderr.write("stdio fixture ready\n")
sys.stderr.flush()

for line in sys.stdin:
    message = json.loads(line)
    if "id" not in message:
        continue
    method = message.get("method")
    if method == "initialize":
        result = {
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "stdio-fixture", "version": "1.0.0"},
        }
    elif method == "tools/list":
        result = {"tools": [{"name": "echo", "inputSchema": {"type": "object"}}]}
    elif method == "tools/call":
        result = {"content": [{"type": "text", "text": str(message["params"]["arguments"]["text"])}]}
    elif method == "ping":
        result = {}
    else:
        reply = {"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32601, "message": "not found"}}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()
        continue
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}) + "\n")
    sys.stdout.flush()
