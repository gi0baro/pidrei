"""pidrei-only: the registry step behind `pi.register_mcp_server()` once an
extension is loaded. pi checks the owner and registers as two statements on its
one thread; here extensions register in parallel, so it is one step."""

from pidrei.core.mcp_servers import McpServerRegistry, RegisteredMcpServer


def test_claim_leaves_a_name_another_extension_owns():
    registry = McpServerRegistry()
    changes = []
    registry.set_change_listener(lambda: changes.append(1))
    first = RegisteredMcpServer(name="docs", config={"command": "one"}, extension_path="/ext/a.py")

    assert registry.claim(first) is None
    assert registry.claim(RegisteredMcpServer(name="docs", config={"command": "two"}, extension_path="/ext/b.py")) == (
        first
    )
    assert registry.get("docs") == first
    assert len(changes) == 1

    again = RegisteredMcpServer(name="docs", config={"command": "three"}, extension_path="/ext/a.py")
    assert registry.claim(again) is None
    assert registry.get("docs") == again
    assert len(changes) == 2
