"""Mirror of pi tui src/layout-node.ts.

The protocol between layout-aware components (``Stack``, ``ScrollView``) and
the layout engine in ``layout``. A component opts in by exposing a
``LAYOUT_NODE`` method returning its node record.

Port note: pi keys the method with ``Symbol.for()`` so it cannot collide with
component fields; Python has no symbols, so the key is a dunder-shaped
attribute name held in the ``LAYOUT_NODE`` constant — call it through
``get_layout_node`` rather than spelling it out.

Records (camelCase like pi): LayoutViewport = {"width", "height"};
StackLayoutEntry = {"component", "basis"?, "grow"?, "shrink"?, "minSize"?,
"maxSize"?, "visible"?}; StackLayoutNode = {"type": "vstack" | "hstack",
"entries", "gap", "align"}; ScrollLayoutNode = {"type": "scroll",
"component", "state"}.
"""

LAYOUT_NODE = "__pidrei_tui_layout_node__"


def get_layout_node(component) -> dict | None:
    node = getattr(component, LAYOUT_NODE, None)
    return node() if callable(node) else None
