"""JSON wire conversion for session events and RPC payloads.

pi serializes events and RPC response data with a bare JSON.stringify: JS
objects already carry camelCase keys and drop undefined values. pidrei's
equivalents are snake_case dataclasses, so this module is the port's
stand-in: it converts arbitrary event/payload values to pi's wire shape.

Rules (matching JSON.stringify over pi's runtime values):
- messages (anything with a `role`) and Usage ride the session serde codec,
  so their wire shape is byte-identical to pi's session files;
- Model instances use the models.json wire shape;
- other dataclasses become camelCase dicts with None fields dropped (None
  maps to pi's `undefined`, which JSON.stringify omits);
- dicts pass through with keys unchanged (they are already wire-shaped;
  explicit None values are pi's `null` and survive);
- lists/tuples recurse; scalars pass through.
"""

import dataclasses
from typing import Any

from pidrei_agent.harness.session.serde import serialize_message, serialize_usage
from pidrei_ai.types import Model, Usage


_SPECIAL_TO_CAMEL = {
    "cache_write_1h": "cacheWrite1h",
}


def _camel(name: str) -> str:
    special = _SPECIAL_TO_CAMEL.get(name)
    if special is not None:
        return special
    parts = name.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


def to_wire(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {key: to_wire(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_wire(item) for item in value]
    if isinstance(value, Model):
        from .model_wire import model_to_dict

        return model_to_dict(value)
    if isinstance(value, Usage):
        return serialize_usage(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        if getattr(value, "role", None) is not None:
            return serialize_message(value)
        wire: dict[str, Any] = {}
        for field in dataclasses.fields(value):
            item = getattr(value, field.name)
            if item is None:
                continue
            wire[_camel(field.name)] = to_wire(item)
        return wire
    return value
