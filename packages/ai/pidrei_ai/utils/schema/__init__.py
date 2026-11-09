"""pi-schema's model-visible rules over `jsonschema` (packages/schema/src/).

pi validates tool arguments with pi-schema, its own JSON Schema compiler, and
pi-schema owns the text the model sees when arguments are rejected and how
loosely typed arguments are coerced. Those rules are ported here: the walk
that explains a failure (explain.py), the coercion (coerce.py), the
validators (validator.py) and their helpers (checks.py, types.py).
`jsonschema` is the engine underneath (engine.py, formats.py). pi's builders
are typing sugar for plain JSON Schema and are not ported: schemas are dicts.
"""

from pidrei_ai.utils.schema.formats import FORMATS
from pidrei_ai.utils.schema.types import ParseResult, ValidationError
from pidrei_ai.utils.schema.validator import CoercingValidator, Validator, compile


__all__ = ["FORMATS", "CoercingValidator", "ParseResult", "ValidationError", "Validator", "compile"]
