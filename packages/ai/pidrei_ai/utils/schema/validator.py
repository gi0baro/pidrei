"""Port of pi-schema's `compile()` and validators (packages/schema/src/validator.ts):
a schema to a `Validator`, whose `check` is the fast boolean path and whose
`errors` explains failures.

pi builds the explainer and the coercer on first use. Here both are built by
`compile()`, so a validator that parallel tool calls share is one value from
the start; each call keeps its own walk state. `compile()` also takes its own
deep copy of the schema, as pi's compiled checks capture the schema when it
compiles: a schema changed afterwards does not change its validator.
"""

import copy
from typing import Any, Literal, overload

from jsonschema import FormatChecker

from pidrei_ai.utils.schema.coerce import Coercer
from pidrei_ai.utils.schema.engine import Engine
from pidrei_ai.utils.schema.explain import Explainer
from pidrei_ai.utils.schema.types import ParseResult, ValidationError


class Validator:
    """A compiled schema."""

    __slots__ = ("_engine", "_explainer")

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._explainer = Explainer(engine)

    def check(self, value: Any) -> bool:
        """Whether `value` is valid."""
        return self._engine.check(self._engine.schema, self._engine.root_scope, value)

    def errors(self, value: Any) -> list[ValidationError]:
        """Why `value` is invalid: one error per failing keyword, `[]` when it is valid."""
        if self.check(value):
            return []
        return self._explainer.errors(value)


class CoercingValidator(Validator):
    """A validator compiled with `coerce=True`: it also coerces loosely typed input towards the schema."""

    __slots__ = ("_coercer",)

    def __init__(self, engine: Engine) -> None:
        super().__init__(engine)
        self._coercer = Coercer(engine)

    def coerce(self, value: Any) -> Any:
        """`value` coerced towards the schema: numeric strings to numbers, `"true"` to `true`, a single value to a
        one-item array, an optional `null` dropped, the first union branch the value converts to. Only the changed
        paths are copied; `value` comes back itself when nothing changes, and is never modified. Values that cannot be
        coerced stay as they are."""
        return self._coercer.coerce(value)

    def parse(self, value: Any) -> ParseResult:
        """Coerces `value`, then checks it: the coerced value when it is valid, else its errors."""
        coerced = self.coerce(value)
        if self.check(coerced):
            return ParseResult(ok=True, value=coerced)
        return ParseResult(ok=False, errors=tuple(self._explainer.errors(coerced)))


@overload
def compile(
    schema: dict | bool,
    *,
    formats: FormatChecker | None = None,
    coerce: Literal[True],
    unresolved_refs: Literal["throw", "reject"] = "throw",
) -> CoercingValidator: ...


@overload
def compile(
    schema: dict | bool,
    *,
    formats: FormatChecker | None = None,
    coerce: Literal[False] = False,
    unresolved_refs: Literal["throw", "reject"] = "throw",
) -> Validator: ...


def compile(
    schema: dict | bool,
    *,
    formats: FormatChecker | None = None,
    coerce: bool = False,
    unresolved_refs: Literal["throw", "reject"] = "throw",
) -> Validator:
    """Compiles `schema` into a validator. Raises `ValueError` if the schema is malformed or, unless
    `unresolved_refs="reject"`, a `$ref` names nothing; with `"reject"` such a reference fails only the values at
    its place, with the error "unresolved reference". Without `formats`, `format` is an annotation."""
    engine = Engine(copy.deepcopy(schema), formats, reject_unresolved=unresolved_refs == "reject")
    return CoercingValidator(engine) if coerce else Validator(engine)
