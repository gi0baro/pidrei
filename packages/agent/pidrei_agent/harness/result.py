"""Result vocabulary and tagged errors (port of pi `harness/result.ts`).

The `Result`/`Ok`/`Err` shapes are pidrei-wide primitives that already live in
`harness/types.py`; this module re-exports them so consumers mirror upstream's
import site. `TaggedError` maps pi's tagged-error class factory onto plain
exception subclassing: the subclass name is the tag.
"""

from collections.abc import Callable, Mapping
from typing import Any, ClassVar

from .types import Err, Ok, Result, err, ok


class TaggedError(Exception):
    """Base for machine-matchable errors: subclass name is the `_tag`."""

    _tag: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._tag = cls.__name__

    def __init__(self, message: str, **props: Any):
        super().__init__(message)
        self.message = message
        self._props = props
        for key, value in props.items():
            setattr(self, key, value)

    def to_json(self) -> dict[str, Any]:
        return {"_tag": self._tag, "message": self.message, **self._props}


def match_error[TValue](error: TaggedError, matchers: Mapping[str, Callable[[Any], TValue]]) -> TValue:
    return matchers[error._tag](error)


__all__ = ["Err", "Ok", "Result", "TaggedError", "err", "match_error", "ok"]
