"""Port of pi's TypeSafe System One adapter (packages/ai/src/api/typesafe-system-one.ts).

TypeSafe's native System One protocol. OpenRouter, Vercel AI Gateway and
OpenCode Zen serve the same protocol, so those providers use this API with
different base URLs.
"""

from collections.abc import Awaitable
from typing import Any

from pidrei_ai.api.classifier_shared import is_record
from pidrei_ai.api.system_one_shared import SystemOneTransport, classify_system_one
from pidrei_ai.types import ClassifierContext, ClassifierModel, ClassifierOptions, ClassifierResult


def _output(body: Any) -> dict[str, Any]:
    if not is_record(body):
        raise RuntimeError("System One API returned an unexpected response")
    return body


_TRANSPORT = SystemOneTransport(
    api="typesafe-system-one",
    label="System One API",
    url=lambda model: f"{model.base_url.rstrip('/')}/systemone",
    payload=lambda model, request: {"model": model.id, **request},
    output=_output,
)


def classify(
    model: ClassifierModel, context: ClassifierContext, options: ClassifierOptions | None = None
) -> Awaitable[ClassifierResult]:
    """TypeSafe System One classification with public `bool` values mapped to wire-level `noul`."""
    return classify_system_one(_TRANSPORT, model, context, options)
