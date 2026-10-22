"""Port of pi's TypeSafe System One adapter (packages/ai/src/api/typesafe-system-one.ts).

TypeSafe's native System One protocol. OpenRouter, Vercel AI Gateway and
OpenCode Zen serve the same protocol, so those providers use this API with
different base URLs.

TypeSafe takes images as `assets`. They are experimental and need the
`TypeSafe-Danger-Zone-Images` opt-in header; without it the API answers 400.
OpenRouter drops `assets` silently and takes images as `image_url` parts in the
state instead. Models behind a proxy that hides both the provider id and the
OpenRouter host must set `compat.image_input`.
"""

from collections.abc import Awaitable
from typing import Any

from pidrei_ai.api.classifier_shared import is_record
from pidrei_ai.api.system_one_shared import ResolvedSystemOneCompat, SystemOneTransport, classify_system_one
from pidrei_ai.types import ClassifierContext, ClassifierModel, ClassifierOptions, ClassifierResult


def _output(body: Any) -> dict[str, Any]:
    if not is_record(body):
        raise RuntimeError("System One API returned an unexpected response")
    return body


def _detect_compat(model: ClassifierModel) -> ResolvedSystemOneCompat:
    # OpenRouter by provider id, which survives base URL overrides, or by host, which covers regional endpoints.
    if model.provider == "openrouter" or "openrouter.ai" in model.base_url:
        return ResolvedSystemOneCompat(image_input="state-image-urls", send_image_opt_in_header=False)
    return ResolvedSystemOneCompat(image_input="assets", send_image_opt_in_header=True)


_TRANSPORT = SystemOneTransport(
    api="typesafe-system-one",
    label="System One API",
    detect_compat=_detect_compat,
    url=lambda model: f"{model.base_url.rstrip('/')}/systemone",
    payload=lambda model, request: {"model": model.id, **request},
    output=_output,
)


def classify(
    model: ClassifierModel, context: ClassifierContext, options: ClassifierOptions | None = None
) -> Awaitable[ClassifierResult]:
    """TypeSafe System One classification with public `bool` values mapped to wire-level `noul`."""
    return classify_system_one(_TRANSPORT, model, context, options)
