"""Port of pi's Cloudflare Workers AI System One adapter
(packages/ai/src/api/cloudflare-workers-ai-system-one.ts).

System One models on the Workers AI REST endpoint:
`POST /accounts/{account}/ai/run` with `{model, input}`. The REST API wraps the
model output in Cloudflare's API envelope. Third-party models such as
`typesafe/jev` add a run record:
`{success, result: {state: "Completed", result: {answers, usage}}}`.
https://developers.cloudflare.com/ai/models/typesafe/jev/
Cloudflare-hosted models such as `@cf/cloudflare/clef` return the output directly:
`{success, result: {model, answers, usage}}`.
https://developers.cloudflare.com/workers-ai/models/clef/
"""

import json
from collections.abc import Awaitable
from typing import Any

from pidrei_ai.api.system_one_shared import SystemOneTransport, classify_system_one, is_record
from pidrei_ai.types import ClassifierContext, ClassifierModel, ClassifierOptions, ClassifierResult


_LABEL = "Cloudflare Workers AI"


def _cloudflare_error_message(errors: Any) -> str:
    if isinstance(errors, list):
        messages = [error["message"] for error in errors if is_record(error) and isinstance(error.get("message"), str)]
        if messages:
            return f"{_LABEL} error: {'; '.join(messages)}"
    return f"{_LABEL} request failed"


def _output(body: Any) -> dict[str, Any]:
    if not is_record(body):
        raise RuntimeError(f"{_LABEL} returned an unexpected response")
    if body.get("success") is False:
        raise RuntimeError(_cloudflare_error_message(body.get("errors")))
    result = body.get("result")
    if not is_record(result):
        raise RuntimeError(f"{_LABEL} returned an unexpected response")
    if "answers" in result:
        return result
    if result.get("state") != "Completed":
        raise RuntimeError(f"{_LABEL} run did not complete (state: {_js_string(result, 'state')})")
    if not is_record(result.get("result")):
        raise RuntimeError(f"{_LABEL} returned an unexpected response")
    return result["result"]


def _js_string(record: dict[str, Any], key: str) -> str:
    """JS `String(record[key])` for the JSON values a run state can hold."""
    if key not in record:
        return "undefined"
    value = record[key]
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return value if isinstance(value, str) else json.dumps(value)


_TRANSPORT = SystemOneTransport(
    api="cloudflare-workers-ai-system-one",
    label=_LABEL,
    url=lambda model: f"{model.base_url.rstrip('/')}/run",
    payload=lambda model, request: {"model": model.id, "input": request},
    output=_output,
)


def classify(
    model: ClassifierModel, context: ClassifierContext, options: ClassifierOptions | None = None
) -> Awaitable[ClassifierResult]:
    """Cloudflare Workers AI System One classification with public `bool` values mapped to wire-level `noul`."""
    return classify_system_one(_TRANSPORT, model, context, options)
