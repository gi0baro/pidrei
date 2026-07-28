"""Port of pi's OpenRouter images adapter (packages/ai/src/api/openrouter-images.ts).

pi uses the `openai` SDK's chat-completions client against OpenRouter's base URL
with `modalities: ["image", "text"]`. Here the one POST goes over the punkreq
seam directly — there is no streaming and no SDK surface worth mirroring beyond
the request itself.
"""

import json
import re
import time
from typing import Any

import tonio.colored as tonio

from pidrei_ai.types import (
    AssistantImages,
    ImageContent,
    ImagesContext,
    ImagesModel,
    ImagesOptions,
    ProviderHeaders,
    ProviderResponse,
    TextContent,
    Usage,
    UsageCost,
)
from pidrei_ai.utils import http
from pidrei_ai.utils.callbacks import maybe_call
from pidrei_ai.utils.cancel import AbortError, CancelToken
from pidrei_ai.utils.error_body import format_provider_error, normalize_provider_error
from pidrei_ai.utils.headers import provider_headers_to_record
from pidrei_ai.utils.provider_retry import retry_provider_request
from pidrei_ai.utils.sanitize_unicode import sanitize_surrogates


_DATA_URL = re.compile(r"^data:([^;]+);base64,(.+)$", re.DOTALL)
_CANCELLED = object()


class OpenRouterImagesError(Exception):
    def __init__(self, status: int, message: str, body: str | None = None):
        super().__init__(message)
        self.status = status
        self.body = body


class _OpenRouterImagesClient:
    """The `openai` SDK's `chat.completions.create`, for this one non-streaming call."""

    def __init__(self, base_url: str, headers: dict[str, str], env=None):
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._headers = headers
        self._env = env

    async def create(
        self, params: dict[str, Any], *, timeout_ms: float | None = None, cancel: CancelToken | None = None
    ):
        # pi's SDK client rejects on an already-aborted signal before sending, and
        # aborts an in-flight request; without this the cancelled request would
        # still reach the provider.
        if cancel is not None and cancel.cancelled:
            raise AbortError("Request aborted")

        client = http.client_for(self._url, self._env)

        async def _send():
            return await client.post(
                self._url, json=params, headers=self._headers, timeout=http.request_timeout(timeout_ms)
            )

        if cancel is None:
            response = await _send()
        else:

            async def _aborted():
                await cancel.wait()
                return _CANCELLED

            response = await tonio.select(_send(), _aborted())
            if response is _CANCELLED:
                raise AbortError("Request aborted")

        body = (await response.read()).decode("utf-8", "replace")
        if not 200 <= response.status_code < 300:
            raise OpenRouterImagesError(response.status_code, f"{response.status_code} {body}", body=body)
        return json.loads(body), ProviderResponse(status=response.status_code, headers=dict(response.headers))


async def generate_images(
    model: ImagesModel, context: ImagesContext, options: ImagesOptions | None = None
) -> AssistantImages:
    output = AssistantImages(
        api=model.api,
        provider=model.provider,
        model=model.id,
        output=[],
        stop_reason="stop",
        timestamp=int(time.time() * 1000),
    )

    try:
        api_key = options.api_key if options else None
        if not api_key:
            raise RuntimeError(f"No API key for provider: {model.provider}")
        client = create_client(model, api_key, options.headers if options else None, options.env if options else None)
        params = build_params(model, context)
        next_params = await maybe_call(options.on_payload if options else None, params, model)
        if next_params is not None:
            params = next_params

        async def _request():
            return await client.create(
                params,
                timeout_ms=options.timeout_ms if options else None,
                cancel=options.cancel if options else None,
            )

        response, raw_response = await retry_provider_request(
            _request,
            max_retries=(options.max_retries if options and options.max_retries is not None else 0),
            max_retry_delay_ms=options.max_retry_delay_ms if options else None,
            cancel=options.cancel if options else None,
        )
        await maybe_call(options.on_response if options else None, raw_response, model)

        output.response_id = response.get("id")
        if response.get("usage"):
            output.usage = parse_usage(response["usage"], model)

        choices = response.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str) and content:
                output.output.append(TextContent(text=content))

            for image in message.get("images") or []:
                raw_url = image.get("image_url")
                image_url = raw_url if isinstance(raw_url, str) else (raw_url or {}).get("url")
                if not image_url or not image_url.startswith("data:"):
                    continue
                match = _DATA_URL.match(image_url)
                if not match:
                    continue
                output.output.append(ImageContent(mime_type=match.group(1), data=match.group(2)))

        return output
    except Exception as error:
        cancelled = options is not None and options.cancel is not None and options.cancel.cancelled
        output.stop_reason = "aborted" if cancelled else "error"
        output.error_message = format_provider_error(normalize_provider_error(error))
        return output


def create_client(
    model: ImagesModel, api_key: str, options_headers: ProviderHeaders | None = None, env=None
) -> _OpenRouterImagesClient:
    headers = provider_headers_to_record({**(model.headers or {}), **(options_headers or {})}) or {}
    headers["authorization"] = f"Bearer {api_key}"
    return _OpenRouterImagesClient(model.base_url, headers, env)


def build_params(model: ImagesModel, context: ImagesContext) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for item in context.input:
        if item.type == "text":
            content.append({"type": "text", "text": sanitize_surrogates(item.text)})
        else:
            content.append({"type": "image_url", "image_url": {"url": f"data:{item.mime_type};base64,{item.data}"}})

    return {
        "model": model.id,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "modalities": ["image", "text"] if "text" in model.output else ["image"],
    }


def parse_usage(raw_usage: dict[str, Any], model: ImagesModel) -> Usage:
    prompt_tokens = raw_usage.get("prompt_tokens") or 0
    details = raw_usage.get("prompt_tokens_details") or {}
    reported_cached_tokens = details.get("cached_tokens") or 0
    cache_write_tokens = details.get("cache_write_tokens") or 0
    cache_read_tokens = (
        max(0, reported_cached_tokens - cache_write_tokens) if cache_write_tokens > 0 else reported_cached_tokens
    )
    input_tokens = max(0, prompt_tokens - cache_read_tokens - cache_write_tokens)
    output_tokens = raw_usage.get("completion_tokens") or 0

    cost = UsageCost(
        input=(model.cost.input / 1_000_000) * input_tokens,
        output=(model.cost.output / 1_000_000) * output_tokens,
        cache_read=(model.cost.cache_read / 1_000_000) * cache_read_tokens,
        cache_write=(model.cost.cache_write / 1_000_000) * cache_write_tokens,
    )
    cost.total = cost.input + cost.output + cost.cache_read + cost.cache_write

    return Usage(
        input=input_tokens,
        output=output_tokens,
        cache_read=cache_read_tokens,
        cache_write=cache_write_tokens,
        total_tokens=input_tokens + output_tokens + cache_read_tokens + cache_write_tokens,
        cost=cost,
    )
