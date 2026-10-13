"""Port of pi's openrouter provider factory (packages/ai/src/providers/openrouter.ts)."""

import json
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from pidrei_ai.api.anthropic_messages_lazy import anthropic_messages_api
from pidrei_ai.api.openai_completions_lazy import openai_completions_api
from pidrei_ai.api.openrouter_images_lazy import openrouter_images_api
from pidrei_ai.api.typesafe_system_one_lazy import typesafe_system_one_api
from pidrei_ai.auth.helpers import env_api_key_auth, lazy_oauth
from pidrei_ai.auth.oauth.load import load_openrouter_oauth
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import CLASSIFIER_MODELS, IMAGE_MODELS, MODELS, parse_cost
from pidrei_ai.registry import Provider, RefreshModelsContext, create_provider
from pidrei_ai.types import AnyModel
from pidrei_ai.utils.model_operations import is_model_type
from pidrei_http import http
from pidrei_utils.cancel import run_cancellable

from .openrouter_api import OpenRouterModel, openrouter_cost


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_USER_MODELS_TIMEOUT_MS = 15_000


async def _get_user_models(url: str, key: str, context: RefreshModelsContext) -> tuple[int, bytes]:
    async def _send() -> tuple[int, bytes]:
        request = http.build_request(
            "GET",
            url,
            headers={"accept": "application/json", "authorization": f"Bearer {key}"},
            timeout=http.oneshot_timeout(OPENROUTER_USER_MODELS_TIMEOUT_MS),
        )
        response = await http.default_fetch(request)
        try:
            body = await response.read()
        except BaseException:
            http.abandon_response(response)
            raise
        return response.status_code, body

    # The refresh's token unwinds the request wherever it is parked.
    return await run_cancellable(_send(), context.cancel)


async def personalize_openrouter_catalog(
    catalog: Sequence[AnyModel], context: RefreshModelsContext
) -> Sequence[AnyModel]:
    """Restrict an OpenRouter catalog to the chat models the key may use, with the key's limits and
    prices from `GET /models/user`. Image and classifier models are returned unchanged. A missing
    limit keeps the catalog value. The request uses the provider's configured base URL, so regional
    endpoints such as `https://us.openrouter.ai/api/v1` apply their region's filtering. When the
    endpoint rejects the key (401/403) or does not exist (404), for example with a proxy configured
    in models.json, the catalog is returned unchanged."""
    credential = context.credential
    if credential is None:
        key = None
    elif credential.type == "oauth":
        key = credential.access
    else:
        key = credential.key
    if not key:
        return catalog
    base_url = (context.base_url if context.base_url is not None else OPENROUTER_BASE_URL).rstrip("/")
    status, body = await _get_user_models(f"{base_url}/models/user", key, context)
    if status in (401, 403, 404):
        return catalog
    if not 200 <= status < 300:
        raise RuntimeError(f"OpenRouter model list request failed: {status}")
    data: Any = json.loads(body)
    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        raise RuntimeError("Invalid OpenRouter model list response")  # noqa: TRY004 (pi throws a plain Error)
    allowed: dict[str, OpenRouterModel] = {
        entry["id"]: entry for entry in data["data"] if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    }
    personalized: list[AnyModel] = []
    for model in catalog:
        if not is_model_type(model, "chat"):
            personalized.append(model)
            continue
        entry = allowed.get(model.id)
        if entry is None:
            continue
        top_provider = entry.get("top_provider") or {}
        pricing = entry.get("pricing")
        personalized.append(
            replace(
                model,
                # `or`: OpenRouter reports a missing limit as absent, null or 0, like the generator's fallback.
                context_window=top_provider.get("context_length")
                or entry.get("context_length")
                or model.context_window,
                max_tokens=top_provider.get("max_completion_tokens") or model.max_tokens,
                # pi: `entry.pricing ? ...`; an empty object is truthy there.
                cost=parse_cost(openrouter_cost(pricing)) if pricing is not None else model.cost,
            )
        )
    return personalized


def openrouter_provider() -> Provider:
    return create_provider(
        id="openrouter",
        name="OpenRouter",
        base_url=OPENROUTER_BASE_URL,
        auth=ProviderAuth(
            api_key=env_api_key_auth("OpenRouter API key", ["OPENROUTER_API_KEY"]),
            oauth=lazy_oauth(
                name="OpenRouter OAuth", load=load_openrouter_oauth, login_label="Sign in with OpenRouter"
            ),
        ),
        models=[
            *MODELS.get("openrouter", []),
            *IMAGE_MODELS.get("openrouter", []),
            *CLASSIFIER_MODELS.get("openrouter", []),
        ],
        api={
            "anthropic-messages": anthropic_messages_api(),
            "openai-completions": openai_completions_api(),
        },
        images={"openrouter-images": openrouter_images_api()},
        # OpenRouter serves TypeSafe's System One protocol at /api/v1/systemone.
        classifiers={"typesafe-system-one": typesafe_system_one_api()},
    )
