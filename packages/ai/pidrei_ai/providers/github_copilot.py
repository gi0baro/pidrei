"""Port of pi's github-copilot provider factory (packages/ai/src/providers/github-copilot.ts).

`api` dispatches on `model.api`: anthropic-messages, openai-completions,
openai-responses. `filter_models` narrows the vendored catalog to what the
authenticated account's model picker actually offers.
"""

from pidrei_ai.api.anthropic_messages_lazy import anthropic_messages_api
from pidrei_ai.api.openai_completions_lazy import openai_completions_api
from pidrei_ai.api.openai_responses_lazy import openai_responses_api
from pidrei_ai.auth.helpers import env_api_key_auth, lazy_oauth
from pidrei_ai.auth.oauth.load import load_github_copilot_oauth
from pidrei_ai.auth.types import Credential, OAuthCredential, ProviderAuth
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import Provider, create_provider
from pidrei_ai.types import Model


def _filter_models(models: list[Model], credential: Credential | None) -> list[Model]:
    if not isinstance(credential, OAuthCredential):
        return models
    available_model_ids = credential.extra.get("availableModelIds")
    if not isinstance(available_model_ids, list) or not all(isinstance(id, str) for id in available_model_ids):
        return models
    available = set(available_model_ids)
    return [model for model in models if model.id in available]


def github_copilot_provider() -> Provider:
    return create_provider(
        id="github-copilot",
        name="GitHub Copilot",
        base_url="https://api.individual.githubcopilot.com",
        auth=ProviderAuth(
            api_key=env_api_key_auth("GitHub Copilot token", ["COPILOT_GITHUB_TOKEN"]),
            oauth=lazy_oauth(name="GitHub Copilot", load=load_github_copilot_oauth),
        ),
        models=list(MODELS.get("github-copilot", [])),
        filter_models=_filter_models,
        api={
            "anthropic-messages": anthropic_messages_api(),
            "openai-completions": openai_completions_api(),
            "openai-responses": openai_responses_api(),
        },
    )
