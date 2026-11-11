"""Port of pi's github-copilot provider factory (packages/ai/src/providers/github-copilot.ts).

`api` dispatches on `model.api`: anthropic-messages, openai-completions,
openai-responses. `filter_models` narrows the vendored catalog to what the
authenticated account's model picker actually offers.
"""

from collections.abc import Awaitable
from dataclasses import replace

from pidrei_ai.api.anthropic_messages_lazy import anthropic_messages_api
from pidrei_ai.api.openai_completions_lazy import openai_completions_api
from pidrei_ai.api.openai_responses_lazy import openai_responses_api
from pidrei_ai.api.typesafe_system_one_lazy import typesafe_system_one_api
from pidrei_ai.auth.helpers import env_api_key_auth, lazy_oauth
from pidrei_ai.auth.oauth.load import load_github_copilot_oauth
from pidrei_ai.auth.types import Credential, OAuthCredential, ProviderAuth
from pidrei_ai.models_generated import CLASSIFIER_MODELS, MODELS
from pidrei_ai.registry import Provider, create_provider
from pidrei_ai.types import (
    ClassifierContext,
    ClassifierModel,
    ClassifierOptions,
    ClassifierResult,
    Model,
    ProviderClassifier,
)


class _CopilotSystemOne:
    """Copilot serves System One at `/v1/systemone`, while its chat APIs have no version prefix.
    The login replaces the catalog base URL with the account's endpoint, so the prefix is added per request.
    """

    __slots__ = ("_classifier",)

    def __init__(self, classifier: ProviderClassifier):
        self._classifier = classifier

    def classify(
        self, model: ClassifierModel, context: ClassifierContext, options: ClassifierOptions | None = None
    ) -> Awaitable[ClassifierResult]:
        return self._classifier.classify(replace(model, base_url=f"{model.base_url.rstrip('/')}/v1"), context, options)


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
            oauth=lazy_oauth(name="GitHub Copilot", is_subscription=True, load=load_github_copilot_oauth),
        ),
        models=[*MODELS.get("github-copilot", []), *CLASSIFIER_MODELS.get("github-copilot", [])],
        filter_models=_filter_models,
        api={
            "anthropic-messages": anthropic_messages_api(),
            "openai-completions": openai_completions_api(),
            "openai-responses": openai_responses_api(),
        },
        classifiers={"typesafe-system-one": _CopilotSystemOne(typesafe_system_one_api())},
    )
