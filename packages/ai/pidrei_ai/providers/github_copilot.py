"""Port of pi's github-copilot provider factory (packages/ai/src/providers/github-copilot.ts).

`api` dispatches on `model.api`: anthropic-messages, openai-completions,
openai-responses. `filter_models` narrows the vendored catalog to what the
authenticated account's model picker actually offers; `filter_all_models`
does the same for every model type, classifiers by the account's recorded
decision models.
"""

from collections.abc import Awaitable
from dataclasses import replace
from typing import Literal

from pidrei_ai.api.anthropic_messages_lazy import anthropic_messages_api
from pidrei_ai.api.openai_completions_lazy import openai_completions_api
from pidrei_ai.api.openai_decisions_lazy import openai_decisions_api
from pidrei_ai.api.openai_responses_lazy import openai_responses_api
from pidrei_ai.api.typesafe_system_one_lazy import typesafe_system_one_api
from pidrei_ai.auth.helpers import env_api_key_auth, lazy_oauth
from pidrei_ai.auth.oauth.load import load_github_copilot_oauth
from pidrei_ai.auth.types import Credential, OAuthCredential, ProviderAuth
from pidrei_ai.models_generated import CLASSIFIER_MODELS, MODELS
from pidrei_ai.registry import Provider, create_provider
from pidrei_ai.types import (
    AnyModel,
    ClassifierContext,
    ClassifierModel,
    ClassifierOptions,
    ClassifierResult,
    Model,
    ProviderClassifier,
)
from pidrei_ai.utils.model_operations import is_model_type


class _CopilotV1:
    """Copilot serves its classifier APIs under `/v1` (`/v1/systemone`, `/v1/decisions`), while its chat APIs have
    no version prefix. The login replaces the catalog base URL with the account's endpoint, so the prefix is added
    per request.
    """

    __slots__ = ("_classifier",)

    def __init__(self, classifier: ProviderClassifier):
        self._classifier = classifier

    def classify(
        self, model: ClassifierModel, context: ClassifierContext, options: ClassifierOptions | None = None
    ) -> Awaitable[ClassifierResult]:
        return self._classifier.classify(replace(model, base_url=f"{model.base_url.rstrip('/')}/v1"), context, options)


def _account_model_ids(
    credential: Credential | None, key: Literal["availableModelIds", "availableClassifierModelIds"]
) -> set[str] | None:
    """The model IDs the OAuth login recorded from the account's `/models` catalog under `key`, or None when
    the credential has no such list (API key auth, or a login from before the list was recorded).
    """
    if not isinstance(credential, OAuthCredential):
        return None
    # Stored credentials are read from auth.json.
    ids = credential.extra.get(key)
    if not isinstance(ids, list) or not all(isinstance(id, str) for id in ids):
        return None
    return set(ids)


# pi tests `available ? ... : ...` and `!available || ...` on a `Set` or `undefined`, and an empty `Set`
# is truthy there: an account whose recorded list is empty gets no models. Hence `is None`, not truthiness.
def _filter_models(models: list[Model], credential: Credential | None) -> list[Model]:
    available = _account_model_ids(credential, "availableModelIds")
    return [model for model in models if model.id in available] if available is not None else models


# Decision models are absent from the account's picker catalog, so the login records them separately.
def _filter_all_models(models: list[AnyModel], credential: Credential | None) -> list[AnyModel]:
    chat = _account_model_ids(credential, "availableModelIds")
    classifiers = _account_model_ids(credential, "availableClassifierModelIds")

    def listed(model: AnyModel) -> bool:
        available = classifiers if is_model_type(model, "classifier") else chat
        return available is None or model.id in available

    return [model for model in models if listed(model)]


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
        filter_all_models=_filter_all_models,
        api={
            "anthropic-messages": anthropic_messages_api(),
            "openai-completions": openai_completions_api(),
            "openai-responses": openai_responses_api(),
        },
        classifiers={
            "openai-decisions": _CopilotV1(openai_decisions_api()),
            "typesafe-system-one": _CopilotV1(typesafe_system_one_api()),
        },
    )
