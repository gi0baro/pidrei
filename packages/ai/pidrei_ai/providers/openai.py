"""Port of pi's openai provider factory (packages/ai/src/providers/openai.ts)."""

from pidrei_ai.api.openai_decisions_lazy import openai_decisions_api
from pidrei_ai.api.openai_responses_lazy import openai_responses_api
from pidrei_ai.auth.helpers import env_api_key_auth, lazy_oauth
from pidrei_ai.auth.oauth.load import load_openai_chatgpt_oauth
from pidrei_ai.auth.types import Credential, ProviderAuth
from pidrei_ai.models_generated import CLASSIFIER_MODELS, MODELS
from pidrei_ai.registry import Provider, create_provider
from pidrei_ai.types import AnyModel
from pidrei_ai.utils.model_operations import is_model_type


def _filter_all_models(models: list[AnyModel], credential: Credential | None) -> list[AnyModel]:
    # Sign in with ChatGPT tokens only reach the Responses API; the Decisions API rejects them.
    if credential is not None and credential.type == "oauth":
        return [model for model in models if not is_model_type(model, "classifier")]
    return models


def openai_provider() -> Provider:
    return create_provider(
        id="openai",
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        auth=ProviderAuth(
            api_key=env_api_key_auth("OpenAI API key", ["OPENAI_API_KEY"]),
            oauth=lazy_oauth(
                name="OpenAI (ChatGPT subscription)",
                is_subscription=True,
                login_label="Sign in with ChatGPT",
                load=load_openai_chatgpt_oauth,
            ),
        ),
        models=[*MODELS.get("openai", []), *CLASSIFIER_MODELS.get("openai", [])],
        filter_all_models=_filter_all_models,
        api=openai_responses_api(),
        classifiers={"openai-decisions": openai_decisions_api()},
    )
