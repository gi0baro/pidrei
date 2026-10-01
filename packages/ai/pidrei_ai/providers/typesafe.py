"""Port of pi's typesafe provider factory (packages/ai/src/providers/typesafe.ts)."""

from pidrei_ai.api.typesafe_system_one_lazy import typesafe_system_one_api
from pidrei_ai.auth.helpers import env_api_key_auth
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import CLASSIFIER_MODELS
from pidrei_ai.registry import Provider, create_provider


def typesafe_provider() -> Provider:
    return create_provider(
        id="typesafe",
        name="TypeSafe",
        auth=ProviderAuth(api_key=env_api_key_auth("TypeSafe API key", ["TYPESAFE_API_KEY"])),
        models=list(CLASSIFIER_MODELS.get("typesafe", [])),
        classifiers={"typesafe-system-one": typesafe_system_one_api()},
    )
