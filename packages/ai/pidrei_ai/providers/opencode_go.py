"""Port of pi's opencode-go provider factory (packages/ai/src/providers/opencode-go.ts).

`api` dispatches on `model.api`: anthropic-messages, openai-completions, openai-responses.
"""

from pidrei_ai.api.anthropic_messages_lazy import anthropic_messages_api
from pidrei_ai.api.openai_completions_lazy import openai_completions_api
from pidrei_ai.api.openai_responses_lazy import openai_responses_api
from pidrei_ai.auth.helpers import env_api_key_auth
from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_generated import MODELS
from pidrei_ai.registry import Provider, create_provider


def opencode_go_provider() -> Provider:
    return create_provider(
        id="opencode-go",
        name="OpenCode Zen Go",
        auth=ProviderAuth(api_key=env_api_key_auth("OpenCode API key", ["OPENCODE_API_KEY"])),
        models=list(MODELS.get("opencode-go", [])),
        api={
            "anthropic-messages": anthropic_messages_api(),
            "openai-completions": openai_completions_api(),
            "openai-responses": openai_responses_api(),
        },
    )
