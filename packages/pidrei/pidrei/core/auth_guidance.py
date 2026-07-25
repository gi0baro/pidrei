"""Mirror of pi coding-agent src/core/auth-guidance.ts."""

import os

from ..config import get_docs_path


UNKNOWN_PROVIDER = "unknown"


def get_provider_login_help() -> str:
    return "\n".join(
        [
            "Use /login to log into a provider via OAuth or API key. See:",
            f"  {os.path.join(get_docs_path(), 'providers.md')}",
            f"  {os.path.join(get_docs_path(), 'models.md')}",
        ]
    )


def format_no_models_available_message() -> str:
    return f"No models available. {get_provider_login_help()}"


def format_no_model_selected_message() -> str:
    return f"No model selected.\n\n{get_provider_login_help()}\n\nThen use /model to select a model."


def format_no_api_key_found_message(provider: str) -> str:
    provider_display = "the selected model" if provider == UNKNOWN_PROVIDER else provider
    return f"No API key found for {provider_display}.\n\n{get_provider_login_help()}"
