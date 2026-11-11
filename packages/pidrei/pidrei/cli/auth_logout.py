"""Port of pi's cli/auth-logout.ts: `pidrei auth logout <provider>`."""

from dataclasses import dataclass

from pidrei_ai.auth.types import AuthOperationOptions
from pidrei_tui import replace_control_characters
from pidrei_utils.cancel import CancelToken

from ..config import APP_NAME
from ..core.model_runtime import CredentialSynchronizationError, ModelRuntime
from ..core.provider_matching import find_matching_providers
from .auth_login import provider_names
from .auth_terminal import AuthOutput


@dataclass(slots=True)
class AuthLogoutOptions:
    """Dependencies of `run_auth_logout`, injected so tests can capture the output."""

    model_runtime: ModelRuntime
    terminal: AuthOutput
    cancel: CancelToken


AUTH_LOGOUT_USAGE = f"{APP_NAME} auth logout <provider>"


async def run_auth_logout(args: list[str], options: AuthLogoutOptions) -> int:
    """Run `pidrei auth logout <provider>`: remove the credential `pidrei auth login` or /login stored, and return
    the exit code. Environment variables and models.json config stay, as with /logout."""
    model_runtime, terminal, cancel = options.model_runtime, options.terminal, options.cancel
    if len(args) != 1 or args[0].startswith("-"):
        terminal.error(f"Usage: {AUTH_LOGOUT_USAGE}")
        return 1
    provider_ref = args[0]

    try:
        stored = await model_runtime.list_credentials(AuthOperationOptions(cancel=cancel))
    except Exception as error:
        terminal.error(f"Could not read stored credentials: {error}")
        return 1
    matches = find_matching_providers(model_runtime.get_providers(), provider_ref, provider_names)
    provider = matches[0] if matches else None
    # Providers registered by extensions are not loaded here, so their stored credentials match by id alone.
    wanted_id = provider.id if provider is not None else provider_ref
    credential = next((candidate for candidate in stored if candidate.provider_id == wanted_id), None)
    if provider is None and credential is None:
        terminal.error(f'Unknown provider "{replace_control_characters(provider_ref)}".')
        stored_ids = sorted(candidate.provider_id for candidate in stored)
        terminal.error(f"Stored credentials: {', '.join(stored_ids) or 'none'}")
        return 1
    provider_name = provider.name if provider is not None else provider_ref
    if credential is None:
        terminal.write(
            f"No stored credentials for {provider_name}. Environment variables and models.json config are unchanged."
        )
        return 0

    try:
        await model_runtime.logout(credential.provider_id, AuthOperationOptions(cancel=cancel))
    except Exception as error:
        message = str(error)
        if isinstance(error, CredentialSynchronizationError):
            terminal.error(
                f"Credentials removed for {provider_name}, but local model state could not be synchronized: {message}"
            )
        else:
            terminal.error(f"Logout failed: {message}")
        return 1
    terminal.write(
        f"Logged out of {provider_name}."
        if credential.type == "oauth"
        else f"Removed stored API key for {provider_name}. Environment variables and models.json config are unchanged."
    )
    return 0
