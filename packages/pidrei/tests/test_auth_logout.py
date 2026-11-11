"""Mirror of pi coding-agent test/auth-logout.test.ts.

pi uses the built-in `radius` provider as a provider whose display name
differs from its id; Radius is not a PiDrei provider, so the mirror uses
`github-copilot` ("GitHub Copilot"). pi's `vi.spyOn(modelRuntime, "logout")`
is an instance attribute set on the runtime.
"""

import pytest

from pidrei.cli.auth_logout import AuthLogoutOptions, run_auth_logout
from pidrei.core.auth_storage import AuthStorage
from pidrei.core.model_runtime import CredentialSynchronizationError, ModelRuntime
from pidrei_ai.auth.types import ApiKeyCredential, Credential, OAuthCredential
from pidrei_ai.models_store import InMemoryModelsStore
from pidrei_utils import clock
from pidrei_utils.cancel import CancelToken


class Output:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.errors: list[str] = []

    def write(self, line: str) -> None:
        self.lines.append(line)

    def error(self, line: str) -> None:
        self.errors.append(line)


async def setup(data: dict[str, Credential]):
    credentials = AuthStorage.in_memory(data)
    model_runtime = await ModelRuntime(
        credentials=credentials,
        models_path=None,
        models_store=InMemoryModelsStore(),
        allow_model_network=False,
    )
    output = Output()

    def run(args: list[str]):
        return run_auth_logout(
            args, AuthLogoutOptions(model_runtime=model_runtime, terminal=output, cancel=CancelToken())
        )

    return credentials, model_runtime, output, run


def oauth_credential() -> OAuthCredential:
    return OAuthCredential(access="access", refresh="refresh", expires=clock.now_ms() + 3_600_000)


@pytest.mark.tonio
async def test_requires_exactly_one_provider():
    _, _, output, run = await setup({})
    for args in ([], ["--api-key"], ["github-copilot", "extra"]):
        output.errors.clear()
        assert await run(args) == 1
        assert output.errors == ["Usage: pidrei auth logout <provider>"]


@pytest.mark.tonio
async def test_removes_a_stored_account_login():
    credentials, _, output, run = await setup({"github-copilot": oauth_credential()})
    assert await run(["github-copilot"]) == 0
    assert output.lines == ["Logged out of GitHub Copilot."]
    assert await credentials.read("github-copilot") is None


@pytest.mark.tonio
async def test_removes_a_stored_api_key_by_provider_name_and_keeps_other_credentials():
    credentials, _, output, run = await setup(
        {"openai": ApiKeyCredential(key="sk-test"), "github-copilot": oauth_credential()}
    )
    assert await run(["OpenAI"]) == 0
    assert output.lines == [
        "Removed stored API key for OpenAI. Environment variables and models.json config are unchanged."
    ]
    assert await credentials.read("openai") is None
    remaining = await credentials.read("github-copilot")
    assert remaining is not None and remaining.type == "oauth"


@pytest.mark.tonio
async def test_succeeds_when_nothing_is_stored_so_it_can_be_repeated():
    _, _, output, run = await setup({})
    assert await run(["github-copilot"]) == 0
    assert output.lines == [
        "No stored credentials for GitHub Copilot. Environment variables and models.json config are unchanged."
    ]


@pytest.mark.tonio
async def test_removes_credentials_of_providers_that_are_not_loaded_matched_by_id():
    credentials, _, output, run = await setup({"extension-provider": oauth_credential()})
    assert await run(["extension-provider"]) == 0
    assert output.lines == ["Logged out of extension-provider."]
    assert await credentials.read("extension-provider") is None


@pytest.mark.tonio
async def test_reports_credential_removal_failures():
    _, model_runtime, output, run = await setup({"github-copilot": oauth_credential()})

    async def logout(*_args, **_kwargs):
        raise Exception("storage failed")

    model_runtime.logout = logout
    assert await run(["github-copilot"]) == 1
    assert output.errors == ["Logout failed: storage failed"]


@pytest.mark.tonio
async def test_reports_committed_credential_removal_separately_from_a_failed_logout():
    _, model_runtime, output, run = await setup({"github-copilot": oauth_credential()})

    async def logout(*_args, **_kwargs):
        raise CredentialSynchronizationError("github-copilot", "logout", None, cause=Exception("snapshot failed"))

    model_runtime.logout = logout
    assert await run(["github-copilot"]) == 1
    assert (
        "Credentials removed for GitHub Copilot, but local model state could not be synchronized:" in output.errors[0]
    )


@pytest.mark.tonio
async def test_rejects_an_unknown_provider_and_lists_the_stored_credentials():
    _, _, output, run = await setup({"github-copilot": oauth_credential()})
    assert await run(["nope"]) == 1
    assert output.errors == ['Unknown provider "nope".', "Stored credentials: github-copilot"]
