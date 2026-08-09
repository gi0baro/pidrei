"""Mirror of pi's suite/regressions/6768-copilot-compaction-base-url.test.ts."""

import dataclasses

import pytest

from pidrei_ai.auth.types import ApiKeyAuth, AuthResult, ModelAuth, OAuthAuth, OAuthCredential, ProviderAuth
from pidrei_ai.providers.faux import faux_assistant_message
from pidrei_ai.registry import ModelsRefreshOptions, create_provider
from pidrei_ai.types import DoneEvent, TextContent, Usage, UsageCost, UserMessage
from pidrei_ai.utils.event_stream import AssistantMessageEventStream

from .coding_session_helpers import now_ms
from .harness import create_harness


INDIVIDUAL_BASE_URL = "https://api.individual.githubcopilot.com"
ENTERPRISE_BASE_URL = "https://api.enterprise.githubcopilot.com"


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


async def _seed_compactable_session(harness) -> None:
    harness.settings_manager.apply_overrides({"compaction": {"keepRecentTokens": 1}})
    now = now_ms()
    await harness.session_manager.append_message(
        UserMessage(content=[TextContent(text="message to compact")], timestamp=now - 1000)
    )
    model = harness.get_model()
    assistant = dataclasses.replace(
        faux_assistant_message("assistant response to compact", timestamp=now - 500),
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(input=100, total_tokens=100, cost=UsageCost()),
    )
    await harness.session_manager.append_message(assistant)
    harness.session.agent.state.messages = harness.session_manager.build_session_context().messages


@pytest.mark.tonio
async def test_uses_the_auth_resolved_base_url_through_the_sdk_style_stream_wrapper(harnesses):
    harness = await create_harness()
    harnesses.append(harness)
    await _seed_compactable_session(harness)
    catalog_model = dataclasses.replace(harness.get_model(), base_url=INDIVIDUAL_BASE_URL)
    harness.session.agent.state.model = catalog_model

    observed: dict = {}

    def respond(request_model, *_args, **_kwargs):
        observed["base_url"] = request_model.base_url
        stream = AssistantMessageEventStream()
        stream.push(
            DoneEvent(
                reason="stop",
                message=dataclasses.replace(
                    faux_assistant_message("summary"),
                    api=request_model.api,
                    provider=request_model.provider,
                    model=request_model.id,
                ),
            )
        )
        return stream

    async def resolve(_ctx, credential, _cancel):
        if credential is not None and getattr(credential, "key", None):
            return AuthResult(auth=ModelAuth(api_key=credential.key), source="explicit token")
        return None

    async def login(_interaction):
        raise Exception("unused")

    async def refresh(credential, _cancel=None):
        return credential

    async def to_auth(credential):
        return ModelAuth(api_key=credential.access, base_url=ENTERPRISE_BASE_URL)

    provider = create_provider(
        id=catalog_model.provider,
        name="Copilot regression provider",
        base_url=INDIVIDUAL_BASE_URL,
        auth=ProviderAuth(
            api_key=ApiKeyAuth(name="Copilot token", resolve=resolve),
            oauth=OAuthAuth(name="Copilot OAuth", login=login, refresh=refresh, to_auth=to_auth),
        ),
        models=[catalog_model],
        api=type("_Api", (), {"stream": staticmethod(respond), "stream_simple": staticmethod(respond)})(),
    )

    async def set_oauth(_credential):
        return OAuthCredential(access="enterprise-token", refresh="refresh-token", expires=now_ms() + 60 * 60_000)

    await harness.auth_storage.modify(catalog_model.provider, set_oauth)
    model_runtime = harness.session.model_runtime
    model_runtime.register_native_provider(provider)
    await model_runtime.refresh(ModelsRefreshOptions(allow_network=False, providers=[catalog_model.provider]))

    # pi assigns `modelRuntime.streamSimple` directly (JS `await` tolerates a
    # non-promise); pidrei's stream functions are awaited, so wrap it.
    async def stream_function(model, context, options=None):
        return model_runtime.stream_simple(model, context, options)

    harness.session.agent.stream_function = stream_function

    await harness.session.compact()

    assert observed["base_url"] == ENTERPRISE_BASE_URL
