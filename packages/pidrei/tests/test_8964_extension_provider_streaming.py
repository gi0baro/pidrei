"""Mirror of pi's suite/regressions/8964-extension-provider-streaming.test.ts.

Regression for #8964: extensions can stream responses from providers
registered with `pi.register_provider()`. pi also asserts the provider's api
never reaches the global pi-ai/compat API registry; pidrei has no such
registry (compat.ts is not ported), so those two assertions have no subject.
"""

import pytest

from pidrei_ai.providers.faux import faux_assistant_message, faux_provider
from pidrei_ai.types import Context, UserMessage

from .harness import create_harness


@pytest.mark.tonio
@pytest.mark.parametrize("method", ["stream", "stream_simple"])
async def test_allows_an_extension_command_to_use_the_model_registry_stream(method):
    faux = faux_provider(provider="extension-provider", api="issue-8964-extension-api")
    received: dict = {}

    async def respond(_context, options, *_rest):
        received["api_key"] = options.api_key if options is not None else None
        return faux_assistant_message("custom provider response")

    faux.set_responses([respond])
    streamed: list[str] = []
    outcome: dict = {}
    model = faux.get_model()

    async def register_provider(pi) -> None:
        pi.register_provider(
            faux.provider.id,
            {
                "api": faux.api,
                "baseUrl": model.base_url,
                "apiKey": "extension-key",
                "models": [
                    {
                        "id": model.id,
                        "name": model.name,
                        "reasoning": model.reasoning,
                        "input": list(model.input),
                        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                        "contextWindow": model.context_window,
                        "maxTokens": model.max_tokens,
                    }
                ],
                "streamSimple": faux.provider.stream_simple,
            },
        )

    async def register_command(pi) -> None:
        async def handler(_args, ctx):
            registered = ctx.model_registry.find(faux.provider.id, model.id)
            stream = getattr(ctx.model_registry, method)(
                registered, Context(messages=[UserMessage(content="Hello", timestamp=0)])
            )
            async for event in stream:
                if event.type == "text_delta":
                    streamed.append(event.delta)
            outcome["result"] = await stream.result()

        pi.register_command("stream-custom", description="Stream a response from the custom provider", handler=handler)

    harness = await create_harness(extension_factories=[register_provider, register_command])
    try:
        await harness.session.prompt("/stream-custom")

        assert received["api_key"] == "extension-key"
        assert "".join(streamed) == "custom provider response"
        result = outcome["result"]
        assert result.stop_reason == "stop"
        assert [(block.type, block.text) for block in result.content] == [("text", "custom provider response")]
    finally:
        harness.cleanup()
