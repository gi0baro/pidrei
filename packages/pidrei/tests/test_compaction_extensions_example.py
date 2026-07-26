"""Mirror of pi's compaction-extensions-example.test.ts.

pi's version is a *compile-time* test: it defines the documentation example
and asserts `typeof exampleExtension === "function"`. The handler is never
invoked, so every `expect` inside it is dead code — the only thing pi is really
checking is that the example still type-checks against `SessionBeforeCompactEvent`
and `SessionCompactEvent`.

Python has no such check, so this does the runtime equivalent instead, which is
strictly stronger: it registers the documented handlers through the real
`ExtensionAPI`, emits the two events through the real `ExtensionRunner` with
the shapes `agent_session.py` builds, and asserts the example reads every
documented field and that its returned compaction reaches the runner.
"""

from types import SimpleNamespace

import pytest

from pidrei.core.event_bus import EventBus
from pidrei.core.extensions.loader import create_extension_runtime, load_extension_from_factory
from pidrei.core.extensions.runner import ExtensionRunner
from pidrei.core.session_manager import SessionManager


PREPARATION = SimpleNamespace(
    messages_to_summarize=[
        {"role": "user", "content": "first request"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second request"},
    ],
    turn_prefix_messages=[],
    tokens_before=1234,
    first_kept_entry_id="entry-7",
    is_split_turn=False,
)


async def make_runner(extension, runtime) -> ExtensionRunner:
    return ExtensionRunner(
        [extension],
        runtime,
        ".",
        SessionManager.in_memory(),
        SimpleNamespace(get_api_key_and_headers=lambda _model: None),
    )


@pytest.mark.tonio
async def test_the_custom_compaction_example_reads_every_documented_field():
    observed: dict = {}

    def factory(pi) -> None:
        async def on_before_compact(event, ctx):
            preparation = event["preparation"]
            observed["branch_entries"] = event["branchEntries"]
            observed["session_manager"] = ctx.session_manager
            observed["model_registry"] = ctx.model_registry
            observed["fields"] = {
                "messages_to_summarize": preparation.messages_to_summarize,
                "turn_prefix_messages": preparation.turn_prefix_messages,
                "tokens_before": preparation.tokens_before,
                "first_kept_entry_id": preparation.first_kept_entry_id,
                "is_split_turn": preparation.is_split_turn,
            }

            summary = "\n".join(
                f"- {message['content'][:100]}"
                for message in preparation.messages_to_summarize
                if message["role"] == "user"
            )
            return {
                "compaction": {
                    "summary": f"User requests:\n{summary}",
                    "firstKeptEntryId": preparation.first_kept_entry_id,
                    "tokensBefore": preparation.tokens_before,
                }
            }

        pi.on("session_before_compact", on_before_compact)

    runtime = create_extension_runtime()
    extension = await load_extension_from_factory(factory, ".", EventBus(), runtime, "<inline:1>")
    runner = await make_runner(extension, runtime)

    result = await runner.emit(
        {
            "type": "session_before_compact",
            "preparation": PREPARATION,
            "branchEntries": [{"type": "message"}],
            "customInstructions": None,
            "reason": "manual",
            "willRetry": False,
            "signal": None,
        }
    )

    fields = observed["fields"]
    assert isinstance(fields["messages_to_summarize"], list)
    assert isinstance(fields["turn_prefix_messages"], list)
    assert isinstance(fields["is_split_turn"], bool)
    assert isinstance(fields["tokens_before"], int)
    assert isinstance(fields["first_kept_entry_id"], str)
    assert isinstance(observed["branch_entries"], list)
    assert callable(observed["session_manager"].get_entries)
    assert callable(observed["model_registry"].get_api_key_and_headers)

    assert result["compaction"] == {
        "summary": "User requests:\n- first request\n- second request",
        "firstKeptEntryId": "entry-7",
        "tokensBefore": 1234,
    }


@pytest.mark.tonio
async def test_the_compact_event_carries_the_documented_fields():
    observed: dict = {}

    def factory(pi) -> None:
        async def on_compact(event, _ctx):
            observed["entry"] = event["compactionEntry"]
            observed["from_extension"] = event["fromExtension"]

        pi.on("session_compact", on_compact)

    runtime = create_extension_runtime()
    extension = await load_extension_from_factory(factory, ".", EventBus(), runtime, "<inline:1>")
    runner = await make_runner(extension, runtime)

    await runner.emit(
        {
            "type": "session_compact",
            "compactionEntry": {"type": "compaction", "summary": "done", "tokensBefore": 1234},
            "fromExtension": True,
            "reason": "manual",
            "willRetry": False,
        }
    )

    assert observed["entry"]["type"] == "compaction"
    assert isinstance(observed["entry"]["summary"], str)
    assert isinstance(observed["entry"]["tokensBefore"], int)
    assert isinstance(observed["from_extension"], bool)
