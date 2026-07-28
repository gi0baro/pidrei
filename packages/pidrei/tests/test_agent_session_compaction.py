"""Mirrors pi coding-agent test/agent-session-compaction.test.ts and
agent-session-tree-navigation.test.ts.

pi runs these behind describe.skipIf(!API_KEY) against real LLMs; pidrei runs
the same flows hermetically with a canned stream function that answers turn
prompts and detects summarization requests via the summarization system
prompt. The "custom summarization instructions" assertion becomes a payload
assert (the instructions reach the request) instead of an LLM-obedience check.
"""

import pytest
import tonio.colored as tonio

from pidrei.core.compaction import SUMMARIZATION_SYSTEM_PROMPT
from pidrei_ai.auth.types import ApiKeyAuth, AuthResult, ModelAuth, ProviderAuth
from pidrei_ai.registry import create_provider
from pidrei_ai.types import DoneEvent, ErrorEvent, StartEvent, Usage, UsageCost
from pidrei_ai.utils.event_stream import AssistantMessageEventStream

from .agent_session_helpers import create_agent_session, create_assistant_message


def _make_llm_stream_fn(*, summary_text="## Goal\nSummarized work.", answer_prefix="answer"):
    """Canned turn/summarization responder; records summarization requests."""
    state = {"turns": 0, "summarization_requests": []}

    async def stream_fn(_model, context, options=None):
        stream = AssistantMessageEventStream()
        if (context.system_prompt or "").startswith(SUMMARIZATION_SYSTEM_PROMPT[:40]):
            state["summarization_requests"].append((context, options))
            message = create_assistant_message(
                summary_text,
                usage=Usage(input=10, output=5, total_tokens=15, cost=UsageCost()),
            )
            stream.push(StartEvent(partial=create_assistant_message("")))
            stream.push(DoneEvent(reason="stop", message=message))
            return stream

        state["turns"] += 1
        message = create_assistant_message(
            f"{answer_prefix} {state['turns']}",
            provider="anthropic",
            model="claude-sonnet-4-5",
            usage=Usage(input=50, output=10, total_tokens=60, cost=UsageCost()),
        )
        stream.push(StartEvent(partial=create_assistant_message("")))
        stream.push(DoneEvent(reason="stop", message=message))
        return stream

    return stream_fn, state


async def _create_session(tmp_dir, *, in_memory=False, stream_fn=None, state=None, settings_overrides=None):
    if stream_fn is None:
        stream_fn, state = _make_llm_stream_fn()
    session = await create_agent_session(
        tmp_dir,
        stream_fn=stream_fn,
        in_memory_session=in_memory,
        settings_overrides=settings_overrides or {"compaction": {"keepRecentTokens": 1}},
        system_prompt="You are a helpful assistant. Be concise.",
    )
    events = []
    session.subscribe(events.append)
    return session, session.session_manager, events, state


class TestCompaction:
    @pytest.mark.tonio
    async def test_triggers_manual_compaction_via_compact(self, tmp_dir):
        session, _sm, _events, _state = await _create_session(tmp_dir)

        await session.prompt("What is 2+2? Reply with just the number.")
        await session.agent.wait_for_idle()

        await session.prompt("What is 3+3? Reply with just the number.")
        await session.agent.wait_for_idle()

        result = await session.compact()

        assert result.summary
        assert result.tokens_before > 0

        messages = session.messages
        assert len(messages) > 0

        # First message should be the summary
        assert messages[0].role == "compactionSummary"
        session.dispose()

    @pytest.mark.tonio
    async def test_manually_compacts_with_provider_resolved_bearer_auth(self, tmp_dir):
        # pi asserts inside its faux responder; here the canned summarizer
        # records the summarization (context, options) and the asserts run
        # after — same observable request auth. Auto-compaction is off because
        # the session is seeded through real prompts (pi seeds entries
        # directly), and keepRecentTokens=1 would auto-compact mid-seed.
        stream_fn, state = _make_llm_stream_fn(summary_text="summary with bearer auth")
        session = await create_agent_session(
            str(tmp_dir),
            stream_fn=stream_fn,
            settings_overrides={"compaction": {"enabled": False, "keepRecentTokens": 1}},
            system_prompt="You are a helpful assistant. Be concise.",
            provider_auth=None,
        )

        async def resolve(_ctx, _credential):
            return AuthResult(
                auth=ModelAuth(headers={"Authorization": "Bearer ambient-token"}),
                source="ambient bearer token",
            )

        session.model_runtime.register_native_provider(
            create_provider(
                id=session.model.provider,
                name="Faux bearer provider",
                auth=ProviderAuth(api_key=ApiKeyAuth(name="Faux bearer token", resolve=resolve)),
                models=[session.model],
                api={},
            )
        )

        await session.prompt("What is 2+2? Reply with just the number.")
        await session.agent.wait_for_idle()
        await session.prompt("What is 3+3? Reply with just the number.")
        await session.agent.wait_for_idle()

        result = await session.compact()

        assert "summary with bearer auth" in result.summary
        # pi's seeded session compacts in one call; prompting with
        # keepRecentTokens=1 produces a split turn, which legitimately
        # summarizes twice (history + turn prefix). Every request must carry
        # the provider-resolved bearer auth and no API key.
        assert state["summarization_requests"]
        for _summary_context, options in state["summarization_requests"]:
            assert options.api_key is None
            assert options.headers == {"Authorization": "Bearer ambient-token"}
        session.dispose()

    @pytest.mark.tonio
    async def test_maintains_valid_session_state_after_compaction(self, tmp_dir):
        session, _sm, _events, _state = await _create_session(tmp_dir)

        await session.prompt("What is the capital of France? One word answer.")
        await session.agent.wait_for_idle()

        await session.prompt("What is the capital of Germany? One word answer.")
        await session.agent.wait_for_idle()

        await session.compact()

        # Session should still be usable
        await session.prompt("What is the capital of Italy? One word answer.")
        await session.agent.wait_for_idle()

        assert len(session.messages) > 0
        assistant_messages = [m for m in session.messages if getattr(m, "role", None) == "assistant"]
        assert len(assistant_messages) > 0
        session.dispose()

    @pytest.mark.tonio
    async def test_persists_compaction_to_session_file(self, tmp_dir):
        session, session_manager, _events, _state = await _create_session(tmp_dir)

        await session.prompt("Say hello")
        await session.agent.wait_for_idle()

        await session.prompt("Say goodbye")
        await session.agent.wait_for_idle()

        await session.compact()

        entries = session_manager.get_entries()
        compaction_entries = [e for e in entries if e["type"] == "compaction"]
        assert len(compaction_entries) == 1

        compaction = compaction_entries[0]
        assert len(compaction["summary"]) > 0
        assert isinstance(compaction["firstKeptEntryId"], str)
        assert compaction["tokensBefore"] > 0
        session.dispose()

    @pytest.mark.tonio
    async def test_works_with_no_session_mode_in_memory_only(self, tmp_dir):
        session, session_manager, _events, _state = await _create_session(tmp_dir, in_memory=True)

        await session.prompt("What is 2+2? Reply with just the number.")
        await session.agent.wait_for_idle()

        await session.prompt("What is 3+3? Reply with just the number.")
        await session.agent.wait_for_idle()

        result = await session.compact()

        assert result.summary

        entries = session_manager.get_entries()
        compaction_entries = [e for e in entries if e["type"] == "compaction"]
        assert len(compaction_entries) == 1
        session.dispose()

    @pytest.mark.tonio
    async def test_emits_compaction_events_during_manual_compaction(self, tmp_dir):
        session, _sm, events, _state = await _create_session(tmp_dir)

        await session.prompt("Say hello")
        await session.agent.wait_for_idle()

        await session.compact()

        compaction_events = [e for e in events if e.type in ("compaction_start", "compaction_end")]
        assert len(compaction_events) == 2
        assert compaction_events[0].type == "compaction_start"
        assert compaction_events[0].reason == "manual"
        assert compaction_events[1].type == "compaction_end"
        assert compaction_events[1].reason == "manual"
        assert compaction_events[1].aborted is False
        assert compaction_events[1].will_retry is False

        message_end_events = [e for e in events if e.type == "message_end"]
        assert len(message_end_events) > 0
        session.dispose()


class TestTreeNavigation:
    @pytest.mark.tonio
    async def test_navigates_to_user_message_and_puts_text_in_editor(self, tmp_dir):
        session, session_manager, _events, _state = await _create_session(tmp_dir)

        await session.prompt("First message")
        await session.agent.wait_for_idle()
        await session.prompt("Second message")
        await session.agent.wait_for_idle()

        tree = session_manager.get_tree()
        assert len(tree) == 1

        root_node = tree[0]
        assert root_node.entry["type"] == "message"

        result = await session.navigate_tree(root_node.entry["id"], {"summarize": False})

        assert result.cancelled is False
        assert result.editor_text == "First message"

        # After navigating to root user message, leaf should be None
        assert session_manager.get_leaf_id() is None
        session.dispose()

    @pytest.mark.tonio
    async def test_navigates_to_non_user_message_without_editor_text(self, tmp_dir):
        session, session_manager, _events, _state = await _create_session(tmp_dir)

        await session.prompt("Hello")
        await session.agent.wait_for_idle()

        entries = session_manager.get_entries()
        assistant_entry = next(
            e for e in entries if e["type"] == "message" and getattr(e["message"], "role", None) == "assistant"
        )

        result = await session.navigate_tree(assistant_entry["id"], {"summarize": False})

        assert result.cancelled is False
        assert result.editor_text is None

        assert session_manager.get_leaf_id() == assistant_entry["id"]
        session.dispose()

    @pytest.mark.tonio
    async def test_creates_branch_summary_when_navigating_with_summarize(self, tmp_dir):
        session, session_manager, _events, _state = await _create_session(tmp_dir)

        await session.prompt("What is 2+2?")
        await session.agent.wait_for_idle()
        await session.prompt("What is 3+3?")
        await session.agent.wait_for_idle()

        tree = session_manager.get_tree()
        root_node = tree[0]

        result = await session.navigate_tree(root_node.entry["id"], {"summarize": True})

        assert result.cancelled is False
        assert result.editor_text == "What is 2+2?"
        assert result.summary_entry is not None
        assert result.summary_entry["type"] == "branch_summary"
        assert result.summary_entry["summary"]

        # Summary should be a root entry (parentId None) since we navigated to root user
        assert result.summary_entry["parentId"] is None

        # Leaf should be the summary entry
        assert session_manager.get_leaf_id() == result.summary_entry["id"]
        session.dispose()

    @pytest.mark.tonio
    async def test_attaches_summary_to_correct_parent_for_nested_user_message(self, tmp_dir):
        session, session_manager, _events, _state = await _create_session(tmp_dir)

        await session.prompt("Message one")
        await session.agent.wait_for_idle()
        await session.prompt("Message two")
        await session.agent.wait_for_idle()
        await session.prompt("Message three")
        await session.agent.wait_for_idle()

        entries = session_manager.get_entries()
        user_entries = [e for e in entries if e["type"] == "message" and getattr(e["message"], "role", None) == "user"]
        assert len(user_entries) == 3

        u2 = user_entries[1]
        a1 = next(e for e in entries if e["id"] == u2["parentId"])

        result = await session.navigate_tree(u2["id"], {"summarize": True})

        assert result.cancelled is False
        assert result.editor_text == "Message two"
        assert result.summary_entry is not None

        # Summary should be attached to a1 (parent of u2)
        assert result.summary_entry["parentId"] == a1["id"]

        children = session_manager.get_children(a1["id"])
        assert len(children) == 2

        child_types = sorted(c["type"] for c in children)
        assert "branch_summary" in child_types
        assert "message" in child_types
        session.dispose()

    @pytest.mark.tonio
    async def test_attaches_summary_to_selected_node_for_assistant_message(self, tmp_dir):
        session, session_manager, _events, _state = await _create_session(tmp_dir)

        await session.prompt("Hello")
        await session.agent.wait_for_idle()
        await session.prompt("Goodbye")
        await session.agent.wait_for_idle()

        entries = session_manager.get_entries()
        assistant_entries = [
            e for e in entries if e["type"] == "message" and getattr(e["message"], "role", None) == "assistant"
        ]
        a1 = assistant_entries[0]

        result = await session.navigate_tree(a1["id"], {"summarize": True})

        assert result.cancelled is False
        assert result.editor_text is None  # No editor text for assistant messages
        assert result.summary_entry is not None

        # Summary should be attached to a1 (the selected node)
        assert result.summary_entry["parentId"] == a1["id"]

        assert session_manager.get_leaf_id() == result.summary_entry["id"]
        session.dispose()

    @pytest.mark.tonio
    async def test_handles_abort_during_summarization(self, tmp_dir):
        base_stream_fn, state = _make_llm_stream_fn()

        async def stream_fn(model, context, options=None):
            if (context.system_prompt or "").startswith(SUMMARIZATION_SYSTEM_PROMPT[:40]):
                # Summarization hangs until aborted, then reports an aborted result.
                cancel = getattr(options, "cancel", None)
                stream = AssistantMessageEventStream()

                async def watch_abort() -> None:
                    while cancel is None or not cancel.cancelled:
                        await tonio.time.sleep(0.005)
                    stream.push(
                        ErrorEvent(
                            reason="aborted",
                            error=create_assistant_message("", stop_reason="aborted"),
                        )
                    )

                tonio.spawn.without_tracking(watch_abort())
                return stream
            return await base_stream_fn(model, context, options)

        session, session_manager, _events, _state = await _create_session(tmp_dir, stream_fn=stream_fn, state=state)

        await session.prompt("Tell me about something")
        await session.agent.wait_for_idle()
        await session.prompt("Continue")
        await session.agent.wait_for_idle()

        entries_before = session_manager.get_entries()
        leaf_before = session_manager.get_leaf_id()

        tree = session_manager.get_tree()
        root_node = tree[0]

        navigation = tonio.spawn(session.navigate_tree(root_node.entry["id"], {"summarize": True}))

        await tonio.time.sleep(0.1)

        # is_compacting should be True during branch summarization
        assert session.is_compacting is True

        session.abort_branch_summary()

        result = await navigation

        assert result.cancelled is True
        assert result.aborted is True
        assert result.summary_entry is None

        # Session should be unchanged
        assert len(session_manager.get_entries()) == len(entries_before)
        assert session_manager.get_leaf_id() == leaf_before
        session.dispose()

    @pytest.mark.tonio
    async def test_does_not_create_summary_when_navigating_without_summarize(self, tmp_dir):
        session, session_manager, _events, _state = await _create_session(tmp_dir)

        await session.prompt("First")
        await session.agent.wait_for_idle()
        await session.prompt("Second")
        await session.agent.wait_for_idle()

        entries_before = len(session_manager.get_entries())

        tree = session_manager.get_tree()
        await session.navigate_tree(tree[0].entry["id"], {"summarize": False})

        assert len(session_manager.get_entries()) == entries_before

        summaries = [e for e in session_manager.get_entries() if e["type"] == "branch_summary"]
        assert summaries == []
        session.dispose()

    @pytest.mark.tonio
    async def test_handles_navigation_to_same_position_noop(self, tmp_dir):
        session, session_manager, _events, _state = await _create_session(tmp_dir)

        await session.prompt("Hello")
        await session.agent.wait_for_idle()

        leaf_before = session_manager.get_leaf_id()
        assert leaf_before
        entries_before = len(session_manager.get_entries())

        result = await session.navigate_tree(leaf_before, {"summarize": False})

        assert result.cancelled is False
        assert session_manager.get_leaf_id() == leaf_before
        assert len(session_manager.get_entries()) == entries_before
        session.dispose()

    @pytest.mark.tonio
    async def test_supports_custom_summarization_instructions(self, tmp_dir):
        stream_fn, state = _make_llm_stream_fn()
        session, session_manager, _events, state = await _create_session(tmp_dir, stream_fn=stream_fn, state=state)

        await session.prompt("What is TypeScript?")
        await session.agent.wait_for_idle()

        tree = session_manager.get_tree()
        custom = "After the summary, you MUST end with exactly: MONKEY MONKEY MONKEY. This is of utmost importance."
        result = await session.navigate_tree(tree[0].entry["id"], {"summarize": True, "custom_instructions": custom})

        assert result.summary_entry is not None
        assert result.summary_entry["summary"]
        # Hermetic substitution for pi's LLM-obedience assert: the custom
        # instructions must reach the summarization request payload.
        assert len(state["summarization_requests"]) == 1
        request_context, _options = state["summarization_requests"][0]
        request_text = request_context.messages[0].content[0].text
        assert f"Additional focus: {custom}" in request_text
        session.dispose()

    @pytest.mark.tonio
    async def test_navigates_between_branches_correctly(self, tmp_dir):
        session, session_manager, _events, _state = await _create_session(tmp_dir, settings_overrides={})

        await session.prompt("Main branch start")
        await session.agent.wait_for_idle()
        await session.prompt("Main branch continue")
        await session.agent.wait_for_idle()

        entries = session_manager.get_entries()
        a1 = next(e for e in entries if e["type"] == "message" and getattr(e["message"], "role", None) == "assistant")

        # Create a branch from a1
        session_manager.branch(a1["id"])
        await session.prompt("Branch path")
        await session.agent.wait_for_idle()

        user_entries = [e for e in entries if e["type"] == "message" and getattr(e["message"], "role", None) == "user"]
        u2 = user_entries[1]  # "Main branch continue"

        result = await session.navigate_tree(u2["id"], {"summarize": True})

        assert result.cancelled is False
        assert result.editor_text == "Main branch continue"
        assert result.summary_entry is not None
        assert len(result.summary_entry["summary"]) > 0
        session.dispose()
