"""Guard for the extension seam that no other test enters.

Not a pi mirror: pi has no test here, which is exactly why Phase 4.5's defect
four — a context entry returning the bound method instead of calling it — sat
latent. The bug class is invisible until an extension actually reads the entry,
and extension *loading* only lands in Phase 5e, so this test stands in for the
loader: it takes the real dicts `AgentSession._bind_extension_core` hands to
`ExtensionRunner.bind_core`, calls every zero-argument entry, and asserts none
of them hands back something still waiting to be called.

Entries that take arguments or mutate the session (compact, fork, new_session,
switch_session, navigate_tree, set_model, …) are checked for presence and
callability only; driving them is the job of their own behaviour tests.
"""

import contextlib
import inspect

import pytest

from pidrei.core.extensions.runner import ExtensionRunner
from pidrei_ai.utils.event_stream import AssistantMessageEventStream

from .agent_session_helpers import create_agent_session


async def _stream_fn(_model, _context, options=None) -> AssistantMessageEventStream:
    return AssistantMessageEventStream()


@contextlib.contextmanager
def _recording_bind_core():
    """Capture the dicts the session binds (no yield fixtures: tonio)."""
    captured: dict[str, dict] = {}
    original = ExtensionRunner.bind_core

    def recording(self, actions, context_actions, provider_actions=None):
        captured["actions"] = actions
        captured["context_actions"] = context_actions
        captured["provider_actions"] = provider_actions or {}
        return original(self, actions, context_actions, provider_actions)

    ExtensionRunner.bind_core = recording
    try:
        yield captured
    finally:
        ExtensionRunner.bind_core = original


def _zero_argument_entries(entries: dict) -> dict:
    zero_arg = {}
    for name, entry in entries.items():
        assert callable(entry), f"{name} must be callable"
        try:
            signature = inspect.signature(entry)
        except TypeError, ValueError:  # pragma: no cover - builtins have no signature
            continue
        required = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if not required:
            zero_arg[name] = entry
    return zero_arg


@pytest.mark.tonio
async def test_every_zero_argument_extension_context_entry_returns_a_value(tmp_dir):
    with _recording_bind_core() as captured:
        await create_agent_session(tmp_dir, stream_fn=_stream_fn)

    assert captured, "AgentSession must bind the extension core"
    context_actions = captured["context_actions"]
    # The keys themselves are already guarded: bind_core indexes the required
    # ones, so a rename on either side fails session construction above.
    assert "get_context_usage" in context_actions

    for name, entry in _zero_argument_entries(context_actions).items():
        result = entry()
        assert not callable(result), f"context action {name!r} returned a callable — missing a call?"


@pytest.mark.tonio
async def test_every_zero_argument_extension_action_returns_a_value(tmp_dir):
    with _recording_bind_core() as captured:
        await create_agent_session(tmp_dir, stream_fn=_stream_fn)

    for name, entry in _zero_argument_entries(captured["actions"]).items():
        result = entry()
        assert not callable(result), f"action {name!r} returned a callable — missing a call?"


@pytest.mark.tonio
async def test_every_extension_context_read_accessor_resolves(tmp_dir):
    """The same seam from the extension's side: the context object's accessors."""
    session = await create_agent_session(tmp_dir, stream_fn=_stream_fn)
    context = session._extension_runner.create_command_context()

    properties = [
        name
        for name in dir(type(context))
        if not name.startswith("_") and isinstance(getattr(type(context), name), property)
    ]
    assert "model" in properties and "thinking_level" in properties

    for name in properties:
        value = getattr(context, name)
        assert not inspect.ismethod(value) and not inspect.isfunction(value), (
            f"context.{name} resolved to a callable — missing a call?"
        )

    for name in ("get_context_usage", "get_system_prompt", "get_system_prompt_options"):
        value = getattr(context, name)()
        assert not inspect.ismethod(value) and not inspect.isfunction(value), (
            f"context.{name}() resolved to a callable — missing a call?"
        )

    assert context.is_idle() is True
    assert context.has_pending_messages() is False
    assert isinstance(context.is_project_trusted(), bool)
