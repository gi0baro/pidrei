"""Mirror of pi's branch-summary-extensions.test.ts.

pi drives this through `test/suite/harness.ts`, which is unported. The
harness's job here is only "an AgentSession whose resource loader carries an
inline extension", so this builds that directly from
`create_agent_session` + `load_extension_from_factory` — the same extension
object the loader would hand the runner, made the same way.
"""

import shutil
import tempfile

import pytest

from pidrei.core.agent_session import ExtensionBindings
from pidrei.core.event_bus import EventBus
from pidrei.core.extensions import LoadExtensionsResult
from pidrei.core.extensions.loader import create_extension_runtime, load_extension_from_factory
from pidrei_ai.types import Usage, UsageCost
from pidrei_ai.utils.event_stream import AssistantMessageEventStream

from .agent_session_helpers import create_agent_session, create_test_resource_loader
from .coding_session_helpers import assistant_msg, user_msg


async def _stream_fn(_model, _context, options=None) -> AssistantMessageEventStream:
    return AssistantMessageEventStream()


@pytest.fixture
def temp_dir(request):
    path = tempfile.mkdtemp(prefix="pidrei-branch-summary-")
    request.addfinalizer(lambda: shutil.rmtree(path, ignore_errors=True))
    return path


@pytest.mark.tonio
async def test_persists_extension_provided_summary_usage_in_session_totals(temp_dir):
    usage = Usage(
        input=10,
        output=20,
        cache_read=30,
        cache_write=40,
        total_tokens=100,
        cost=UsageCost(input=0.1, output=0.2, cache_read=0.3, cache_write=0.4, total=1),
    )

    def factory(pi) -> None:
        async def on_session_before_tree(_event, _ctx):
            return {"summary": {"summary": "Summary provided by extension", "usage": usage}}

        pi.on("session_before_tree", on_session_before_tree)

    runtime = create_extension_runtime()
    extension = await load_extension_from_factory(factory, temp_dir, EventBus(), runtime, "<inline:1>")
    resource_loader = create_test_resource_loader(LoadExtensionsResult(extensions=[extension], runtime=runtime))

    session = await create_agent_session(
        temp_dir, stream_fn=_stream_fn, resource_loader=resource_loader, in_memory_session=False
    )
    await session.bind_extensions(ExtensionBindings())

    target_id = await session.session_manager.append_message(user_msg("first branch"))
    await session.session_manager.append_message(assistant_msg("first reply"))
    await session.session_manager.append_message(user_msg("abandoned branch work"))
    source_id = await session.session_manager.append_message(assistant_msg("abandoned reply"))

    result = await session.navigate_tree(target_id, {"summarize": True})
    summary_entry = result.summary_entry

    # Session entries are wire dicts here, not objects, so the keys stay camelCase.
    assert summary_entry["type"] == "branch_summary"
    assert summary_entry["parentId"] is None
    assert summary_entry["fromId"] == source_id
    assert summary_entry["fromHook"] is True
    assert summary_entry["summary"] == "Summary provided by extension"
    assert summary_entry["usage"] == usage

    stats = session.get_session_stats()
    assert (stats.tokens.input, stats.tokens.output) == (12, 22)
    assert (stats.tokens.cache_read, stats.tokens.cache_write) == (30, 40)
    assert stats.tokens.total == 104
    assert stats.cost == 1

    session.dispose()
