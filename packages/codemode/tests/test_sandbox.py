"""Mirror of pi codemode test/sandbox.test.ts, with scripts in Python.

JavaScript-only cases are dropped (prelude parse, non-Error throws, microtask
spinning, the never-settling promise, worker path strings); the rest are
adapted or re-specified for Monty. A script's value is its last line, not a
`return` (Monty's checker rejects a top-level `return`). Cases with no pi
counterpart are at the end: the type check, `all_settled`, `call_tool`,
refused positional arguments, dropped `None` arguments, the execution cap and
output merging.

Scripts are type-checked unless a case is about what happens at run time
(`type_check=False`).
"""

import math
import os
import signal
import threading

import pytest
import tonio.colored as tonio
from tonio.colored.exceptions import CancelledError

from pidrei_codemode import (
    CodemodeError,
    CodemodeGlobal,
    CodemodeImageItem,
    CodemodePool,
    CodemodeSandbox,
    CodemodeStoreWrites,
    CodemodeTextItem,
    CodemodeTool,
)
from pidrei_utils.cancel import CancelToken


@pytest.fixture
async def make_sandbox(pool):
    sandboxes: list[CodemodeSandbox] = []

    def make(tools=(), **options) -> CodemodeSandbox:
        sandbox = CodemodeSandbox(pool, tools=tools, **({"timeout_ms": 10_000} | options))
        sandboxes.append(sandbox)
        return sandbox

    try:
        yield make
    finally:
        for sandbox in sandboxes:
            await sandbox.close()


def tool(name: str, execute, **fields) -> CodemodeTool:
    return CodemodeTool(name=name, execute=execute, **fields)


async def _echo(args):
    return args


async def _fail(_args):
    raise RuntimeError("tool exploded")


echo = tool("echo", _echo)
fail = tool("fail", _fail)


def hanging_tool(name: str = "hang"):
    """A tool that never returns; `started` is set when it runs, `cancelled`
    when it is cancelled."""
    started = tonio.Event()
    cancelled = tonio.Event()

    async def hang(_args):
        started.set()
        try:
            await tonio.Event().wait()
        except CancelledError:
            cancelled.set()
            raise

    return tool(name, hang), started, cancelled


def statuses(result):
    return [(call.name, call.status) for call in result.calls]


TYPE_CHECK_FAILED = "The script did not run: type checking failed.\n"

# Base64 of the leading bytes of each format. image() only inspects the signature.
PNG = "iVBORw0KGgo="
JPEG = "/9j/4A=="
GIF = "R0lGODlh"
WEBP = "UklGRgAAAABXRUJQ"


# -- script execution --------------------------------------------------------


@pytest.mark.tonio
async def test_returns_the_value_of_the_last_line_after_a_json_round_trip(make_sandbox):
    sandbox = make_sandbox()
    result = await sandbox.execute("{'a': 1, 'b': [True, 'x'], 'c': (1, 2)}")
    assert result.ok
    assert result.value == {"a": 1, "b": [True, "x"], "c": [1, 2]}
    assert result.output == ()
    assert result.calls == ()
    assert (await sandbox.execute("'plain'")).value == "plain"
    empty = await sandbox.execute("")
    assert empty.ok and empty.value is None
    # A last line that is not an expression gives no value.
    assigned = await sandbox.execute("x = 41")
    assert assigned.ok and assigned.value is None
    # The checker rejects a top-level `return`.
    returned = await sandbox.execute("x = 1\nreturn x")
    assert not returned.ok
    assert returned.error.stack.startswith(TYPE_CHECK_FAILED)
    assert "`return` statement outside of a function" in returned.error.stack


@pytest.mark.tonio
async def test_supports_top_level_await(make_sandbox):
    result = await make_sandbox().execute("async def f():\n    return 41\nx = await f()\nx + 1")
    assert result.ok and result.value == 42


@pytest.mark.tonio
async def test_collects_text_image_and_print_output_in_order(make_sandbox):
    result = await make_sandbox(type_check=False).execute(
        f"""print("hello", 1, {{"a": 1}})
text({{"json": True}})
text(None)
text(7)
image("data:image/png;base64,{PNG}")
image({{"image_url": "data:image/jpeg;base64,{JPEG}"}})
image({{"type": "image", "data": "{GIF}", "mimeType": "image/gif"}})
image("data:image/png;base64,{WEBP}")
image({{"type": "image", "data": "{PNG}"}})
print(ValueError("bad"))"""
    )
    assert result.ok
    assert result.output == (
        CodemodeTextItem("hello 1 {'a': 1}"),
        CodemodeTextItem('{"json":true}'),
        CodemodeTextItem("None"),
        CodemodeTextItem("7"),
        CodemodeImageItem(PNG, "image/png"),
        CodemodeImageItem(JPEG, "image/jpeg"),
        CodemodeImageItem(GIF, "image/gif"),
        # The MIME type comes from the data, not from the declared type.
        CodemodeImageItem(WEBP, "image/webp"),
        CodemodeImageItem(PNG, "image/png"),
        CodemodeTextItem("bad"),
    )


@pytest.mark.tonio
async def test_rejects_invalid_text_and_image_arguments(make_sandbox):
    result = await make_sandbox(type_check=False).execute(
        """errors = []
cases = [
    lambda: text({1, 2}),
    lambda: image(""),
    lambda: image("https://example.com/a.png"),
    lambda: image("data:image/png,raw"),
    lambda: image({"type": "text", "text": "x"}),
    lambda: image({"type": "image", "data": ""}),
    lambda: image(42),
    lambda: image("data:image/png;base64,AAAA!"),
    lambda: image("data:image/png;base64,AAAAA"),
    lambda: image("data:image/png;base64,AA=A"),
    lambda: image("data:image/png;base64,"),
    lambda: image("data:image/png;base64,AAAA\\n[Output truncated]"),
    lambda: image({"type": "image", "data": "AAAA!", "mimeType": "image/png"}),
    lambda: image("data:image/png;base64,AAAA"),
    lambda: image("data:image/png;base64,QUJD"),
    lambda: image("data:image/jpeg;base64,/9j/9w=="),
]
for run in cases:
    try:
        run()
        errors.append("no error")
    except Exception as e:
        errors.append(type(e).__name__ + ": " + str(e))
errors"""
    )
    assert result.ok
    assert result.output == ()
    errors = result.value
    assert errors[0].startswith("TypeError: ") and "not JSON serializable" in errors[0]
    assert errors[1:] == [
        "TypeError: image expects a non-empty image URL string, an object with image_url, or a raw MCP image block",
        "TypeError: remote image URLs are not supported in tool outputs. Pass a base64 data URI instead",
        "TypeError: invalid image output. Pass a base64 data URI instead",
        'TypeError: image only accepts MCP image blocks, got "text"',
        "TypeError: image expected MCP image data",
        "TypeError: image expects a non-empty image URL string, an object with image_url, or a raw MCP image block",
        *["TypeError: invalid image output. The image data is not valid base64 (truncated or corrupted?)"] * 6,
        *["TypeError: invalid image output. The image data is not a PNG, JPEG, GIF, or WebP image"] * 3,
    ]


@pytest.mark.tonio
async def test_accepts_wrapped_and_large_base64_image_data(make_sandbox):
    large = "iVBORw0KGgoA" + "QUJD" * (256 * 1024)
    result = await make_sandbox().execute(
        f'image("data:image/png;base64,iVBORw0K\\r\\nGgo=\\n")\nimage("data:image/png;base64,{large}")'
    )
    assert result.ok
    assert result.output == (CodemodeImageItem(PNG, "image/png"), CodemodeImageItem(large, "image/png"))


@pytest.mark.tonio
async def test_ends_the_script_successfully_on_exit_keeping_output_and_store_writes(make_sandbox):
    result = await make_sandbox([echo]).execute(
        """text("before")
store("k", 1)
await tools.echo(x=1)
try:
    exit()
except Exception:
    pass
text("after")
"unreachable\""""
    )
    assert result.ok
    assert result.value is None
    assert result.output == (CodemodeTextItem("before"),)
    assert result.store_writes == CodemodeStoreWrites(set={"k": 1})


@pytest.mark.tonio
async def test_keeps_output_produced_before_a_failure(make_sandbox):
    result = await make_sandbox().execute('text("partial")\nraise RuntimeError("boom")')
    assert not result.ok
    assert result.output == (CodemodeTextItem("partial"),)


@pytest.mark.tonio
async def test_reports_syntax_errors_with_the_scripts_line_number(make_sandbox):
    unchecked = await make_sandbox(type_check=False).execute("a = 1\nb =\na")
    assert not unchecked.ok
    assert (unchecked.error.kind, unchecked.error.name) == ("script", "SyntaxError")
    assert "line 2" in unchecked.error.stack
    # With the check on, the checker reports it.
    checked = await make_sandbox().execute("a = 1\nb =\na")
    assert checked.error.stack.startswith(TYPE_CHECK_FAILED)
    assert "main.py:2:" in checked.error.stack


@pytest.mark.tonio
async def test_reports_raised_errors_with_the_scripts_line_number(make_sandbox):
    result = await make_sandbox().execute("a = 1\nraise TypeError('boom ' + str(a))")
    assert not result.ok
    assert (result.error.kind, result.error.name, result.error.message) == ("script", "TypeError", "boom 1")
    assert "line 2" in result.error.stack


@pytest.mark.tonio
async def test_formats_tracebacks_like_cpython_without_prelude_frames(make_sandbox):
    result = await make_sandbox().execute("def f():\n    raise ValueError('outer')\n\nf()")
    assert not result.ok
    stack = result.error.stack
    assert stack.startswith("Traceback (most recent call last):\n")
    assert "line 4, in <module>" in stack
    assert "line 2, in f" in stack
    assert stack.endswith("ValueError: outer")
    # The prelude is the session's first feed: none of its frames show.
    assert "<python-input-0>" not in stack


@pytest.mark.tonio
async def test_reports_a_non_serializable_value_as_a_script_error(make_sandbox):
    result = await make_sandbox().execute("{1, 2}")
    assert not result.ok
    assert (result.error.kind, result.error.name) == ("script", "TypeError")


# -- tools -------------------------------------------------------------------


@pytest.mark.tonio
async def test_exposes_tools_as_async_functions_and_records_calls(make_sandbox):
    seen = []

    async def add(args):
        seen.append(args)
        return {"sum": args["a"] + args["b"]}

    result = await make_sandbox([tool("add", add)]).execute(
        "first = await tools.add(a=1, b=2)\nsecond = await tools.add(a=first['sum'], b=10)\nsecond['sum']"
    )
    assert result.ok and result.value == 13
    assert seen == [{"a": 1, "b": 2}, {"a": 3, "b": 10}]
    assert statuses(result) == [("add", "ok"), ("add", "ok")]
    assert all(call.duration_ms >= 0 for call in result.calls)


@pytest.mark.tonio
async def test_runs_concurrent_calls_and_lists_tool_names(make_sandbox):
    lock = threading.Lock()
    entered = []
    both = tonio.Event()

    async def gate(args):
        # Returns only once both gate calls are in flight at the same time.
        with lock:
            entered.append(args["x"])
            if len(entered) == 2:
                both.set()
        await both.wait(5)
        if not both.is_set():
            raise RuntimeError("the calls did not overlap")
        return args["x"]

    async def value(args):
        return args["x"]

    result = await make_sandbox([tool("value", value), tool("gate", gate)]).execute(
        """import asyncio
a, b, c = await asyncio.gather(tools.gate(x=1), tools.gate(x=2), tools.value(x=3))
{'values': [a, b, c], 'names': [t['name'] for t in ALL_TOOLS]}"""
    )
    assert result.ok
    assert result.value == {"values": [1, 2, 3], "names": ["value", "gate"]}


@pytest.mark.tonio
async def test_exposes_tools_under_normalized_identifiers_and_lists_them_in_all_tools(make_sandbox):
    def returning(value):
        async def execute(_args):
            return value

        return execute

    result = await make_sandbox(
        [
            tool("my-tool", returning("dash"), description="Dashes"),
            tool("my_tool", returning("underscore"), description="Shadowed"),
            tool("mcp__docs__search", returning("mcp")),
            tool("class", returning("keyword")),
            tool("search", _echo, input_schema={"type": "object", "properties": {"max-results": {"type": "integer"}}}),
        ]
    ).execute(
        """{
    'all': ALL_TOOLS,
    'calls': [
        await tools.my_tool(),
        await tools.mcp__docs__search(),
        await tools.class_(),
        await tools.search(**{'max-results': 5}),
    ],
}"""
    )
    assert result.ok
    assert result.value == {
        "all": [
            {"name": "my_tool", "description": "Dashes"},
            {"name": "mcp__docs__search", "description": ""},
            {"name": "class_", "description": ""},
            {"name": "search", "description": ""},
        ],
        "calls": ["dash", "mcp", "keyword", {"max-results": 5}],
    }


@pytest.mark.tonio
async def test_passes_missing_arguments_and_none_results_through(make_sandbox):
    async def none(_args):
        return None

    result = await make_sandbox([tool("noop", _echo), tool("none", none)]).execute(
        "[await tools.noop(), await tools.none()]"
    )
    assert result.ok and result.value == [{}, None]


@pytest.mark.tonio
async def test_turns_tool_errors_into_catchable_errors_in_the_script(make_sandbox):
    result = await make_sandbox([fail]).execute(
        """try:
    await tools.fail()
    outcome = 'no error'
except Exception as e:
    outcome = [type(e).__name__, str(e)]
outcome"""
    )
    assert result.ok and result.value == ["RuntimeError", "tool exploded"]
    assert statuses(result) == [("fail", "error")]


@pytest.mark.tonio
async def test_rejects_calls_to_unknown_tools(make_sandbox):
    unchecked = await make_sandbox(type_check=False).execute("await tools.missing()")
    assert not unchecked.ok
    assert (unchecked.error.kind, unchecked.error.name) == ("script", "AttributeError")
    assert unchecked.error.message.startswith("tools.missing does not exist.")
    # With the check on the script never runs.
    checked = await make_sandbox().execute("await tools.missing()")
    assert not checked.ok
    assert checked.error.kind == "script"
    assert checked.error.stack.startswith(TYPE_CHECK_FAILED)
    assert "missing" in checked.error.stack


@pytest.mark.tonio
async def test_cancels_unawaited_calls_when_the_script_ends(make_sandbox):
    slow, running, cancelled = hanging_tool("slow")

    async def ready(_args):
        # Holds the script until the unawaited call is running.
        await running.wait(5)
        return running.is_set()

    # The checker rejects a bare unawaited call (see the pidrei cases).
    result = await make_sandbox([slow, tool("ready", ready)], type_check=False).execute(
        "tools.slow()\nawait tools.ready()\n'early'"
    )
    assert result.ok and result.value == "early"
    assert statuses(result) == [("slow", "cancelled"), ("ready", "ok")]
    await cancelled.wait(5)
    assert cancelled.is_set()


@pytest.mark.tonio
async def test_names_close_matches_when_a_script_calls_a_tool_that_does_not_exist(make_sandbox):
    sandbox = make_sandbox([echo, tool("web-search", _echo)], type_check=False)

    async def attempt(expression: str):
        result = await sandbox.execute(expression)
        return result.value if result.ok else result.error.message

    expected = (
        "tools.Echo does not exist. Did you mean tools.echo? ALL_TOOLS lists every tool; search_tools(query) "
        'finds tools by topic. Check for a tool with has_tool("Echo").'
    )
    assert await attempt("await tools.Echo()") == expected
    assert await attempt("await call_tool('Echo')") == expected
    assert "Did you mean tools.web_search?" in await attempt("await tools.websearch()")
    assert "Available: echo, web_search." in await attempt("await tools.nothing()")
    assert await attempt("[has_tool('echo'), has_tool('nothing')]") == [True, False]
    # Reading a member without calling it is Monty's own AttributeError.
    read = await sandbox.execute("f = tools.nothing")
    assert read.error.name == "AttributeError"


@pytest.mark.tonio
async def test_supports_register_and_unregister_between_executions(make_sandbox):
    sandbox = make_sandbox()
    sandbox.register_tool(echo)
    with pytest.raises(ValueError, match="already registered"):
        sandbox.register_tool(echo)
    assert [registered.name for registered in sandbox.tools] == ["echo"]
    assert (await sandbox.execute("await tools.echo(x='a')")).value == {"x": "a"}
    assert sandbox.unregister_tool("echo")
    assert (await sandbox.execute("has_tool('echo')")).value is False


# -- store and load ----------------------------------------------------------


@pytest.mark.tonio
async def test_reads_the_snapshot_and_reports_writes(make_sandbox):
    result = await make_sandbox().execute(
        """seen = load('counter')
store('counter', seen + 1)
store('list', [1, {'a': None}])
store('old', None)
[seen, load('counter'), load('missing'), load('old')]""",
        store={"counter": 41, "old": "x"},
    )
    assert result.ok
    assert result.value == [41, 42, None, None]
    assert result.store_writes == CodemodeStoreWrites(set={"counter": 42, "list": [1, {"a": None}]}, delete=("old",))


@pytest.mark.tonio
async def test_returns_copies_so_mutating_a_loaded_value_does_not_change_the_store(make_sandbox):
    result = await make_sandbox().execute(
        """value = load('obj')
value['a'] = 2
kept = {'b': 1}
store('kept', kept)
kept['b'] = 2
[load('obj')['a'], load('kept')['b']]""",
        store={"obj": {"a": 1}},
    )
    assert result.ok and result.value == [1, 1]
    assert result.store_writes.set == {"kept": {"b": 1}}


@pytest.mark.tonio
async def test_rejects_invalid_keys_values_and_oversized_writes_inside_the_script(make_sandbox):
    result = await make_sandbox(type_check=False).execute(
        """def attempt(fn):
    try:
        fn()
        return 'ok'
    except Exception as e:
        return type(e).__name__

def fill():
    for i in range(8):
        store('k' + str(i), 'x' * (200 * 1024))

[
    attempt(lambda: store(1, 'x')),
    attempt(lambda: load({})),
    attempt(lambda: store('s', {1})),
    attempt(lambda: store('big', 'x' * (300 * 1024))),
    attempt(fill),
]"""
    )
    assert result.ok
    assert result.value == ["TypeError", "TypeError", "TypeError", "ValueError", "ValueError"]


@pytest.mark.tonio
async def test_explains_oversized_writes(make_sandbox):
    result = await make_sandbox().execute("store('img', 'x' * (300 * 1024))")
    assert not result.ok
    assert 'store("img") value has 307202 characters of JSON' in result.error.message
    assert "Show images with image()" in result.error.message


@pytest.mark.tonio
async def test_reserves_the_names_of_the_builtin_globals(pool):
    async def execute(_args, _kwargs):
        return None

    for name in ("store", "load", "has_tool", "all_settled", "call_tool"):
        with pytest.raises(ValueError, match="Invalid global"):
            CodemodeSandbox(pool, globals=[CodemodeGlobal(name, execute)])


# -- globals -----------------------------------------------------------------


@pytest.mark.tonio
async def test_exposes_globals_as_top_level_functions_without_recording_them_as_calls(make_sandbox):
    seen = []

    async def attach(args, kwargs):
        seen.append((args, kwargs))

    result = await make_sandbox([echo], globals=[CodemodeGlobal("attach", attach)]).execute(
        "await attach(ref=1)\nf = attach\nawait f('positional')\nawait tools.echo(x=2)"
    )
    assert result.ok and result.value == {"x": 2}
    assert [call.name for call in result.calls] == ["echo"]
    assert seen == [((), {"ref": 1}), (("positional",), {})]


@pytest.mark.tonio
async def test_groups_namespaced_globals_and_passes_their_arguments(make_sandbox):
    seen = []

    async def list_models(args, _kwargs):
        seen.append(list(args))

    async def first(args, _kwargs):
        return args[0]

    result = await make_sandbox(
        globals=[CodemodeGlobal("models.list", list_models), CodemodeGlobal("models.first", first)]
    ).execute("await models.list('classifier', None, 3)\nawait models.list()\nawait models.first('a', 'ignored')")
    assert result.ok and result.value == "a"
    assert seen == [["classifier", None, 3], []]


@pytest.mark.tonio
async def test_names_the_members_of_a_namespace_when_a_script_calls_one_that_does_not_exist(make_sandbox):
    async def execute(_args, _kwargs):
        return None

    result = await make_sandbox(
        globals=[CodemodeGlobal("models.classify", execute), CodemodeGlobal("models.generateImages", execute)],
        type_check=False,
    ).execute("await models.generateImage()")
    assert not result.ok
    assert result.error.message == "models.generateImage does not exist. Did you mean models.generateImages?"


@pytest.mark.tonio
async def test_rejects_invalid_and_reserved_global_names(pool):
    async def execute(_args, _kwargs):
        return None

    for name in (
        "a.b.c",
        "a.",
        ".a",
        "tools.x",
        "store.x",
        "a.not-valid",
        "not-valid",
        "tools",
        "print",
        "class",
        "a.class",
    ):
        with pytest.raises(ValueError, match="Invalid global"):
            CodemodeSandbox(pool, globals=[CodemodeGlobal(name, execute)])
    with pytest.raises(ValueError, match="conflicts with the namespace"):
        CodemodeSandbox(pool, globals=[CodemodeGlobal("models", execute), CodemodeGlobal("models.list", execute)])


# -- limits and lifetime -----------------------------------------------------


@pytest.mark.tonio
async def test_terminates_a_synchronous_infinite_loop_on_timeout(make_sandbox):
    result = await make_sandbox().execute("while True:\n    pass", timeout_ms=200)
    assert not result.ok
    assert result.error == CodemodeError(kind="timeout", message="Execution timed out after 200 ms")


@pytest.mark.tonio
async def test_tool_time_does_not_count_toward_the_execution_cap(make_sandbox):
    async def slow(_args):
        # Longer than the cap below: a script waiting on a tool is suspended,
        # and the cap counts execution only.
        await tonio.Event().wait(0.5)
        return "late"

    result = await make_sandbox([tool("slow", slow)], max_execution_secs=0.2).execute(
        "await tools.slow()", timeout_ms=math.inf
    )
    assert result.ok and result.value == "late"


@pytest.mark.tonio
async def test_stops_a_script_that_uses_up_its_execution_time_and_the_script_cannot_catch_it(make_sandbox):
    result = await make_sandbox(max_execution_secs=0.2).execute(
        "try:\n    while True:\n        pass\nexcept Exception:\n    text('caught')", timeout_ms=math.inf
    )
    assert not result.ok
    assert result.error == CodemodeError(
        kind="timeout",
        message="Execution timed out after 0.2 seconds of script execution (time spent waiting on tools does not count)",
    )
    assert result.output == ()


@pytest.mark.tonio
async def test_ending_while_a_call_is_pending_ends_the_script(make_sandbox):
    hang, _started, _cancelled = hanging_tool()
    result = await make_sandbox([hang], type_check=False).execute("tools.hang()\n'early'", timeout_ms=math.inf)
    assert result.ok and result.value == "early"
    assert statuses(result) == [("hang", "cancelled")]


@pytest.mark.tonio
async def test_aborts_via_cancel_and_cancels_in_flight_calls(make_sandbox):
    hang, started, cancelled = hanging_tool()
    sandbox = make_sandbox([hang])

    token = CancelToken()
    pending = tonio.spawn(sandbox.execute("await tools.hang()\n'never'", cancel=token))
    await started.wait(5)
    token.cancel(Exception("user cancelled"))
    result = await pending
    assert result.error == CodemodeError(kind="aborted", message="user cancelled")
    assert statuses(result) == [("hang", "cancelled")]
    await cancelled.wait(5)
    assert cancelled.is_set()

    # A token cancelled before the script starts: it never runs.
    cancelled_early = CancelToken()
    cancelled_early.cancel()
    early = await sandbox.execute("1", cancel=cancelled_early)
    assert early.error == CodemodeError(kind="aborted", message="Operation was aborted")


@pytest.mark.tonio
async def test_close_aborts_in_flight_executions_and_rejects_new_ones(make_sandbox):
    hang, started, _cancelled = hanging_tool()
    sandbox = make_sandbox([hang])
    pending = tonio.spawn(sandbox.execute("await tools.hang()\n'never'"))
    await started.wait(5)
    await sandbox.close()
    assert (await pending).error == CodemodeError(kind="aborted", message="Sandbox closed")
    with pytest.raises(RuntimeError, match="closed"):
        await sandbox.execute("1")


@pytest.mark.tonio
async def test_runs_executions_in_parallel_without_sharing_state(make_sandbox):
    sandbox = make_sandbox([echo], type_check=False)
    handles = [
        tonio.spawn(sandbox.execute("shared = 'a'\nawait tools.echo()\nshared")),
        tonio.spawn(sandbox.execute("shared = 'b'\nawait tools.echo()\nshared")),
        tonio.spawn(
            sandbox.execute(
                "try:\n    shared\n    found = 'defined'\nexcept NameError:\n    found = 'undefined'\nfound"
            )
        ),
    ]
    results = [await handle for handle in handles]
    assert [result.value for result in results] == ["a", "b", "undefined"]


@pytest.mark.tonio
async def test_turns_deep_recursion_into_a_catchable_recursion_error(make_sandbox):
    result = await make_sandbox().execute(
        """def dive(n):
    return dive(n + 1)
try:
    dive(0)
    outcome = 'no error'
except RecursionError as e:
    outcome = type(e).__name__
outcome"""
    )
    assert result.ok and result.value == "RecursionError"


@pytest.mark.tonio
async def test_reports_a_missing_monty_binary_as_a_sandbox_error():
    pool = await CodemodePool(binary_path="/nonexistent/monty")
    try:
        result = await CodemodeSandbox(pool).execute("1")
    finally:
        await pool.close()
    assert not result.ok
    assert result.error.kind == "sandbox"
    assert result.error.message.startswith("Failed to start the sandbox worker:")


@pytest.mark.tonio
async def test_reports_a_worker_killed_from_outside_mid_script_as_a_sandbox_error(make_sandbox):
    sandbox = None

    async def kill(_args):
        [execution] = sandbox._running
        os.kill(execution._pid, signal.SIGKILL)

    sandbox = make_sandbox([tool("kill", kill)])
    result = await sandbox.execute("await tools.kill()\n1")
    assert result.error == CodemodeError(kind="sandbox", message="The sandbox worker exited before the script settled")
    # The pool replaces the worker.
    assert (await sandbox.execute("2")).value == 2


# -- escape hatches ----------------------------------------------------------


@pytest.mark.tonio
async def test_has_no_host_access(make_sandbox):
    result = await make_sandbox(type_check=False).execute(
        """import asyncio
import os
import time
results = []

def attempt(fn):
    try:
        fn()
        results.append('ok')
    except Exception as e:
        results.append(type(e).__name__)

attempt(lambda: os.getenv('HOME'))
attempt(lambda: open('/etc/hostname'))
attempt(lambda: time.sleep(0.01))
try:
    await asyncio.sleep(0.01)
    results.append('ok')
except Exception as e:
    results.append(type(e).__name__)
try:
    import socket
    results.append('imported')
except ImportError as e:
    results.append(type(e).__name__)
try:
    import subprocess
    results.append('imported')
except ImportError as e:
    results.append(type(e).__name__)
results"""
    )
    assert result.ok, result.error
    assert "ok" not in result.value[:4]
    assert result.value[4:] == ["ModuleNotFoundError", "ModuleNotFoundError"]


@pytest.mark.tonio
async def test_keeps_eval_and_exec_inside_the_sandbox(make_sandbox):
    result = await make_sandbox(type_check=False).execute(
        """x = eval('1 + 1')
exec('y = 2')
try:
    eval("open('/etc/hostname')")
    reached = 'opened'
except Exception as e:
    reached = type(e).__name__
[x, y, reached]"""
    )
    assert result.ok, result.error
    assert result.value[:2] == [2, 2]
    assert result.value[2] != "opened"


@pytest.mark.tonio
async def test_rejects_imports_of_modules_that_are_not_present(make_sandbox):
    result = await make_sandbox(type_check=False).execute(
        "try:\n    import json5\n    found = 'imported'\nexcept ImportError as e:\n    found = type(e).__name__\nfound"
    )
    assert result.ok and result.value == "ModuleNotFoundError"


@pytest.mark.tonio
async def test_assignments_to_tools_affect_only_that_script(make_sandbox):
    sandbox = make_sandbox([echo], type_check=False)
    assert (await sandbox.execute("tools = None\ntools")).value is None
    assert (await sandbox.execute("await tools.echo(x=1)")).value == {"x": 1}


# -- pidrei additions --------------------------------------------------------


_TYPED_ECHO_SCHEMA = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}


@pytest.mark.tonio
async def test_a_script_rejected_by_the_type_check_makes_no_tool_calls(make_sandbox):
    seen = []

    async def record(args):
        seen.append(args)
        return args

    script = "await tools.echo(x='a')\nawait tools.echo(bad=1)"
    rejected = await make_sandbox([tool("echo", record, input_schema=_TYPED_ECHO_SCHEMA)]).execute(script)
    assert not rejected.ok
    assert rejected.error.stack.startswith(TYPE_CHECK_FAILED)
    assert "bad" in rejected.error.stack
    assert rejected.calls == ()
    assert seen == []
    # With the check off the same script runs, both calls included.
    unchecked = await make_sandbox([tool("echo", record, input_schema=_TYPED_ECHO_SCHEMA)], type_check=False).execute(
        script
    )
    assert unchecked.ok
    assert seen == [{"x": "a"}, {"bad": 1}]


@pytest.mark.tonio
async def test_rejects_a_bare_unawaited_call(make_sandbox):
    seen = []

    async def record(args):
        seen.append(args)

    result = await make_sandbox([tool("write", record)]).execute("tools.write(path='a')\n'done'")
    assert not result.ok
    assert result.error.stack.startswith(TYPE_CHECK_FAILED)
    assert "unused-awaitable" in result.error.stack
    assert seen == []


@pytest.mark.tonio
async def test_hints_at_all_settled_when_gather_gets_return_exceptions(make_sandbox):
    result = await make_sandbox([echo]).execute(
        "import asyncio\nawait asyncio.gather(tools.echo(x='a'), return_exceptions=True)"
    )
    assert not result.ok
    assert result.error.name == "NotImplementedError"
    assert result.error.stack.endswith(
        "Hint: asyncio.gather() does not support return_exceptions here. Use await all_settled(...) to keep the "
        "results of the calls that succeed."
    )


@pytest.mark.tonio
async def test_all_settled_keeps_the_results_of_the_calls_that_succeed(make_sandbox):
    result = await make_sandbox([echo, fail]).execute(
        "await all_settled(tools.echo(x=1), tools.fail(), tools.echo(x=2))"
    )
    assert result.ok
    assert result.value == [
        {"status": "fulfilled", "value": {"x": 1}},
        {"status": "rejected", "reason": "tool exploded"},
        {"status": "fulfilled", "value": {"x": 2}},
    ]


@pytest.mark.tonio
async def test_call_tool_calls_a_tool_by_name_like_tools(make_sandbox):
    result = await make_sandbox([echo]).execute("name = 'ec' + 'ho'\nawait call_tool(name, x=1)")
    assert result.ok and result.value == {"x": 1}
    assert statuses(result) == [("echo", "ok")]


@pytest.mark.tonio
async def test_refuses_positional_arguments_with_the_keyword_form(make_sandbox):
    seen = []

    async def read(args):
        seen.append(args)
        return ""

    schema = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
    sandbox = make_sandbox([tool("read", read, input_schema=schema)], type_check=False)
    direct = await sandbox.execute("await tools.read('a.py')")
    assert (direct.error.name, direct.error.message) == (
        "TypeError",
        "tools.read() takes keyword arguments only, for example tools.read(path=...).",
    )
    by_name = await sandbox.execute("await call_tool('read', 'a.py')")
    assert by_name.error.message == "call_tool() takes keyword arguments only, for example call_tool('read', path=...)."
    assert seen == []


@pytest.mark.tonio
async def test_drops_none_for_optional_parameters_that_do_not_accept_null(make_sandbox):
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "limit": {"type": "number"}, "tag": {"type": ["string", "null"]}},
        "required": ["path"],
    }
    result = await make_sandbox([tool("read", _echo, input_schema=schema)]).execute(
        "await tools.read(path='a', limit=None, tag=None)"
    )
    assert result.ok and result.value == {"path": "a", "tag": None}


@pytest.mark.tonio
async def test_merges_consecutive_print_output_into_one_text_item(make_sandbox):
    result = await make_sandbox().execute("print('a')\nprint('b')\ntext('c')\nprint('d')")
    assert result.ok
    assert result.output == (CodemodeTextItem("a\nb"), CodemodeTextItem("c"), CodemodeTextItem("d"))


@pytest.mark.tonio
async def test_allows_many_host_calls_per_script(make_sandbox):
    result = await make_sandbox().execute("for i in range(3000):\n    text(i)")
    assert result.ok
    assert len(result.output) == 3000
