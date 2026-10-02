"""Static checks that the mirrored suites structurally cannot catch.

Promoted from the debugging scratchpad after Phase 4.5: every defect that
kept interactive mode from booting was invisible to unit tests because the
tests drove methods against hand-built fakes. These checks look at the code
itself instead.

  1. `await self.m()` where `m` is a sync method — legal in JS (`await
     undefined`), a TypeError here. Sync methods that *return* an awaitable
     (pi's run-until-first-await prologue pattern) are exempted by name via
     ALLOWED_SYNC_AWAITS.
  2. bare `self.m(...)` / `m(...)` statements where the callee is `async def`
     — a coroutine created and dropped, so the work never happens.
  3. `tonio._*` imports in package source — private runtime API; anything
     needed goes through tonio's public surface.
  4. pi is `async` where we are `def` — the shape with something to lose in
     translation. `SettingsManager._enqueue_write` was flattened from pi's
     promise chain to an inline write, which silently changed `reload()`
     semantics and made an unrelated design problem look unsolvable for a
     whole session. Needs a pi checkout (`PIDREI_UPSTREAM_CHECKOUT`) and is
     skipped without one. Matching is class-qualified (`Class.method`):
     matching bare names collides across unrelated classes and gave ~60 false
     positives, which is a check nobody would trust. Known-and-justified pairs
     live in `JUSTIFIED_SYNC_PORTS`, each with a reason — "it was ported that
     way" is not one.
  5. `self.x` reads with no matching definition on the class — the port's
     public/private name drift (`self._show_new_version_notification` vs the
     defined `show_new_version_notification`), which only bites on the code
     path that happens to run. Classes with a non-local base or dynamic
     `setattr(self, ...)` are skipped rather than guessed at.
  6. probing whether a value is awaitable — `hasattr(x, "__await__")`,
     `inspect.isawaitable`/`iscoroutine`/`iscoroutinefunction`,
     `asyncio.iscoroutine`, `isinstance(x, Awaitable | Coroutine)`. Every
     callback contract is async-only (pi's `T | Promise<T>` unions are a
     JS-ism); a probe is how a union creeps back in during a port. Probes
     that *refuse* an awaitable (synchronous-only contracts) are allowed by
     enclosing function in ALLOWED_AWAITABLE_PROBES, each with a reason.
  7. the blocking-I/O shape. Anything doing I/O the runtime has primitives
     for (fs, net, subprocesses, pipes, fds) is `async def` or returns an
     awaitable; the blocking calls themselves live only in functions named
     `*_blocking`, which run on the pool and nowhere else:
       - a stdlib blocking call (`open`, `os.stat`, `subprocess.run`,
         `socket.socket`, `time.sleep`, pathlib I/O on a `Path(...)`, ...)
         outside a `*_blocking` function or a lambda handed to
         `spawn_blocking`;
       - a call to a `*_blocking` function from anywhere else (pool-only code
         run on the runtime);
       - an `async def *_blocking`.
     Calls resolve through each file's imports to qualified stdlib names, so
     unrelated same-named methods never match. I/O hidden inside third-party
     calls is out of its reach.

Run via `make audit`. Exit code 1 on any finding.
"""

import ast
import os
import pathlib
import re
import sys


ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGES = [
    "ai/pidrei_ai",
    "agent/pidrei_agent",
    "codemode/pidrei_codemode",
    "pidrei/pidrei",
    "protocol/pidrei_protocol",
    "client/pidrei_client",
    "tui/pidrei_tui",
    "server/pidrei_server",
    "utils/pidrei_utils",
]

# Our package -> the pi package it ports, for the async/sync drift check.
PACKAGE_UPSTREAM = {
    "ai/pidrei_ai": "ai",
    "agent/pidrei_agent": "agent",
    "codemode/pidrei_codemode": "codemode",
    "pidrei/pidrei": "coding-agent",
    "protocol/pidrei_protocol": "protocol",
    "client/pidrei_client": "client",
    "tui/pidrei_tui": "tui",
    "server/pidrei_server": "server",
}

# Sync methods that deliberately return a coroutine: pi's async methods run
# synchronously up to their first await, and these mirror that by splitting a
# sync prologue from the awaited remainder.
ALLOWED_SYNC_AWAITS = {
    "_show_extension_selector",
    "_show_extension_editor",
    # server: queues the frame synchronously (wire ordering decided at call
    # time) and returns the coroutine that waits for it to go out. Documented
    # at the definition.
    "_send_message",
}

# Sync-prologue methods whose returned awaitable is a runtime-driven Deferred:
# the awaited remainder is already spawned before the method returns, so
# dropping the return value is pi's `void this.foo()` and loses no work. A
# dropped *coroutine* would silently lose its work — never list a method here
# unless its remainder is spawned (`driven(...)`/settled Deferreds only).
DROPPABLE_AWAITABLES = {
    "_disconnect",
    "_fail_protocol",
}

# `Class.method` pairs where pi is `async` and we are deliberately not.
# Each needs a reason; "it was ported that way" is not one. Entries that stop
# matching pi are dead weight — prune them rather than leaving them to rot.
JUSTIFIED_SYNC_PORTS = {
    # Synchronous handlers (spec/ui-island.md, "Input"): pi's part before its
    # first await runs in the caller (a keypress, a selector callback) under
    # the UI state lock, and the handler spawns the rest itself, as pi's
    # caller `void`s the promise. Callers never wait for these.
    "SessionSelectorComponent._load_scope",
    "SessionSelectorComponent._confirm_rename",
    "SessionSelectorComponent._refresh_sessions_after_mutation",
    "InteractiveMode._handle_follow_up",
    "InteractiveMode._flush_compaction_queue",
    "InteractiveMode._handle_model_command",
    "InteractiveMode.handle_clone_command",
    "InteractiveMode._handle_login_command",
    "InteractiveMode._start_provider_login",
    "InteractiveMode._show_api_key_login_dialog",
    "InteractiveMode._show_login_dialog",
    "InteractiveMode.handle_import_command",
    "InteractiveMode._handle_copy_command",
    "InteractiveMode._handle_clear_command",
    "InteractiveMode.handle_compact_command",
    # pi's is async only to `await this.init()` lazily; pidrei always
    # subscribes after init, so there is nothing to await. Documented.
    "InteractiveMode._handle_event",
    # pi chains request promises; we chain spawned tasks on completion Events,
    # because a Python coroutine cannot be awaited twice. Same shape as
    # `SettingsManager._enqueue_write`. Documented at the definition.
    "Editor._start_autocomplete_request",
    # pi's async body queues the frame before its first await, deciding wire
    # order at call time; the sync prologue keeps that ordering on tonio
    # (a lazily-started coroutine would decide it at schedule time).
    # Documented at the definition.
    "PiServer._send_message",
}


def _is_self_call(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _collect_module_functions(paths: list[pathlib.Path]) -> tuple[set[str], set[str]]:
    """Module-level function names across the packages, split by asyncness.

    Names defined both ways somewhere (rare) are dropped from the async set:
    the check is a heuristic and must not fire on an ambiguous name.
    """
    async_names: set[str] = set()
    sync_names: set[str] = set()
    for path in paths:
        tree = ast.parse(path.read_text(), str(path))
        for node in tree.body:
            if isinstance(node, ast.AsyncFunctionDef):
                async_names.add(node.name)
            elif isinstance(node, ast.FunctionDef):
                sync_names.add(node.name)
    return async_names - sync_names, sync_names


# Probes that refuse an awaitable (enforcing synchronous-only) rather than
# accepting both colours, by enclosing function (`Class.method` or a module
# function name). Each with a reason.
ALLOWED_AWAITABLE_PROBES = {
    # What runs under the UI state lock must not await: `apply` and the
    # extension UI contexts refuse an awaitable result (spec/ui-island.md,
    # "`ctx.ui`").
    "call_sync",
    # Extension timers refuse an async callback when the timer is created,
    # not at its first fire (spec/ui-island.md, "The `tui` extensions receive").
    "ExtensionTui._guarded",
}

_AWAITABLE_PROBE_FUNCTIONS = {"isawaitable", "iscoroutine", "iscoroutinefunction"}
_AWAITABLE_TYPES = {"Awaitable", "Coroutine"}


def _awaitable_probe(call: ast.Call) -> str | None:
    """The source of `call` when it tests whether a value is awaitable (check 6)."""
    func = call.func
    name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
    if name in _AWAITABLE_PROBE_FUNCTIONS:
        return ast.unparse(call)
    if (
        name == "hasattr"
        and len(call.args) == 2
        and isinstance(call.args[1], ast.Constant)
        and call.args[1].value == "__await__"
    ):
        return ast.unparse(call)
    if name == "isinstance" and len(call.args) == 2:
        types = call.args[1]
        candidates = types.elts if isinstance(types, ast.Tuple) else [types]
        for candidate in candidates:
            type_name = (
                candidate.attr
                if isinstance(candidate, ast.Attribute)
                else candidate.id
                if isinstance(candidate, ast.Name)
                else None
            )
            if type_name in _AWAITABLE_TYPES:
                return ast.unparse(call)
    return None


def _allowed_probe_nodes(tree: ast.Module) -> set[int]:
    """The ids of every node inside a function named in ALLOWED_AWAITABLE_PROBES."""
    allowed: set[int] = set()
    functions = (ast.FunctionDef, ast.AsyncFunctionDef)
    for node in tree.body:
        scopes = []
        if isinstance(node, functions):
            scopes.append((node.name, node))
        elif isinstance(node, ast.ClassDef):
            scopes.extend((f"{node.name}.{item.name}", item) for item in node.body if isinstance(item, functions))
        for name, scope in scopes:
            if name in ALLOWED_AWAITABLE_PROBES:
                allowed.update(id(inner) for inner in ast.walk(scope))
    return allowed


def _check_file(path: pathlib.Path, findings: list[str], imported_async: set[str]) -> None:
    source = path.read_text()
    tree = ast.parse(source, str(path))
    rel = path.relative_to(ROOT)

    module_async = {node.name for node in tree.body if isinstance(node, ast.AsyncFunctionDef)}
    # Only consider imported names this file actually pulled in, so a
    # same-named local helper elsewhere cannot cause a false positive.
    file_imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            file_imports.update(alias.asname or alias.name for alias in node.names)
    module_async |= file_imports & imported_async

    allowed_probes = _allowed_probe_nodes(tree)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and id(node) not in allowed_probes
            and (probe := _awaitable_probe(node)) is not None
        ):
            findings.append(f"{rel}:{node.lineno}: `{probe}` probes for an awaitable (callbacks are async-only)")
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(("tonio._", "tonio.colored._")):
            findings.append(f"{rel}:{node.lineno}: imports private tonio API `{node.module}`")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(("tonio._", "tonio.colored._")):
                    findings.append(f"{rel}:{node.lineno}: imports private tonio API `{alias.name}`")

    for cls in [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]:
        methods: dict[str, str] = {}
        for item in cls.body:
            if isinstance(item, ast.FunctionDef):
                # A sync def annotated `-> Awaitable[...]` is deliberately
                # awaitable: it returns the awaitable rather than adding a
                # coroutine frame (PLAN: no single-`return await` wrappers).
                returns = ast.unparse(item.returns) if item.returns is not None else ""
                methods[item.name] = "awaitable" if returns.startswith("Awaitable[") else "sync"
            elif isinstance(item, ast.AsyncFunctionDef):
                methods[item.name] = "async"

        for node in ast.walk(cls):
            if isinstance(node, ast.Await):
                name = _is_self_call(node.value)
                if name and methods.get(name) == "sync" and name not in ALLOWED_SYNC_AWAITS:
                    findings.append(f"{rel}:{node.lineno}: `await self.{name}()` but `{name}` is a sync method")
            if isinstance(node, ast.Expr):
                name = _is_self_call(node.value)
                if name is None:
                    continue
                if methods.get(name) == "awaitable" and name in DROPPABLE_AWAITABLES:
                    continue
                if methods.get(name) in ("async", "awaitable") or (name not in methods and name in module_async):
                    findings.append(f"{rel}:{node.lineno}: `{name}(...)` is async but its coroutine is dropped")

        _check_self_attributes(cls, rel, tree, findings)


def _class_defined_names(cls: ast.ClassDef) -> set[str] | None:
    """Every `self.x` the class defines, or None when it cannot be known."""
    names: set[str] = set()
    for node in ast.walk(cls):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node in cls.body:
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and node in cls.body:
                    names.add(target.id)
                elif (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    names.add(target.attr)
                elif isinstance(target, ast.Tuple):
                    for element in target.elts:
                        if (
                            isinstance(element, ast.Attribute)
                            and isinstance(element.value, ast.Name)
                            and element.value.id == "self"
                        ):
                            names.add(element.attr)
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and node in cls.body:
                names.add(target.id)
            elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                names.add(target.attr)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "setattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "self"
        ):
            # Dynamic attributes: only literal names are knowable.
            if isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                names.add(node.args[1].value)
            else:
                return None
    return names


def _check_self_attributes(cls: ast.ClassDef, rel: pathlib.Path, tree: ast.Module, findings: list[str]) -> None:
    local_classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    defined: set[str] = set()
    pending = [cls]
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current.name in seen:
            continue
        seen.add(current.name)
        names = _class_defined_names(current)
        if names is None:
            return  # dynamic setattr somewhere in the hierarchy
        defined |= names
        for base in current.bases:
            if isinstance(base, ast.Name) and base.id in local_classes:
                pending.append(local_classes[base.id])
            elif not (isinstance(base, ast.Name) and base.id == "object"):
                return  # base defined elsewhere: its attributes are unknown

    for node in ast.walk(cls):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Load)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr not in defined
            and not node.attr.startswith("__")
        ):
            findings.append(f"{rel}:{node.lineno}: `self.{node.attr}` is never defined on `{cls.name}`")


def _camel_to_snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


_TS_CLASS_RE = re.compile(r"^(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)")
_TS_ASYNC_RE = re.compile(
    r"^\s+(?:private |public |protected |static |readonly |override )*async\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("
)


def _pi_async_methods(pi_root: pathlib.Path, pi_package: str) -> set[str]:
    """`Class.method` for every `async` method in one pi package.

    Class-qualified on purpose: matching bare method names collides across
    unrelated classes (`create`, `resolve`, `stop`) and buries the signal.
    Class names are identical between pi and the port, so they compare
    directly; only the method needs snake-casing.
    """
    methods: set[str] = set()
    src = pi_root / "packages" / pi_package / "src"
    if not src.is_dir():
        return methods
    for path in src.rglob("*.ts"):
        current: str | None = None
        for line in path.read_text(encoding="utf-8").splitlines():
            class_match = _TS_CLASS_RE.match(line)
            if class_match:
                current = class_match.group(1)
                continue
            if line.startswith("}"):
                current = None
                continue
            if current is None:
                continue
            async_match = _TS_ASYNC_RE.match(line)
            if async_match:
                methods.add(f"{current}.{_camel_to_snake(async_match.group(1))}")
    return methods


def _check_sync_ports_of_pi_async(findings: list[str], notes: list[str]) -> None:
    pi_dir = os.environ.get("PIDREI_UPSTREAM_CHECKOUT")
    if not pi_dir or not pathlib.Path(pi_dir).is_dir():
        notes.append("pi async/sync drift: skipped (set PIDREI_UPSTREAM_CHECKOUT to enable)")
        return
    pi_root = pathlib.Path(pi_dir)
    drift: list[str] = []

    for ours, theirs in PACKAGE_UPSTREAM.items():
        pi_async = _pi_async_methods(pi_root, theirs)
        if not pi_async:
            # Either the pi package is absent, or it genuinely has no async
            # methods to drift from (pi's `protocol` is all pure functions).
            notes.append(f"pi async/sync drift: no async methods in pi package {theirs!r}, skipped")
            continue
        for path in sorted((ROOT / "packages" / ours).rglob("*.py")):
            rel = path.relative_to(ROOT)
            for cls in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(cls, ast.ClassDef):
                    continue
                for item in cls.body:
                    if not isinstance(item, ast.FunctionDef):
                        continue  # an async def cannot have drifted
                    # pi's names carry no underscore prefix, so the comparison
                    # strips ours; the allowlist keys stay as written here.
                    if f"{cls.name}.{item.name.lstrip('_')}" not in pi_async:
                        continue
                    if f"{cls.name}.{item.name}" in JUSTIFIED_SYNC_PORTS:
                        continue
                    returns = ast.unparse(item.returns) if item.returns is not None else ""
                    if returns.startswith("Awaitable["):
                        continue  # deliberately awaitable, just not a coroutine
                    drift.append(f"{rel}:{item.lineno}: `{cls.name}.{item.name}` is sync but pi's is `async`")

    findings.extend(drift)


# Operations on an already-open descriptor (`os.read`/`os.write`, the flag
# and tty calls) are not listed: that is readiness-driven I/O, non-blocking by
# definition. What is listed reaches the filesystem by path, waits on a child,
# or runs a process.
_OS_BLOCKING = (
    "stat", "lstat", "statvfs", "access", "listdir", "scandir", "walk", "fwalk",
    "mkdir", "makedirs", "remove", "unlink", "rmdir", "removedirs", "rename", "renames", "replace",
    "chmod", "chown", "lchown", "utime", "symlink", "readlink", "link", "truncate",
    "open", "pread", "pwrite", "sendfile", "fsync", "fdatasync", "chdir",
    "system", "popen", "wait", "waitpid", "wait3", "wait4", "mkfifo", "mknod",
)  # fmt: skip
_OS_PATH_BLOCKING = (
    "exists", "lexists", "isdir", "isfile", "islink", "ismount",
    "getsize", "getmtime", "getatime", "getctime", "realpath", "samefile",
)  # fmt: skip

# Qualified stdlib callables that block: I/O the runtime has primitives for,
# and the sleep.
_BLOCKING_CALLS = {
    "open",
    "input",
    "io.open",
    "time.sleep",
    "glob.glob",
    "glob.iglob",
    "urllib.request.urlopen",
    "ssl.create_default_context",
    "sqlite3.connect",
    "webbrowser.open",
    "webbrowser.open_new",
    "webbrowser.open_new_tab",
    *(f"os.{name}" for name in _OS_BLOCKING),
    *(f"os.path.{name}" for name in _OS_PATH_BLOCKING),
    *(
        f"subprocess.{name}"
        for name in ("run", "Popen", "call", "check_call", "check_output", "getoutput", "getstatusoutput")
    ),
    *(
        f"shutil.{name}"
        for name in (
            "copy",
            "copy2",
            "copyfile",
            "copyfileobj",
            "copymode",
            "copystat",
            "copytree",
            "rmtree",
            "move",
            "which",
            "disk_usage",
            "chown",
            "make_archive",
            "unpack_archive",
        )
    ),
    *(
        f"socket.{name}"
        for name in (
            "socket",
            "create_connection",
            "create_server",
            "getaddrinfo",
            "gethostbyname",
            "gethostbyaddr",
            "getfqdn",
        )
    ),
    "select.select",
    "pty.spawn",
    *(
        f"tempfile.{name}"
        for name in (
            "mkstemp",
            "mkdtemp",
            "NamedTemporaryFile",
            "TemporaryFile",
            "SpooledTemporaryFile",
            "TemporaryDirectory",
            "gettempdir",
        )
    ),
}
# pathlib methods that touch the filesystem, matched only on a receiver that
# is syntactically a `pathlib.Path` (not `tonio.colored.fs.Path`, whose
# methods are awaited pool calls).
_PATHLIB_BLOCKING = {
    "exists", "is_dir", "is_file", "is_symlink", "is_mount", "stat", "lstat", "owner", "group",
    "read_text", "read_bytes", "write_text", "write_bytes", "iterdir", "glob", "rglob", "walk",
    "mkdir", "rmdir", "unlink", "touch", "open", "rename", "replace", "resolve", "chmod",
    "samefile", "readlink", "symlink_to", "hardlink_to",
}  # fmt: skip
_PATHLIB_TYPES = {"pathlib.Path", "pathlib.PosixPath", "pathlib.WindowsPath"}
_POOL_ENTRY_POINTS = {"spawn_blocking", "map_blocking"}

# Functions (`file:qualname`, relative to `packages/`) whose blocking calls
# only ever run where there is no runtime yet. Each with a reason.
ALLOWED_OFF_RUNTIME_BLOCKING = {
    # The fallback when no stdio writer is running: the writer starts first
    # thing in `_run_main`, so this path is the pre-runtime start-up.
    "pidrei/pidrei/core/output_guard.py:_write",
    # Both take the inline path only on `RuntimeNotInitializedError`: there
    # is no runtime at all.
    "pidrei/pidrei/utils/open_browser.py:open_browser",
    "pidrei/pidrei/utils/temp_file_writer.py:TempFileWriter.__init__",
}


def _import_aliases(tree: ast.Module) -> dict[str, str]:
    """Local name -> qualified module path, from absolute imports anywhere in the file."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    aliases[alias.asname] = alias.name
                else:
                    root = alias.name.split(".")[0]
                    aliases[root] = root
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return aliases


def _qualify(node: ast.AST, aliases: dict[str, str], shadowed: set[str]) -> str | None:
    if isinstance(node, ast.Name):
        if node.id in aliases:
            return aliases[node.id]
        if node.id in ("open", "input") and node.id not in shadowed:
            return node.id
        return None
    if isinstance(node, ast.Attribute):
        base = _qualify(node.value, aliases, shadowed)
        return f"{base}.{node.attr}" if base else None
    return None


def _callee_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _is_pool_entry(call: ast.AST) -> bool:
    return isinstance(call, ast.Call) and _callee_name(call) in _POOL_ENTRY_POINTS


def _enclosing(node: ast.AST, parents: dict[int, ast.AST]):
    current = parents.get(id(node))
    while current is not None:
        yield current
        current = parents.get(id(current))


def _in_pool_context(node: ast.AST, parents: dict[int, ast.AST]) -> bool:
    """Inside a `*_blocking` function, or a lambda handed straight to the pool;
    the nearest `async def` ends the search (a coroutine is runtime code)."""
    for ancestor in _enclosing(node, parents):
        if isinstance(ancestor, ast.AsyncFunctionDef):
            return False
        if isinstance(ancestor, ast.FunctionDef) and ancestor.name.endswith("_blocking"):
            return True
        if isinstance(ancestor, ast.Lambda):
            call = parents.get(id(ancestor))
            if _is_pool_entry(call) and call.args and call.args[0] is ancestor:
                return True
    return False


def _function_label(node: ast.AST, parents: dict[int, ast.AST]) -> str:
    names = [
        ancestor.name
        for ancestor in _enclosing(node, parents)
        if isinstance(ancestor, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    ]
    return ".".join(reversed(names)) or "<module>"


def _is_path_expr(node: ast.AST, aliases: dict[str, str], shadowed: set[str], path_names: set[str]) -> bool:
    if isinstance(node, ast.Call):
        return _qualify(node.func, aliases, shadowed) in _PATHLIB_TYPES
    if isinstance(node, ast.Name):
        return node.id in path_names
    if isinstance(node, ast.Attribute) and node.attr in ("parent", "parents"):
        return _is_path_expr(node.value, aliases, shadowed, path_names)
    if isinstance(node, ast.Subscript):
        return _is_path_expr(node.value, aliases, shadowed, path_names)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _is_path_expr(node.left, aliases, shadowed, path_names)
    return False


def _check_blocking_io(path: pathlib.Path, findings: list[str]) -> None:
    tree = ast.parse(path.read_text(), str(path))
    rel = path.relative_to(ROOT)
    aliases = _import_aliases(tree)
    # Only module-level names shadow a builtin: a method called `open` is an
    # attribute, and must not hide the file's real `open(...)` calls.
    shadowed = {node.name for node in tree.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)} | {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    parents: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node

    # Names bound to a `pathlib.Path(...)`, per enclosing function.
    path_names: dict[int, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _is_path_expr(node.value, aliases, shadowed, set()):
            scope = next(
                (a for a in _enclosing(node, parents) if isinstance(a, ast.FunctionDef | ast.AsyncFunctionDef)), tree
            )
            path_names.setdefault(id(scope), set()).update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )

    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name.endswith("_blocking"):
            findings.append(f"{rel}:{node.lineno}: `async def {node.name}`: `*_blocking` functions run on the pool")
        if not isinstance(node, ast.Call):
            continue
        if not any(
            isinstance(ancestor, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda)
            for ancestor in _enclosing(node, parents)
        ):
            continue  # module or class body: import time, out of scope
        label = _function_label(node, parents)
        if f"{path.relative_to(ROOT / 'packages')}:{label}" in ALLOWED_OFF_RUNTIME_BLOCKING:
            continue
        callee = _callee_name(node)
        qualified = _qualify(node.func, aliases, shadowed)
        # The suffix is ours: stdlib names that merely end in it
        # (`os.set_blocking`) are judged by the blocking-call list below.
        is_stdlib = qualified is not None and qualified.split(".")[0] in sys.stdlib_module_names
        if callee and callee.endswith("_blocking") and callee not in _POOL_ENTRY_POINTS and not is_stdlib:
            if not _in_pool_context(node, parents):
                findings.append(
                    f"{rel}:{node.lineno}: `{callee}(...)` runs pool-only code on the runtime (in `{label}`)"
                )
            continue
        blocking = None
        if qualified in _BLOCKING_CALLS:
            blocking = qualified
        elif isinstance(node.func, ast.Attribute) and node.func.attr in _PATHLIB_BLOCKING:
            scope = next(
                (a for a in _enclosing(node, parents) if isinstance(a, ast.FunctionDef | ast.AsyncFunctionDef)), tree
            )
            if _is_path_expr(node.func.value, aliases, shadowed, path_names.get(id(scope), set())):
                blocking = f"pathlib.Path.{node.func.attr}"
        if blocking and not _in_pool_context(node, parents):
            findings.append(
                f"{rel}:{node.lineno}: blocking `{blocking}` outside a `*_blocking` function (in `{label}`)"
            )


def main() -> int:
    findings: list[str] = []
    notes: list[str] = []
    paths = [path for package in PACKAGES for path in sorted((ROOT / "packages" / package).rglob("*.py"))]
    async_names, _sync_names = _collect_module_functions(paths)
    for path in paths:
        _check_file(path, findings, async_names)
        _check_blocking_io(path, findings)
    _check_sync_ports_of_pi_async(findings, notes)

    for note in notes:
        print(f"note: {note}")
    if notes:
        print()
    for finding in findings:
        print(finding)
    print(f"\n{len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
