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
     needed goes through tonio's public surface (see TONIO_BUGS.md).
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
    "pidrei/pidrei",
    "protocol/pidrei_protocol",
    "client/pidrei_client",
    "tui/pidrei_tui",
    "server/pidrei_server",
]

# Our package -> the pi package it ports, for the async/sync drift check.
PACKAGE_UPSTREAM = {
    "ai/pidrei_ai": "ai",
    "agent/pidrei_agent": "agent",
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
    "_load_scope",
    "_show_extension_selector",
    "_show_extension_editor",
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
    # pi's `execFile` variant. We run the subprocess synchronously on the
    # debounce thread, which is not a runtime worker, so there is nothing to
    # get off. Documented at both definitions.
    "FooterDataProvider._refresh_git_branch_async",
    "FooterDataProvider._resolve_git_branch_async",
    # pi's run-until-first-await prologue: a sync def returning a coroutine.
    "SessionSelectorComponent._load_scope",
    # pi's is async only to `await this.init()` lazily; pidrei always
    # subscribes after init, so there is nothing to await. Documented.
    "InteractiveMode._handle_event",
    # pi chains request promises; we chain spawned tasks on completion Events,
    # because a Python coroutine cannot be awaited twice. Same shape as
    # `SettingsManager._enqueue_write`. Documented at the definition.
    "Editor._start_autocomplete_request",
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

    for node in ast.walk(tree):
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


def main() -> int:
    findings: list[str] = []
    notes: list[str] = []
    paths = [path for package in PACKAGES for path in sorted((ROOT / "packages" / package).rglob("*.py"))]
    async_names, _sync_names = _collect_module_functions(paths)
    for path in paths:
        _check_file(path, findings, async_names)
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
