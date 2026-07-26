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
  4. `self.x` reads with no matching definition on the class — the port's
     public/private name drift (`self._show_new_version_notification` vs the
     defined `show_new_version_notification`), which only bites on the code
     path that happens to run. Classes with a non-local base or dynamic
     `setattr(self, ...)` are skipped rather than guessed at.

Run via `make audit`. Exit code 1 on any finding.
"""

import ast
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGES = ["ai/pidrei_ai", "agent/pidrei_agent", "pidrei/pidrei", "tui/pidrei_tui", "server/pidrei_server"]

# Sync methods that deliberately return a coroutine: pi's async methods run
# synchronously up to their first await, and these mirror that by splitting a
# sync prologue from the awaited remainder.
ALLOWED_SYNC_AWAITS = {
    "_load_scope",
    "_show_extension_selector",
    "_show_extension_editor",
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
                methods[item.name] = "sync"
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
                if methods.get(name) == "async" or (name not in methods and name in module_async):
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


def main() -> int:
    findings: list[str] = []
    paths = [path for package in PACKAGES for path in sorted((ROOT / "packages" / package).rglob("*.py"))]
    async_names, _sync_names = _collect_module_functions(paths)
    for path in paths:
        _check_file(path, findings, async_names)

    for finding in findings:
        print(finding)
    print(f"\n{len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
