# Examples

Working extensions, runnable as they are:

```bash
pidrei -e <path-to-example>
```

## Extensions

| Example | Shows |
|---------|-------|
| [`extensions/trigger_compact.py`](extensions/trigger_compact.py) | Triggering compaction from an event handler |
| [`extensions/input_transform_streaming.py`](extensions/input_transform_streaming.py) | Rewriting user input; streaming output back |
| [`extensions/git_merge_and_resolve.py`](extensions/git_merge_and_resolve.py) | `pi.exec`, follow-up messages, keeping blocking I/O off the event loop |
| [`extensions/plan_mode/`](extensions/plan_mode/) | Flags, commands, shortcuts, tool gating, context filtering, widgets, custom message types — the widest use of the API |

`plan_mode` is the one to read if you only read one. It is also a directory
extension, so it shows the `__init__.py` layout and relative imports.

## Writing your own

[docs/extensions.md](../docs/extensions.md) documents the whole API. The short
version:

```python
def extension(pi):
    async def handle(_args, ctx):
        ctx.ui.notify("hello", "info")

    pi.register_command("hello", handler=handle, description="Say hello")
```

Save as `.pidrei/extensions/hello.py` in a project and it loads on next start.

## A note on style

pi keeps each example in a single factory closure, which is the JavaScript
idiom. These follow that where it stays readable, and use a class where it does
not — `plan_mode` is a `PlanMode` object with a `wire()` method, because a
closure holding that much state reads worse in Python and is harder to test.
Both forms are equally valid extensions.
