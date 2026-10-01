# Virtual Models

A virtual model is a selectable model that picks a physical model for each
request. Use one to route by task, cost, or conversation state. For example, a
router can send quick questions to a small model and hard problems to a large
one, while the user selects a single model.

Register virtual models from an [extension](extensions.md). They appear in
`/model`, `--model`, scoped models, and settings like any other model. A
virtual model can be listed under any provider, including one with physical
models, such as `openai-codex/auto`.

## Selection and dispatch

A virtual model selects a model and a thinking level. A router maps that pair
to a physical pair for each request:

```
selected (virtual model, virtual level)  ->  dispatched (physical model, physical level)
jev/auto:low                             ->  anthropic/claude-sonnet-4-5:high
```

The virtual thinking level is an input to the router. Its meaning is up to the
router; it need not correspond to a reasoning budget.

pidrei keeps the two pairs apart:

| | Selection | Dispatch |
|---|---|---|
| Recorded in | `model_change` and `thinking_level_change` entries | Each assistant message: `provider`, `api`, `model`, `thinkingLevel` |
| Visible as | `ctx.model`, `ctx.thinking_level`, `PIDREI_MODEL`, `PIDREI_REASONING_LEVEL`, `/model` | The assistant message of each response |

Providers only receive physical models. Assistant messages name the physical
model, so replaying a conversation across different physical models works the
same as after a manual model switch. Resuming a session restores the virtual
selection from its latest `model_change` entry. If the virtual model is no
longer registered, pidrei falls back to the physical model that answered last.

In interactive mode, the footer shows the routed model next to the selection,
for example `auto • high → gpt-5.6-luna • medium`. `/session` lists the cost for
each physical model.

Context usage uses the limits of the physical model that produced the latest
response, even if that response came before switching to the virtual model.
Without such a response, it uses the limits declared on the virtual model, if
any. Compaction checks the same limits, and again the limits of the model each
request is routed to. If that model's context window is too small for the
conversation, pidrei compacts before sending the request; the route stays as
the router chose it.

## Register a virtual model

```python
from pidrei.core.extensions import ExtensionVirtualModel
from pidrei.core.virtual_models import ModelRoute


async def extension(pi):
    async def route(request, ctx):
        # Tool follow-ups and retries stay on the model that handled the turn.
        sticky = request.failed or request.previous
        if request.reason != "user" and sticky is not None:
            return ModelRoute(model=sticky.model, thinking_level=sticky.thinking_level or "medium")
        model_id = "claude-sonnet-4-5" if request.thinking_level == "high" else "claude-haiku-4-5"
        return ModelRoute(model=ctx.model_registry.find("anthropic", model_id), thinking_level="medium")

    pi.register_virtual_model(
        ExtensionVirtualModel(provider="router", id="auto", name="Auto", thinking_levels=["low", "high"], route=route)
    )
```

- `provider` is the provider the model is listed under. It can be any provider
  ID. A provider can list several virtual models next to its physical ones. On
  a physical provider, the virtual model is available when that provider has
  credentials. Under an ID that no provider uses, it is always available.
- `id` must not be the ID of a physical model of that provider. If a catalog
  refresh later adds a physical model with the same ID, the virtual model hides
  it.
- `thinking_levels` lists the levels offered for selection. It defaults to
  `["off"]`.
- `context_window` and `max_tokens` are shown before the first response. Unset
  limits are unknown.
- `input` lists the input types offered for selection. It defaults to text and
  images; physical models without image support receive placeholders.

Registration follows the same queuing and reload rules as
`pi.register_provider()`. Registering the same provider and ID again replaces
the virtual model. `pi.unregister_virtual_model(provider, id)` removes it;
`pi.unregister_provider()` does not. SDK code can register one without an
extension: `model_runtime.register_virtual_model(VirtualModelDefinition(...))`,
whose `route(request)` takes no context.

## Route requests

`route(request, ctx)` is async and runs before every request made with the
virtual model; it returns a `ModelRoute(model=..., thinking_level=...)`. The
model can be any physical model in the catalog whose provider has credentials;
look it up with `ctx.model_registry`. A virtual model cannot route to another
virtual model. pidrei clamps the thinking level to the returned model.

| Field | Meaning |
|---|---|
| `model`, `thinking_level` | The selected virtual model and level |
| `reason` | Why the request is made, see below |
| `previous` | Physical model and thinking level of the latest successful response in `messages` |
| `failed` | For `"retry"`: physical model, thinking level, and assistant `message` of the failed request, which `messages` no longer contains. The message carries `stop_reason` and `error_message`. None when routing itself failed |
| `state` | Router state last returned on this session branch, see below |
| `messages` | The conversation for this request, including system messages |
| `cancel` | Cancel token of the request |

| `reason` | Request |
|---|---|
| `"user"` | First request after a message the user wrote, including steering and follow-up messages |
| `"continuation"` | Any other request in the agent loop, such as after tool results or extension messages |
| `"retry"` | Automatic retry after a failed request, including after compaction for a context overflow |
| `"direct"` | Request made outside the agent loop, such as a compaction summary or an extension calling `ctx.model_registry.stream_simple()` |

Returning `previous` for `"continuation"` and `failed` for `"retry"` keeps
prompt caches and thinking signatures valid. Switching models between turns is
allowed but loses the prompt cache. A retry can also switch to another model,
for example when `failed.message.error_message` reports that a provider is
overloaded or the context overflowed.

If `route()` raises, or returns a virtual model or a model without credentials,
the request ends with an error response.

## Keep routing state

`route()` can return `state` next to the model. pidrei stores it on the session
branch and passes it back as `request.state` on later requests. Use it for
decisions the transcript does not record, such as classifier results or a
routing phase:

```python
async def route(request, ctx):
    state = request.state or {"phase": "plan"}
    model_id = "claude-opus-4-5" if state["phase"] == "plan" else "claude-haiku-4-5"
    return ModelRoute(model=ctx.model_registry.find("anthropic", model_id), thinking_level="medium", state=state)
```

- State must be JSON-serializable. Returning None or `request.state` itself
  keeps the current state.
- pidrei stores any other returned object as new state, before the request is
  sent, even when it equals the current state (strings and numbers compare by
  value). Return a new object only when the state changes. The state stays
  stored if the request later fails.
- State follows the session tree, so forks and `/tree` navigation see the
  state of their branch. It survives compaction. It is stored as a custom entry
  with `customType` `pi.virtual-model-state` and `data`
  `{"provider", "modelId", "state"}`.
- `"direct"` requests have no state, and pidrei ignores state they return.

The transcript already records the selection and every dispatched model, and
`ctx.session_manager.get_branch()` exposes both.

Routers can call other models through `ctx.model_registry`, for example
`ctx.model_registry.classify()` with a classifier model from
`ctx.model_registry.find_of_type("classifier", provider, id)`. The call adds
latency before the first token of the turn.

See the `jev_router.py` example extension for a complete router. It plans on a
strong OpenAI Codex model chosen by the Jev classifier, lets that model make
the first edit, and then switches once to a cheaper model, accepting a single
prompt-cache miss. It keeps the phase as router state.
