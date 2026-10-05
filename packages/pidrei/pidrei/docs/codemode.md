# Codemode

The `codemode` tool lets the model write a Python script that calls pidrei's other tools and runs non-LLM models, such as classifiers and image models. Only the script's output reaches the model, so a script can run calls in parallel and filter large results before the model sees them. To turn it on, see [Enable codemode](cli.md#enable-codemode).

pi's codemode scripts are JavaScript. pidrei's are Python, run on [Monty](https://github.com/pydantic/monty), so everything below differs from pi's codemode where Python does.

## Scripts

The tool input is raw Python source, not JSON and not a markdown code fence. It runs in a sandboxed interpreter: top-level `await` works, and if the last line is an expression, its value is added to the output. Scripts have no third-party packages, file system, network, environment, subprocesses, or sleep; they reach the outside world only through tools and `models`.

The interpreter supports a subset of Python. Not supported: `match`, `yield`, `del`, class inheritance (so no custom exception classes), method decorators, `async with` and `async for`, `asyncio.create_task`, and `asyncio.gather(return_exceptions=True)` (use `all_settled()`). These standard library modules can be imported, each in part: `asyncio`, `base64`, `binascii`, `collections`, `copy`, `dataclasses`, `datetime`, `functools`, `itertools`, `json`, `math`, `os`, `pathlib`, `random`, `re`, `sys`, `time`, `typing`, and `unicodedata`. Operations that would reach the system, such as reading a file, the environment, or `time.sleep()`, raise an error.

Every script is type-checked against the declarations of the tools and globals before it runs, and a script that fails the check does not run: the result lists the problems and says that no tool calls were made. The checker is strict. For example, a value that may be `None` must be checked before it is used:

```python
model = await models.get_model_of_type("classifier", "typesafe", "jev-latest")
assert model is not None
```

Set `codemode.typeCheck` to `false` (see [Settings](#settings)) to run scripts unchecked.

A script may start with an options line:

```python
# @options: {"max_output_tokens": 2000, "timeout_ms": 60000}
```

- `max_output_tokens` (default 10000) limits the output. Longer output keeps its start and end, and the full text is written to a temp file whose path is included in the result. A script fails when its output passes 16777216 characters of text and base64 image data or 100000 `text()` and `image()` calls; write large data to a file with a tool instead.
- `timeout_ms` is a hard deadline for the whole script, time spent waiting on tools included. It is unset by default. Image generation can take minutes, so do not set a short deadline for scripts that generate images.

The result starts with `Script completed` or `Script failed`, the wall time, and the output. A failed script keeps its partial output, followed by `Script error:` and the error, which for an exception is a Python traceback with the script's line numbers. Tool calls are real: calls made before a failure are not undone. Calls still running when the script ends are cancelled, including calls that were never awaited.

## Globals

| Global | Purpose |
|---|---|
| `tools.<name>(...)` | Call a tool. See [Call tools](#call-tools). |
| `text(value)` | Add a text item to the output. Strings, numbers, booleans and `None` are added as their string form, other values as JSON. |
| `image(value)` | Add an image to the output: a base64 `data:` URL, an `{'image_url': ...}` dict, or an image block `{'type': 'image', 'data': ..., 'mimeType': ...}` such as those returned by MCP tools and `models.generate_images()`. Remote URLs are not supported. PNG, JPEG, GIF, and WebP are accepted. |
| `print(...)` | Add a text item to the output. Consecutive prints become one item. |
| last line | If the script's last line is an expression, its value is added like `text()`. `None` adds nothing. |
| `exit()` | End the script successfully. |
| `store(key, value)` / `load(key)` | Keep small JSON values across `codemode` calls. See [Store values](#store-values). |
| `await all_settled(*calls)` | Wait for every call and return `{'status': 'fulfilled', 'value': ...}` or `{'status': 'rejected', 'reason': ...}` for each, in order. `asyncio.gather()` raises on the first failure instead. |
| `ALL_TOOLS` | Every callable tool as `{'name': ..., 'description': ...}`, including tools the description does not list. |
| `has_tool(name)` | Whether a tool with this identifier can be called. |
| `await call_tool(name, **args)` | Call a tool whose identifier is computed at run time; the same as `await tools.<name>(**args)`. |
| `await search_tools(query, limit=8, namespace=None)` | Rank callable tools by relevance (BM25). Returns a list of `{'name': ..., 'description': ...}`. |
| `await describe_tool(name)` | A tool's description and Python declaration, or `None`. |
| `await describe_namespace(name)` | `{'name', 'description'?, 'instructions'?, 'tools'}` for a namespace such as an MCP server, or `None`. |
| `models` | List and run non-LLM models. See [Models](#models). |

## Call tools

Every tool the session can call is an async method of `tools`, named by its identifier: characters that are not valid in a Python identifier become `_`, and a name that is a Python keyword gets a trailing `_`, so the MCP tool `mcp__docs-site__search` is `tools.mcp__docs_site__search`. Tools take their arguments as keyword arguments only:

```python
source = await tools.read(path="src/app.py", limit=50)
```

A tool with a parameter whose name is not a valid identifier takes its arguments as a dict: `await tools.search(**{'max-results': 5})`. `has_tool()`, `call_tool()` and `describe_tool()` take the identifier too, the form `ALL_TOOLS` and `search_tools()` return.

What a call returns depends on the tool:

- Tools with an output schema return a structured value, usually a `dict`. `bash` returns `{'output', 'truncated', 'full_output_path'?, 'exit_code', 'wall_time_seconds'}`, also for non-zero exit codes. Its `output` is not limited to the 2000 lines or 50KB the model sees: it holds up to 1 MiB, and longer output keeps its first and last 512 KiB around an omission marker, with `truncated` set and the full output in `full_output_path`.
- MCP tools return their `CallToolResult`, including `isError` and `structuredContent`.
- Other tools, such as `read`, `edit`, and `write`, return their text output.

Keys inside returned values keep their wire spelling (`structuredContent`, `mimeType`, `stopReason`).

A call that fails, is blocked, or gets invalid arguments raises a `RuntimeError` that carries the tool's error text; catch it with `except Exception`. Use `all_settled()` to keep the results of the calls that succeed:

```python
results = await all_settled(tools.read(path="a.py"), tools.read(path="b.py"))
for result in results:
    if result["status"] == "fulfilled":
        text(result["value"][:200])
    else:
        text(f"failed: {result['reason']}")
```

The `codemode` description lists tools with their Python declarations, grouped by namespace (for example one MCP server). Tools with `deferred` exposure, which includes MCP tools with the default `codemode` exposure, are not listed, so the description stays the same while MCP servers connect. Listed declarations share a budget of 3000 estimated tokens (`codemode.inlineBudget`). Scripts find the other tools with `search_tools()`, `describe_tool()`, `describe_namespace()`, or by filtering `ALL_TOOLS`.

While `codemode` is active, `codemode.mode` decides how the other tools are presented. With `on` (default) declared tools stay declared, and their descriptions say how to call them from scripts. With `only` they are hidden from the model and listed in the `codemode` description instead, so the model calls them through scripts.

## Store values

`store(key, value)` keeps a JSON value under a string key for later `codemode` calls; `store(key, None)` deletes the key. `load(key)` returns the value, or `None`. Writes are kept only when the script succeeds: each successful script that stores values appends a `codemode-store` custom entry to the session, so resumed sessions keep the values and each branch sees only the values written on its path.

The store is for small state such as IDs, cursors, or summaries. One value may have at most 262144 characters of JSON and all values together at most 1048576. Do not store image data; show images with `image()` or write them to a file with a tool.

## Models

`models` reaches the model catalog and runs non-LLM models with the session's credentials: classifiers, which answer typed questions about JSON state, and image models, which generate images. Chat models are listed but cannot be run from scripts. Which classifier and image models exist is described in [Classifier models](models.md#classifier-models).

```python
type ModelType = Literal["chat", "image", "classifier"]


class ModelRef(TypedDict):
    provider: str
    id: str


# A catalog entry. `provider` and `id` identify it; other keys depend on the type.
class ModelInfo(TypedDict):
    type: NotRequired[ModelType]
    provider: str
    id: str
    name: str
    api: str
    input: list[Literal["text", "image"]]
    contextWindow: NotRequired[int]


class Models:
    # Every known model of a type, optionally for one provider.
    async def get_models_of_type(self, type: ModelType, provider: str | None = None) -> list[ModelInfo]: ...
    # Models of a type whose provider has working credentials.
    async def get_available_of_type(self, type: ModelType, provider: str | None = None) -> list[ModelInfo]: ...
    # One catalog entry, or None.
    async def get_model_of_type(self, type: ModelType, provider: str, id: str) -> ModelInfo | None: ...
    # Answer `context['questions']` about `context['state']`; answers are in `result['answers']` by question ID.
    async def classify(self, model: ModelRef, context: ClassifierContext) -> ClassifierResult: ...
    # Generate images from `context['input']` text and image blocks; show `result['output']` blocks with image(). Can take minutes.
    async def generate_images(self, model: ModelRef, context: ImagesContext) -> ImagesResult: ...


models: Models
```

`classify()` and `generate_images()` use only the `provider` and `id` of `model`, so `{'provider': ..., 'id': ...}` works as well as a catalog entry. They do not raise on provider errors: check `stopReason` and `errorMessage`. At most four such calls run at once per script; more calls wait for a free slot, so `asyncio.gather()` over many items is fine. Their usage is added to the `codemode` tool result and counts toward the session cost.

Model IDs differ between providers, for example `typesafe/jev-latest` and `openrouter/typesafe/jev-1.13`. Use `models.get_available_of_type(type)` to find the IDs that work with the current credentials.

### Classify

```python
class ClassifierContext(TypedDict):
    # The data to classify.
    state: dict[str, Any]
    # Questions by ID. One call answers all of them.
    questions: dict[str, ChoiceQuestion | ScoreQuestion | BoolQuestion]


# Pick one label. `criteria` maps each label to what it means.
class ChoiceQuestion(TypedDict):
    type: Literal["choice"]
    instructions: str
    criteria: dict[str, str]


# Score on an ordered scale. `criteria` describes each level, lowest first.
class ScoreQuestion(TypedDict):
    type: Literal["score"]
    instructions: str
    criteria: list[str]


# Yes or no.
class BoolQuestion(TypedDict):
    type: Literal["bool"]
    instructions: str
    criteria: BoolCriteria


BoolCriteria = TypedDict("BoolCriteria", {"true": str, "false": str})


class ClassifierResult(TypedDict):
    provider: str
    model: str
    # Answers by question ID.
    answers: dict[str, ChoiceAnswer | ScoreAnswer | BoolAnswer]
    usage: NotRequired[ModelUsage]
    stopReason: Literal["stop", "error", "aborted"]
    errorMessage: NotRequired[str]


class ChoiceAnswer(TypedDict):
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float]
    confidence: float


# `score` is the expected level index, from 0 to len(criteria) - 1.
class ScoreAnswer(TypedDict):
    type: Literal["score"]
    score: float
    confidence: float


# `probability` is the probability of true.
class BoolAnswer(TypedDict):
    type: Literal["bool"]
    probability: float


# Token counts and cost in USD, when the service reports them.
class ModelUsage(TypedDict):
    input: int
    output: int
    totalTokens: int
    cost: ModelUsageCost


class ModelUsageCost(TypedDict):
    total: float
```

An answer is one of three types, so check its `type` before reading the fields of one of them. Classify several items by calling `classify()` once per item. This script sorts feedback messages, for example ones a tool returned earlier in the script:

```python
import asyncio

jev = await models.get_model_of_type("classifier", "typesafe", "jev-latest")
assert jev is not None
results = await asyncio.gather(
    *[
        models.classify(
            jev,
            {
                "state": {"message": message},
                "questions": {
                    "sentiment": {
                        "type": "choice",
                        "instructions": "How does the user feel about the product?",
                        "criteria": {
                            "positive": "Satisfied or happy",
                            "negative": "Unhappy or frustrated",
                            "neutral": "Neither",
                        },
                    },
                    "urgency": {
                        "type": "score",
                        "instructions": "How urgently does this need a reply?",
                        "criteria": ["no reply needed", "reply this week", "reply today"],
                    },
                },
            },
        )
        for message in messages
    ]
)
rows = []
for message, result in zip(messages, results):
    if result["stopReason"] != "stop":
        rows.append({"message": message, "error": result.get("errorMessage")})
        continue
    sentiment = result["answers"]["sentiment"]
    urgency = result["answers"]["urgency"]
    if sentiment["type"] == "choice" and urgency["type"] == "score":
        rows.append({"message": message, "sentiment": sentiment["choice"], "urgency": urgency["score"]})
rows
```

### Generate images

```python
class ImagesContext(TypedDict):
    # The prompt as text blocks, plus image blocks to edit or use as references.
    input: list[TextBlock | ImageBlock]


class ImagesResult(TypedDict):
    provider: str
    model: str
    # Generated images, and text blocks for models that also return text.
    output: list[TextBlock | ImageBlock]
    usage: NotRequired[ModelUsage]
    stopReason: Literal["stop", "error", "aborted"]
    errorMessage: NotRequired[str]


class TextBlock(TypedDict):
    type: Literal["text"]
    text: str


# `data` is base64.
class ImageBlock(TypedDict):
    type: Literal["image"]
    data: str
    mimeType: str
```

Show generated images with `image(block)`. Do not add `data` to the output with `text()`, `print()`, or the last line: it is large and the model cannot read it as text. Generated images are not saved to disk; to keep one, write it to a file with a tool.

```python
# @options: {"timeout_ms": 300000}
painter = await models.get_model_of_type("image", "openrouter", "google/gemini-2.5-flash-image")
assert painter is not None
result = await models.generate_images(
    painter,
    {
        "input": [{"type": "text", "text": "A red fox in the snow, watercolor"}],
    },
)
if result["stopReason"] != "stop":
    text(result.get("errorMessage"))
    exit()
for block in result["output"]:
    if block["type"] == "image":
        image(block)
    else:
        text(block["text"])
```

## Settings

The `codemode` section of `settings.json`:

| Setting | Type | Default | Description |
|---|---|---|---|
| `codemode.mode` | `"on"` \| `"only"` | `"on"` | How the `codemode` tool presents tools while it is active. `on`: declared tools get a note on calling them from scripts appended to their description, and `codemode` lists only tools that are not declared. `only`: `codemode` lists every tool scripts can call, and active built-in and extension tools are hidden from the model, so it reaches them through `codemode`. |
| `codemode.inlineBudget` | number | `3000` | Estimated tokens (characters / 4) the `codemode` tool's description may spend on tool declarations. Tools that do not fit are left out and found with `search_tools()`. `0` lists only namespaces. |
| `codemode.typeCheck` | boolean | `true` | Type-check scripts before they run. pidrei only. With `false`, a mistake such as an unknown keyword argument fails only when the call is reached, after the calls before it have run. |

## Limits

- A script has 256 MB of memory. Running out ends the script with a `MemoryError` it cannot catch; filter or aggregate large data instead of accumulating it.
- A script may spend at most 60 seconds executing, not counting time spent waiting on tools and models. pi has no such limit. `timeout_ms` sets a deadline that includes the waiting.
- Deep recursion raises a `RecursionError`, which the script can catch.
- Scripts cannot start other `codemode` scripts.
