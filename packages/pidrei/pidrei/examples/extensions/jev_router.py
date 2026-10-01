"""Jev router - a virtual model that plans on a strong model and implements on a cheap one.

Registers `jev/auto`, which routes between three OpenAI Codex models:

- Planning: GPT-5.6 Sol for complex work, GPT-5.6 Terra otherwise. The Jev
  classifier rates the first user message; planning stays on the chosen model.
- Implementation: GPT-5.6 Luna.

The planning model explores, plans, and makes the first edit. After the first
successful `edit` or `write` tool call, the next request of the same turn goes
to Luna, and the session stays there. A session therefore switches models once
and accepts a single prompt-cache miss.

The phase is router state: pidrei stores it on the session branch, so it
follows the session tree and survives compaction. The selected thinking level
passes through as the reasoning effort of the chosen model. Requests outside
the agent loop, such as compaction summaries, go to Luna.

Requires TypeSafe credentials (TYPESAFE_API_KEY) and an OpenAI Codex login.
Usage:
    pidrei -e ./examples/extensions/jev_router.py --model jev/auto
"""

from pidrei.core.extensions import ExtensionVirtualModel
from pidrei.core.virtual_models import ModelRoute
from pidrei_ai.types import ClassifierChoiceQuestion, ClassifierContext, ClassifierOptions


PROVIDER = "openai-codex"
SOL = "gpt-5.6-sol"
TERRA = "gpt-5.6-terra"
LUNA = "gpt-5.6-luna"

# Tools whose successful result means implementation has started.
EDIT_TOOLS = {"edit", "write"}


def route_to(request, ctx, model_id: str, state: dict | None = None) -> ModelRoute:
    """Router state is `{"phase": "planning" | "implementation", "model": <OpenAI Codex model id>}`."""
    model = ctx.model_registry.find(PROVIDER, model_id)
    if model is None:
        raise Exception(f"Model {PROVIDER}/{model_id} is not in the catalog")
    return ModelRoute(model=model, thinking_level=request.thinking_level, state=state)


def last_user_text(messages) -> str:
    user_messages = [message for message in messages if message.role == "user"]
    content = user_messages[-1].content if user_messages else ""
    if isinstance(content, str):
        return content
    return "\n".join(block.text for block in content if block.type == "text")


def edited_this_turn(messages) -> bool:
    """Whether a tool call since the last user message edited a file successfully."""
    last_user = max((index for index, message in enumerate(messages) if message.role == "user"), default=-1)
    return any(
        message.role == "toolResult" and message.tool_name in EDIT_TOOLS and not message.is_error
        for message in messages[last_user + 1 :]
    )


async def choose_planning_model(request, ctx) -> str:
    """Planning model for a new session: Sol for complex work, Terra otherwise or when Jev is unavailable."""
    # Keep a planning model the session already uses, so switching to jev/auto costs no cache miss.
    previous = request.previous.model if request.previous is not None else None
    if previous is not None and previous.provider == PROVIDER and previous.id in (SOL, TERRA):
        return previous.id

    jev = ctx.model_registry.find_of_type("classifier", "typesafe", "jev-latest")
    if jev is None:
        return TERRA
    result = await ctx.model_registry.classify(
        jev,
        ClassifierContext(
            state={"prompt": last_user_text(request.messages)[:16_000]},
            questions={
                "complexity": ClassifierChoiceQuestion(
                    instructions="How demanding is the software engineering work requested in `prompt`?",
                    criteria={
                        "standard": "Ordinary features, fixes, reviews, or questions",
                        "complex": "Subtle design, cross-cutting changes, or hard debugging",
                    },
                )
            },
        ),
        ClassifierOptions(cancel=request.cancel),
    )
    answer = result.answers.get("complexity") if result.stop_reason == "stop" else None
    if answer is not None and answer.type == "choice" and answer.probabilities.get("complex", 0) >= 0.5:
        return SOL
    return TERRA


async def extension(pi):
    async def route(request, ctx) -> ModelRoute:
        if request.reason == "direct":
            return route_to(request, ctx, LUNA)
        state = request.state
        if state is None:
            model = await choose_planning_model(request, ctx)
            return route_to(request, ctx, model, {"phase": "planning", "model": model})
        # The planning model made the first edit: hand the rest of the work to Luna.
        if state["phase"] == "planning" and edited_this_turn(request.messages):
            return route_to(request, ctx, LUNA, {"phase": "implementation", "model": LUNA})
        return route_to(request, ctx, state["model"])

    pi.register_virtual_model(
        ExtensionVirtualModel(
            provider="jev",
            id="auto",
            name="Auto (Jev)",
            thinking_levels=["low", "medium", "high", "xhigh"],
            # Shared by all three models; shown before the first response.
            context_window=272_000,
            max_tokens=128_000,
            route=route,
        )
    )
