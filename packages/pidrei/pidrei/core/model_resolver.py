"""Mirror of pi coding-agent src/core/model-resolver.ts.

Model resolution, scoping, and initial selection. Glob matching implements
the minimatch subset used by model patterns (`*`/`?` do not cross `/`,
`**` does, `[...]` character classes, case-insensitive).
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Literal

from pidrei_ai.auth.types import AuthOperationOptions
from pidrei_ai.registry import models_are_equal
from pidrei_ai.types import Model

from ..cli.args import is_valid_thinking_level
from .defaults import DEFAULT_THINKING_LEVEL
from .output_guard import write_stderr, write_stdout


# Default chat model IDs for providers with built-in chat models.
DEFAULT_MODEL_PER_PROVIDER: dict[str, str] = {
    "amazon-bedrock": "us.anthropic.claude-opus-4-6-v1",
    "ant-ling": "Ring-2.6-1T",
    "anthropic": "claude-opus-4-8",
    "openai": "gpt-5.5",
    "azure": "gpt-5.4",
    "openai-codex": "gpt-6.1-sol",
    "nvidia": "nvidia/nemotron-3.5-super-vl-120b-a12b",
    "deepseek": "deepseek-v4-pro",
    "google": "gemini-3.1-pro-preview",
    "google-vertex": "gemini-3.1-pro-preview",
    "github-copilot": "gpt-6.1-sol",
    "openrouter": "moonshotai/kimi-k2.6",
    "vercel-ai-gateway": "zai/glm-5.1",
    "xai": "grok-4.7",
    "groq": "openai/gpt-oss-120b",
    "cerebras": "gpt-oss-120b",
    "zai": "glm-5.3",
    "zai-coding-cn": "glm-5.3",
    # Deviation: pi still names `devstral-medium-latest`, which models.dev
    # retired before the 0.84.3 catalog regen; a default that resolves to
    # nothing is worse than a one-id drift, so mistral points at its successor.
    "mistral": "mistral-medium-latest",
    "minimax": "MiniMax-M2.7",
    "minimax-cn": "MiniMax-M2.7",
    "moonshotai": "kimi-k2.6",
    "moonshotai-cn": "kimi-k2.6",
    "huggingface": "moonshotai/Kimi-K2.6",
    "fireworks": "accounts/fireworks/models/kimi-k3",
    "together": "moonshotai/Kimi-K3",
    "baseten": "zai-org/GLM-5.2",
    "opencode": "kimi-k2.6",
    "opencode-go": "kimi-k3",
    "kimi-coding": "kimi-for-coding",
    "meta": "muse-spark-1.3",
    "cloudflare-workers-ai": "@cf/moonshotai/kimi-k2.6",
    "cloudflare-ai-gateway": "workers-ai/@cf/moonshotai/kimi-k2.6",
    "qwen-token-plan": "qwen3.7-max",
    "qwen-token-plan-cn": "qwen3.7-max",
    "qwen-token-plan-individual": "qwen3.8-max",
    "xiaomi": "mimo-v2.6-pro",
    "xiaomi-token-plan-cn": "mimo-v2.6-pro",
    "xiaomi-token-plan-ams": "mimo-v2.6-pro",
    "xiaomi-token-plan-sgp": "mimo-v2.6-pro",
}


@dataclass(slots=True)
class ScopedModel:
    model: Model
    # Thinking level if explicitly specified in pattern (e.g., "model:high"), None otherwise
    thinking_level: str | None = None


def _minimatch(value: str, pattern: str) -> bool:
    """minimatch subset: `*`/`?` stop at `/`, `**` crosses, `[...]` classes, nocase."""
    regex = ""
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "*":
            if pattern[index : index + 2] == "**":
                regex += ".*"
                index += 2
                continue
            regex += "[^/]*"
        elif char == "?":
            regex += "[^/]"
        elif char == "[":
            end = pattern.find("]", index + 1)
            if end == -1:
                regex += re.escape(char)
            else:
                regex += pattern[index : end + 1]
                index = end + 1
                continue
        else:
            regex += re.escape(char)
        index += 1
    return re.fullmatch(regex, value, re.IGNORECASE) is not None


def _is_alias(id: str) -> bool:
    return id.endswith("-latest") or re.search(r"-\d{8}$", id) is None


# Model reference resolution. Candidates are collected before a provider is chosen, so catalog order
# never decides the provider and an ambiguous match is reported instead of falling through to a
# weaker one.


@dataclass(slots=True, kw_only=True)
class ModelReferenceRequest:
    # Bare model id or `provider/id`, optionally followed by `:<thinking>`.
    text: str
    # Only consider models from this provider.
    provider: str | None = None
    # A trailing `:<suffix>` is only considered when the full text matches no model, since ids such as
    # OpenRouter's `qwen/qwen3-coder:exacto` contain colons.
    # - `none`: the suffix stays part of the id
    # - `strict`: a valid thinking level is parsed; any other suffix stays part of the id
    # - `lenient`: like `strict`, but an invalid suffix is dropped with a warning
    thinking_suffix: Literal["none", "strict", "lenient"]
    # Match id and name substrings when no exact match exists.
    fuzzy: bool


@dataclass(slots=True)
class ModelReferenceTarget:
    provider: str
    # Non-empty; the first model is the one the provider prefix or constraint matched.
    provider_models: list[Model]
    # Text after the provider prefix as typed, including any `:<suffix>`.
    model_text: str
    # `model_text` without a valid thinking suffix (unless `thinking_suffix` is `none`).
    model_id: str
    thinking_level: str | None = None


@dataclass(slots=True)
class ModelReferenceResult:
    status: Literal["resolved", "ambiguous", "not-found"]
    # Set when resolved.
    model: Model | None = None
    # Set when ambiguous.
    candidates: list[Model] = field(default_factory=list)
    usable_count: int = 0
    thinking_level: str | None = None
    warning: str | None = None
    # Set on not-found when a provider is explicit or named by a `provider/` prefix, for building a
    # custom model id.
    target: ModelReferenceTarget | None = None


def _split_thinking_suffix(text: str) -> tuple[str, str] | None:
    colon_index = text.rfind(":")
    suffix = text[colon_index + 1 :]
    if colon_index == -1 or not is_valid_thinking_level(suffix):
        return None
    return text[:colon_index], suffix


def _split_provider_prefix(text: str, models: list[Model]) -> tuple[str, str] | None:
    slash_index = text.find("/")
    if slash_index == -1:
        return None
    prefix = text[:slash_index].strip().lower()
    model_text = text[slash_index + 1 :].strip()
    if not prefix or not model_text:
        return None
    provider = next((model.provider for model in models if model.provider.lower() == prefix), None)
    return (provider, model_text) if provider else None


def resolve_model_reference(
    request: ModelReferenceRequest,
    models: list[Model],
    usable: Callable[[str], bool],
) -> ModelReferenceResult:
    """`usable` only breaks ties between providers, so a unique match resolves even without
    credentials (`--api-key` first-time setup)."""
    text = request.text.strip()
    if not text:
        return ModelReferenceResult(status="not-found")
    constraint = request.provider.lower() if request.provider else None
    pool = [model for model in models if model.provider.lower() == constraint] if constraint else list(models)
    result = _resolve_text(text, request, pool, usable)
    if result.status != "not-found":
        return result

    split = _split_provider_prefix(text, pool)
    if constraint:
        anchor = pool[0] if pool else None
    else:
        anchor = next((model for model in pool if split and model.provider == split[0]), None)
    if anchor is None:
        return result
    model_text = split[1] if split else text
    suffix = None if request.thinking_suffix == "none" else _split_thinking_suffix(model_text)
    return ModelReferenceResult(
        status="not-found",
        target=ModelReferenceTarget(
            provider=anchor.provider,
            provider_models=[
                anchor,
                *(model for model in pool if model.provider == anchor.provider and model is not anchor),
            ],
            model_text=model_text,
            model_id=suffix[0] if suffix else model_text,
            thinking_level=suffix[1] if suffix else None,
        ),
    )


def find_exact_model_reference_match(model_reference: str, available_models: list[Model]) -> Model | None:
    """Exact `provider/id` or bare id lookup without thinking suffixes or fuzzy matching."""
    result = resolve_model_reference(
        ModelReferenceRequest(text=model_reference, thinking_suffix="none", fuzzy=False),
        available_models,
        lambda _provider: True,
    )
    return result.model if result.status == "resolved" else None


def _resolve_text(
    text: str,
    request: ModelReferenceRequest,
    pool: list[Model],
    usable: Callable[[str], bool],
) -> ModelReferenceResult:
    reference = _select_reference(text, pool, request.fuzzy, usable)
    if reference is not None:
        return reference

    if request.fuzzy:
        fuzzy = _select_fuzzy(_fuzzy_matches(text, pool), usable)
        if fuzzy is not None:
            return fuzzy

    if request.thinking_suffix == "none":
        return ModelReferenceResult(status="not-found")
    thinking = _split_thinking_suffix(text)
    if thinking is not None:
        base, thinking_level = thinking
        inner = _resolve_text(base, request, pool, usable)
        if inner.status == "not-found":
            return inner
        # Only use this thinking level if no warning from inner recursion
        return replace(inner, thinking_level=None if inner.warning else thinking_level)
    # Keep an invalid suffix as part of the id and fail, rather than resolve a different model
    if request.thinking_suffix == "strict":
        return ModelReferenceResult(status="not-found")
    colon_index = text.rfind(":")
    if colon_index == -1:
        return ModelReferenceResult(status="not-found")

    suffix = text[colon_index + 1 :]
    inner = _resolve_text(text[:colon_index], request, pool, usable)
    if inner.status == "not-found":
        return inner
    return replace(
        inner,
        thinking_level=None,
        warning=f'Invalid thinking level "{suffix}" in pattern "{text}". Using default instead.',
    )


def _select_reference(
    text: str,
    pool: list[Model],
    fuzzy: bool,
    usable: Callable[[str], bool],
) -> ModelReferenceResult | None:
    """`provider/rest` is only a hint: the whole text may also be a model id on another provider,
    such as gateway id `zai/glm-5` or commandcode id `xiaomi/mimo-v2.6-pro`. The named provider wins
    when usable, since that is what the syntax usually means; otherwise a sole usable raw match wins,
    since the named provider could not serve the request. An unusable named match is still preferred
    over unusable raw matches, so `--api-key` can set up the named provider."""
    lower_text = text.lower()
    raw = [model for model in pool if model.id.lower() == lower_text]
    split = _split_provider_prefix(text, pool)
    named: Model | None = None
    if split:
        provider, model_text = split
        provider_models = [model for model in pool if model.provider == provider]
        lower_model_text = model_text.lower()
        # The provider's own literal id (OpenRouter's `openrouter/free`) beats its fuzzy matches
        named = next((model for model in provider_models if model.id.lower() == lower_model_text), None)
        if named is None:
            named = next((model for model in raw if model.provider == provider), None)
        if named is None and fuzzy:
            named = _best_fuzzy_match(_fuzzy_matches(model_text, provider_models))

    others = [model for model in raw if model is not named]
    candidates = [named, *others] if named is not None else others
    if not candidates:
        return None
    if len({model.provider for model in candidates}) == 1:
        return ModelReferenceResult(status="resolved", model=candidates[0])
    if named is not None and usable(named.provider):
        return ModelReferenceResult(status="resolved", model=named)
    usable_others = [model for model in others if usable(model.provider)]
    if len(usable_others) == 1:
        return ModelReferenceResult(status="resolved", model=usable_others[0])
    if not usable_others and named is not None:
        return ModelReferenceResult(status="resolved", model=named)
    return ModelReferenceResult(status="ambiguous", candidates=candidates, usable_count=len(usable_others))


def _fuzzy_matches(text: str, models: list[Model]) -> list[Model]:
    lower_text = text.lower()
    return [
        model
        for model in models
        if lower_text in model.id.lower() or (model.name is not None and lower_text in model.name.lower())
    ]


def _select_fuzzy(matches: list[Model], usable: Callable[[str], bool]) -> ModelReferenceResult | None:
    if not matches:
        return None
    multiple_providers = len({model.provider for model in matches}) > 1
    usable_matches = [model for model in matches if usable(model.provider)] if multiple_providers else []
    pool = usable_matches or matches
    best = _best_fuzzy_match(pool)
    if best is None:
        return None
    # The same id on several providers would otherwise be decided by catalog order
    tied = [model for model in pool if model.id == best.id]
    if len({model.provider for model in tied}) == 1:
        return ModelReferenceResult(status="resolved", model=best)
    return ModelReferenceResult(
        status="ambiguous",
        candidates=tied,
        usable_count=len(tied) if usable_matches else 0,
    )


def _best_fuzzy_match(matches: list[Model]) -> Model | None:
    """Prefer aliases (`claude-sonnet-4-5`) over dated versions (`claude-sonnet-4-5-20250929`), then
    the latest."""
    aliases = [model for model in matches if _is_alias(model.id)]
    return max(aliases or matches, key=lambda model: model.id, default=None)


def _build_fallback_model(provider_models: list[Model], model_id: str) -> Model:
    default_id = DEFAULT_MODEL_PER_PROVIDER.get(provider_models[0].provider)
    base_model = (
        next((model for model in provider_models if model.id == default_id), provider_models[0])
        if default_id
        else provider_models[0]
    )

    return replace(base_model, id=model_id, name=model_id)


@dataclass(slots=True)
class ModelScopeDiagnostic:
    type: str
    code: str  # "no-match" | "invalid-thinking-level" | "ambiguous"
    message: str
    pattern: str


@dataclass(slots=True)
class ResolveModelScopeResult:
    scoped_models: list[ScopedModel]
    diagnostics: list[ModelScopeDiagnostic]


def resolve_model_scope_from_models(
    patterns: list[str],
    models: list[Model],
) -> ResolveModelScopeResult:
    """Resolve scope patterns (`--models`, `enabledModels`) to models with optional thinking levels.

    Every model in `models` is usable, so credentials cannot choose between providers offering the
    same id. All of them are added, with a warning so the user can pick one with `provider/id`.
    """
    scoped_models: list[ScopedModel] = []
    diagnostics: list[ModelScopeDiagnostic] = []

    def add(model: Model, thinking_level: str | None) -> None:
        if not any(models_are_equal(scoped.model, model) for scoped in scoped_models):
            scoped_models.append(ScopedModel(model=model, thinking_level=thinking_level))

    def no_match(pattern: str) -> None:
        diagnostics.append(
            ModelScopeDiagnostic(
                type="warning", code="no-match", message=f'No models match pattern "{pattern}"', pattern=pattern
            )
        )

    def ambiguous(pattern: str, candidates: list[Model]) -> None:
        providers = sorted({model.provider for model in candidates})
        diagnostics.append(
            ModelScopeDiagnostic(
                type="warning",
                code="ambiguous",
                message=(
                    f'Pattern "{pattern}" matches several providers: {", ".join(providers)}. '
                    "Use provider/id to choose one."
                ),
                pattern=pattern,
            )
        )

    for pattern in patterns:
        # Check if pattern contains glob characters
        if "*" in pattern or "?" in pattern or "[" in pattern:
            # Extract optional thinking level suffix (e.g., "provider/*:high")
            thinking = _split_thinking_suffix(pattern)
            glob_pattern, thinking_level = thinking if thinking else (pattern, None)

            # Ids such as `bracketed-model[1m]` are not globs
            exact = resolve_model_reference(
                ModelReferenceRequest(text=glob_pattern, thinking_suffix="none", fuzzy=False),
                models,
                lambda _provider: True,
            )
            if exact.status == "resolved":
                matches = [exact.model]
            elif exact.status == "ambiguous":
                matches = exact.candidates
                ambiguous(pattern, matches)
            else:
                # Match against "provider/modelId" format OR just model ID
                # This allows "*sonnet*" to match without requiring "anthropic/*sonnet*"
                matches = [
                    model
                    for model in models
                    if _minimatch(f"{model.provider}/{model.id}", glob_pattern) or _minimatch(model.id, glob_pattern)
                ]
            if not matches:
                no_match(pattern)
            for model in matches:
                add(model, thinking_level)
            continue

        result = resolve_model_reference(
            ModelReferenceRequest(text=pattern, thinking_suffix="lenient", fuzzy=True),
            models,
            lambda _provider: True,
        )
        if result.status == "not-found":
            no_match(pattern)
            continue
        if result.warning:
            diagnostics.append(
                ModelScopeDiagnostic(
                    type="warning", code="invalid-thinking-level", message=result.warning, pattern=pattern
                )
            )
        if result.status == "ambiguous":
            ambiguous(pattern, result.candidates)
        for model in [result.model] if result.status == "resolved" else result.candidates:
            add(model, result.thinking_level)

    return ResolveModelScopeResult(scoped_models=scoped_models, diagnostics=diagnostics)


async def resolve_model_scope_with_diagnostics(
    patterns: list[str],
    model_runtime,
    options: AuthOperationOptions | None = None,
) -> ResolveModelScopeResult:
    return resolve_model_scope_from_models(patterns, list(await model_runtime.get_available(None, options)))


def _warn(message: str) -> None:
    write_stderr(f"\x1b[33mWarning: {message}\x1b[0m\n")


async def resolve_model_scope(
    patterns: list[str], model_runtime, options: AuthOperationOptions | None = None
) -> list[ScopedModel]:
    result = await resolve_model_scope_with_diagnostics(patterns, model_runtime, options)
    for diagnostic in result.diagnostics:
        _warn(diagnostic.message)
    return result.scoped_models


@dataclass(slots=True)
class ResolveCliModelResult:
    model: Model | None
    thinking_level: str | None = None
    warning: str | None = None
    # Error message suitable for CLI display. When set, model is None.
    error: str | None = None


def resolve_cli_model(
    *,
    cli_provider: str | None = None,
    cli_model: str | None = None,
    cli_thinking: str | None = None,
    # `--api-key` was supplied for the selected provider.
    inline_api_key: bool = False,
    model_runtime,
) -> ResolveCliModelResult:
    """Resolve a single model from CLI flags (--provider/--model, fuzzy rules)."""
    if not cli_model:
        return ResolveCliModelResult(model=None)

    # Important: use *all* models here, not just models with pre-configured auth.
    # This allows "--api-key" to be used for first-time setup.
    models = list(model_runtime.get_models())
    if not models:
        return ResolveCliModelResult(
            model=None,
            error="No models available. Check your installation or add models to models.json.",
        )

    provider = (
        next((model.provider for model in models if model.provider.lower() == cli_provider.lower()), None)
        if cli_provider
        else None
    )
    if cli_provider and not provider:
        return ResolveCliModelResult(
            model=None,
            error=f'Unknown provider "{cli_provider}". Use --list-models to see available providers/models.',
        )

    result = resolve_model_reference(
        ModelReferenceRequest(text=cli_model, provider=provider, thinking_suffix="strict", fuzzy=True),
        models,
        # An inline key goes to whichever provider is selected, so stored credentials must not choose it
        (lambda _provider: False) if inline_api_key else model_runtime.configured_auth_check(),
    )
    if result.status == "resolved":
        return ResolveCliModelResult(model=result.model, thinking_level=result.thinking_level, warning=result.warning)
    if result.status == "ambiguous":
        matches = ", ".join(sorted(f"{model.provider}/{model.id}" for model in result.candidates))
        if inline_api_key:
            auth_hint = "--api-key does not choose a provider."
        elif result.usable_count == 0:
            auth_hint = "No matching provider is authenticated."
        else:
            auth_hint = "More than one matching provider is authenticated."
        return ResolveCliModelResult(
            model=None,
            error=(
                f'Model "{cli_model}" is ambiguous across providers: {matches}. '
                f"{auth_hint} Use --provider or provider/model."
            ),
        )

    target = result.target
    if target is None:
        return ResolveCliModelResult(
            model=None,
            error=f'Model "{cli_model}" not found. Use --list-models to see available models.',
        )

    # Use the thinking level suffix only when --thinking is not explicitly provided; otherwise it
    # stays part of the custom model id.
    # e.g. "zai-org/GLM-5.1-FP8:high" → model_id="zai-org/GLM-5.1-FP8", fallback_thinking="high"
    model_id = target.model_text if cli_thinking else target.model_id
    fallback_thinking = None if cli_thinking else target.thinking_level
    fallback_model = _build_fallback_model(target.provider_models, model_id)
    requested_thinking = cli_thinking or fallback_thinking
    model = (
        replace(fallback_model, reasoning=True)
        if requested_thinking and requested_thinking != "off"
        else fallback_model
    )
    return ResolveCliModelResult(
        model=model,
        thinking_level=fallback_thinking,
        warning=f'Model "{model_id}" not found for provider "{target.provider}". Using custom model id.',
    )


@dataclass(slots=True)
class InitialModelResult:
    model: Model | None
    thinking_level: str
    fallback_message: str | None = None


async def find_initial_model(
    *,
    scoped_models: list[ScopedModel],
    is_continuing: bool,
    default_provider: str | None = None,
    default_model_id: str | None = None,
    default_thinking_level: str | None = None,
    model_thinking_levels: dict[str, str] | None = None,
    model_runtime,
) -> InitialModelResult:
    """Find the initial model: scoped models, saved default, then the first
    available model with valid auth."""
    # 1. Use first model from scoped models (skip if continuing/resuming)
    if scoped_models and not is_continuing:
        scoped_model = scoped_models[0]
        per_model = (model_thinking_levels or {}).get(f"{scoped_model.model.provider}/{scoped_model.model.id}")
        thinking = scoped_model.thinking_level or per_model or default_thinking_level or DEFAULT_THINKING_LEVEL
        return InitialModelResult(model=scoped_model.model, thinking_level=thinking)

    # 2. Try saved default from settings if auth is configured.
    if default_provider and default_model_id:
        found = model_runtime.get_model(default_provider, default_model_id)
        if found is not None and model_runtime.has_configured_auth(found.provider):
            per_model = (model_thinking_levels or {}).get(f"{default_provider}/{default_model_id}")
            thinking = per_model or default_thinking_level or DEFAULT_THINKING_LEVEL
            return InitialModelResult(model=found, thinking_level=thinking)

    # 3. Try first available model with valid API key
    available_models = list(model_runtime.get_available_snapshot())

    if available_models:
        for provider, default_id in DEFAULT_MODEL_PER_PROVIDER.items():
            match = next(
                (model for model in available_models if model.provider == provider and model.id == default_id), None
            )
            if match is not None:
                return InitialModelResult(model=match, thinking_level=DEFAULT_THINKING_LEVEL)

        return InitialModelResult(model=available_models[0], thinking_level=DEFAULT_THINKING_LEVEL)

    # 4. No model found
    return InitialModelResult(model=None, thinking_level=DEFAULT_THINKING_LEVEL)


async def restore_model_from_session(
    saved_provider: str,
    saved_model_id: str,
    current_model: Model | None,
    should_print_messages: bool,
    model_runtime,
) -> tuple[Model | None, str | None]:
    """Restore model from session, with fallback to available models."""
    restored_model = model_runtime.get_model(saved_provider, saved_model_id)

    has_configured_auth = (
        model_runtime.has_configured_auth(restored_model.provider) if restored_model is not None else False
    )

    if restored_model is not None and has_configured_auth:
        if should_print_messages:
            write_stdout(f"Restored model: {saved_provider}/{saved_model_id}\n")
        return restored_model, None

    reason = "model no longer exists" if restored_model is None else "no auth configured"

    if should_print_messages:
        _warn(f"Could not restore model {saved_provider}/{saved_model_id} ({reason}).")

    if current_model is not None:
        if should_print_messages:
            write_stdout(f"Falling back to: {current_model.provider}/{current_model.id}\n")
        return current_model, (
            f"Could not restore model {saved_provider}/{saved_model_id} ({reason}). "
            f"Using {current_model.provider}/{current_model.id}."
        )

    available_models = list(model_runtime.get_available_snapshot())

    if available_models:
        fallback_model: Model | None = None
        for provider, default_id in DEFAULT_MODEL_PER_PROVIDER.items():
            match = next(
                (model for model in available_models if model.provider == provider and model.id == default_id), None
            )
            if match is not None:
                fallback_model = match
                break

        if fallback_model is None:
            fallback_model = available_models[0]

        if should_print_messages:
            write_stdout(f"Falling back to: {fallback_model.provider}/{fallback_model.id}\n")

        return fallback_model, (
            f"Could not restore model {saved_provider}/{saved_model_id} ({reason}). "
            f"Using {fallback_model.provider}/{fallback_model.id}."
        )

    return None, None
