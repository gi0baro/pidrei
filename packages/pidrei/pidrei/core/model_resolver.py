"""Mirror of pi coding-agent src/core/model-resolver.ts.

Model resolution, scoping, and initial selection. Glob matching implements
the minimatch subset used by model patterns (`*`/`?` do not cross `/`,
`**` does, `[...]` character classes, case-insensitive).
"""

import re
import sys
from dataclasses import dataclass, replace

from pidrei_ai.auth.types import AuthOperationOptions
from pidrei_ai.registry import models_are_equal
from pidrei_ai.types import Model

from ..cli.args import is_valid_thinking_level
from .defaults import DEFAULT_THINKING_LEVEL


# Default model IDs for each known provider
DEFAULT_MODEL_PER_PROVIDER: dict[str, str] = {
    "amazon-bedrock": "us.anthropic.claude-opus-4-6-v1",
    "ant-ling": "Ring-2.6-1T",
    "anthropic": "claude-opus-4-8",
    "openai": "gpt-5.5",
    "azure-openai-responses": "gpt-5.4",
    "openai-codex": "gpt-5.5",
    "nvidia": "nvidia/nemotron-3-super-120b-a12b",
    "deepseek": "deepseek-v4-pro",
    "google": "gemini-3.1-pro-preview",
    "google-vertex": "gemini-3.1-pro-preview",
    "github-copilot": "gpt-5.4",
    "openrouter": "moonshotai/kimi-k2.6",
    "vercel-ai-gateway": "zai/glm-5.1",
    "xai": "grok-4.5",
    "groq": "openai/gpt-oss-120b",
    "cerebras": "zai-glm-4.7",
    "zai": "glm-5.1",
    "zai-coding-cn": "glm-5.1",
    "mistral": "devstral-medium-latest",
    "minimax": "MiniMax-M2.7",
    "minimax-cn": "MiniMax-M2.7",
    "moonshotai": "kimi-k2.6",
    "moonshotai-cn": "kimi-k2.6",
    "huggingface": "moonshotai/Kimi-K2.6",
    "fireworks": "accounts/fireworks/models/kimi-k2p6",
    "together": "moonshotai/Kimi-K2.6",
    "baseten": "zai-org/GLM-5.2",
    "opencode": "kimi-k2.6",
    "opencode-go": "kimi-k2.6",
    "kimi-coding": "kimi-for-coding",
    "cloudflare-workers-ai": "@cf/moonshotai/kimi-k2.6",
    "cloudflare-ai-gateway": "workers-ai/@cf/moonshotai/kimi-k2.6",
    "qwen-token-plan": "qwen3.7-max",
    "qwen-token-plan-cn": "qwen3.7-max",
    "qwen-token-plan-individual": "qwen3.8-max",
    "xiaomi": "mimo-v2.5-pro",
    "xiaomi-token-plan-cn": "mimo-v2.5-pro",
    "xiaomi-token-plan-ams": "mimo-v2.5-pro",
    "xiaomi-token-plan-sgp": "mimo-v2.5-pro",
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
    """A model ID looks like an alias when it has no date suffix (-YYYYMMDD)."""
    if id.endswith("-latest"):
        return True
    return re.search(r"-\d{8}$", id) is None


def find_exact_model_reference_match(model_reference: str, available_models: list[Model]) -> Model | None:
    """Find an exact model reference match: bare model id or canonical
    provider/modelId. Ambiguous bare-id matches across providers are rejected."""
    trimmed_reference = model_reference.strip()
    if not trimmed_reference:
        return None

    normalized_reference = trimmed_reference.lower()

    canonical_matches = [
        model for model in available_models if f"{model.provider}/{model.id}".lower() == normalized_reference
    ]
    if len(canonical_matches) == 1:
        return canonical_matches[0]
    if len(canonical_matches) > 1:
        return None

    slash_index = trimmed_reference.find("/")
    if slash_index != -1:
        provider = trimmed_reference[:slash_index].strip()
        model_id = trimmed_reference[slash_index + 1 :].strip()
        if provider and model_id:
            provider_matches = [
                model
                for model in available_models
                if model.provider.lower() == provider.lower() and model.id.lower() == model_id.lower()
            ]
            if len(provider_matches) == 1:
                return provider_matches[0]
            if len(provider_matches) > 1:
                return None

    id_matches = [model for model in available_models if model.id.lower() == normalized_reference]
    return id_matches[0] if len(id_matches) == 1 else None


def _try_match_model(model_pattern: str, available_models: list[Model]) -> Model | None:
    exact_match = find_exact_model_reference_match(model_pattern, available_models)
    if exact_match is not None:
        return exact_match

    # No exact match - fall back to partial matching
    lowered = model_pattern.lower()
    matches = [
        model
        for model in available_models
        if lowered in model.id.lower() or (model.name is not None and lowered in model.name.lower())
    ]

    if not matches:
        return None

    aliases = [model for model in matches if _is_alias(model.id)]
    dated_versions = [model for model in matches if not _is_alias(model.id)]

    if aliases:
        # Prefer alias - if multiple aliases, pick the one that sorts highest
        aliases.sort(key=lambda model: model.id, reverse=True)
        return aliases[0]
    dated_versions.sort(key=lambda model: model.id, reverse=True)
    return dated_versions[0]


@dataclass(slots=True)
class ParsedModelResult:
    model: Model | None
    thinking_level: str | None = None
    warning: str | None = None


def _build_fallback_model(provider: str, model_id: str, available_models: list[Model]) -> Model | None:
    provider_models = [model for model in available_models if model.provider == provider]
    if not provider_models:
        return None

    default_id = DEFAULT_MODEL_PER_PROVIDER.get(provider)
    base_model = (
        next((model for model in provider_models if model.id == default_id), provider_models[0])
        if default_id
        else provider_models[0]
    )

    return replace(base_model, id=model_id, name=model_id)


def parse_model_pattern(
    pattern: str,
    available_models: list[Model],
    *,
    allow_invalid_thinking_level_fallback: bool = True,
) -> ParsedModelResult:
    """Parse a pattern to extract model and thinking level, tolerating colons
    inside model IDs (e.g. OpenRouter's :exacto suffix)."""
    exact_match = _try_match_model(pattern, available_models)
    if exact_match is not None:
        return ParsedModelResult(model=exact_match)

    last_colon_index = pattern.rfind(":")
    if last_colon_index == -1:
        return ParsedModelResult(model=None)

    prefix = pattern[:last_colon_index]
    suffix = pattern[last_colon_index + 1 :]

    if is_valid_thinking_level(suffix):
        result = parse_model_pattern(
            prefix, available_models, allow_invalid_thinking_level_fallback=allow_invalid_thinking_level_fallback
        )
        if result.model is not None:
            # Only use this thinking level if no warning from inner recursion
            return ParsedModelResult(
                model=result.model,
                thinking_level=None if result.warning else suffix,
                warning=result.warning,
            )
        return result

    if not allow_invalid_thinking_level_fallback:
        # In strict mode (CLI --model parsing), treat it as part of the model id and fail.
        # This avoids accidentally resolving to a different model.
        return ParsedModelResult(model=None)

    result = parse_model_pattern(
        prefix, available_models, allow_invalid_thinking_level_fallback=allow_invalid_thinking_level_fallback
    )
    if result.model is not None:
        return ParsedModelResult(
            model=result.model,
            warning=f'Invalid thinking level "{suffix}" in pattern "{pattern}". Using default instead.',
        )
    return result


@dataclass(slots=True)
class ModelScopeDiagnostic:
    type: str
    code: str  # "no-match" | "invalid-thinking-level"
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
    available_models = list(models)
    scoped_models: list[ScopedModel] = []
    diagnostics: list[ModelScopeDiagnostic] = []

    for pattern in patterns:
        # Check if pattern contains glob characters
        if "*" in pattern or "?" in pattern or "[" in pattern:
            # Extract optional thinking level suffix (e.g., "provider/*:high")
            colon_index = pattern.rfind(":")
            glob_pattern = pattern
            thinking_level: str | None = None

            if colon_index != -1:
                suffix = pattern[colon_index + 1 :]
                if is_valid_thinking_level(suffix):
                    thinking_level = suffix
                    glob_pattern = pattern[:colon_index]

            exact_match = find_exact_model_reference_match(glob_pattern, available_models)
            if exact_match is not None:
                if not any(models_are_equal(scoped.model, exact_match) for scoped in scoped_models):
                    scoped_models.append(ScopedModel(model=exact_match, thinking_level=thinking_level))
                continue

            # Match against "provider/modelId" format OR just model ID
            # This allows "*sonnet*" to match without requiring "anthropic/*sonnet*"
            matching_models = [
                model
                for model in available_models
                if _minimatch(f"{model.provider}/{model.id}", glob_pattern) or _minimatch(model.id, glob_pattern)
            ]

            if not matching_models:
                diagnostics.append(
                    ModelScopeDiagnostic(
                        type="warning", code="no-match", message=f'No models match pattern "{pattern}"', pattern=pattern
                    )
                )
                continue

            for model in matching_models:
                if not any(models_are_equal(scoped.model, model) for scoped in scoped_models):
                    scoped_models.append(ScopedModel(model=model, thinking_level=thinking_level))
            continue

        parsed = parse_model_pattern(pattern, available_models)

        if parsed.warning:
            diagnostics.append(
                ModelScopeDiagnostic(
                    type="warning", code="invalid-thinking-level", message=parsed.warning, pattern=pattern
                )
            )

        if parsed.model is None:
            diagnostics.append(
                ModelScopeDiagnostic(
                    type="warning", code="no-match", message=f'No models match pattern "{pattern}"', pattern=pattern
                )
            )
            continue

        if not any(models_are_equal(scoped.model, parsed.model) for scoped in scoped_models):
            scoped_models.append(ScopedModel(model=parsed.model, thinking_level=parsed.thinking_level))

    return ResolveModelScopeResult(scoped_models=scoped_models, diagnostics=diagnostics)


async def resolve_model_scope_with_diagnostics(
    patterns: list[str],
    model_runtime,
    options: AuthOperationOptions | None = None,
) -> ResolveModelScopeResult:
    return resolve_model_scope_from_models(patterns, list(await model_runtime.get_available(None, options)))


def _warn(message: str) -> None:
    print(f"\x1b[33mWarning: {message}\x1b[0m", file=sys.stderr)


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
    model_runtime,
) -> ResolveCliModelResult:
    """Resolve a single model from CLI flags (--provider/--model, fuzzy rules)."""
    if not cli_model:
        return ResolveCliModelResult(model=None)

    # Important: use *all* models here, not just models with pre-configured auth.
    # This allows "--api-key" to be used for first-time setup.
    available_models = list(model_runtime.get_models())
    if not available_models:
        return ResolveCliModelResult(
            model=None,
            error="No models available. Check your installation or add models to models.json.",
        )

    # Build canonical provider lookup (case-insensitive)
    provider_map: dict[str, str] = {}
    for model in available_models:
        provider_map[model.provider.lower()] = model.provider

    provider = provider_map.get(cli_provider.lower()) if cli_provider else None
    if cli_provider and not provider:
        return ResolveCliModelResult(
            model=None,
            error=f'Unknown provider "{cli_provider}". Use --list-models to see available providers/models.',
        )

    # If no explicit --provider, try to interpret "provider/model" format first.
    pattern = cli_model
    inferred_provider = False

    if not provider:
        slash_index = cli_model.find("/")
        if slash_index != -1:
            maybe_provider = cli_model[:slash_index]
            canonical = provider_map.get(maybe_provider.lower())
            if canonical:
                provider = canonical
                pattern = cli_model[slash_index + 1 :]
                inferred_provider = True

    # If no provider was inferred from the slash, try exact matches without provider inference.
    # This handles models whose IDs naturally contain slashes (e.g. OpenRouter-style IDs).
    if not provider:
        lower = cli_model.lower()
        exact = next(
            (
                model
                for model in available_models
                if model.id.lower() == lower or f"{model.provider}/{model.id}".lower() == lower
            ),
            None,
        )
        if exact is not None:
            return ResolveCliModelResult(model=exact)

    if cli_provider and provider:
        # If both were provided, tolerate --model <provider>/<pattern> by stripping the provider prefix
        prefix = f"{provider}/"
        if cli_model.lower().startswith(prefix.lower()):
            pattern = cli_model[len(prefix) :]

    candidates = [model for model in available_models if model.provider == provider] if provider else available_models
    parsed = parse_model_pattern(pattern, candidates, allow_invalid_thinking_level_fallback=False)

    if parsed.model is not None:
        # If provider inference matched an unauthenticated provider/model pair, prefer
        # one exact raw model-id match that is authenticated.
        if inferred_provider:
            raw_exact_matches = [
                model
                for model in available_models
                if model.id.lower() == cli_model.lower() and not models_are_equal(model, parsed.model)
            ]
            if raw_exact_matches and not model_runtime.has_configured_auth(parsed.model.provider):
                authenticated_raw_matches = [
                    model for model in raw_exact_matches if model_runtime.has_configured_auth(model.provider)
                ]
                if len(authenticated_raw_matches) == 1:
                    return ResolveCliModelResult(model=authenticated_raw_matches[0])
        return ResolveCliModelResult(model=parsed.model, thinking_level=parsed.thinking_level, warning=parsed.warning)

    # If we inferred a provider from the slash but found no match within that provider,
    # fall back to matching the full input as a raw model id across all models.
    if inferred_provider:
        lower = cli_model.lower()
        exact = next(
            (
                model
                for model in available_models
                if model.id.lower() == lower or f"{model.provider}/{model.id}".lower() == lower
            ),
            None,
        )
        if exact is not None:
            return ResolveCliModelResult(model=exact)
        fallback = parse_model_pattern(cli_model, available_models, allow_invalid_thinking_level_fallback=False)
        if fallback.model is not None:
            return ResolveCliModelResult(
                model=fallback.model, thinking_level=fallback.thinking_level, warning=fallback.warning
            )

    if provider:
        # Parse thinking level suffix from the pattern before building the fallback model,
        # but only when --thinking is not explicitly provided.
        fallback_pattern = pattern
        fallback_thinking: str | None = None
        if not cli_thinking:
            last_colon = pattern.rfind(":")
            if last_colon != -1:
                suffix = pattern[last_colon + 1 :]
                if is_valid_thinking_level(suffix):
                    fallback_pattern = pattern[:last_colon]
                    fallback_thinking = suffix

        fallback_model = _build_fallback_model(provider, fallback_pattern, available_models)
        if fallback_model is not None:
            requested_thinking = cli_thinking if cli_thinking else fallback_thinking
            model = (
                replace(fallback_model, reasoning=True)
                if requested_thinking and requested_thinking != "off"
                else fallback_model
            )
            not_found = f'Model "{fallback_pattern}" not found for provider "{provider}". Using custom model id.'
            fallback_warning = f"{parsed.warning} {not_found}" if parsed.warning else not_found
            return ResolveCliModelResult(model=model, thinking_level=fallback_thinking, warning=fallback_warning)

    display = f"{provider}/{pattern}" if provider else cli_model
    return ResolveCliModelResult(
        model=None,
        warning=parsed.warning,
        error=f'Model "{display}" not found. Use --list-models to see available models.',
    )


@dataclass(slots=True)
class InitialModelResult:
    model: Model | None
    thinking_level: str
    fallback_message: str | None = None


async def find_initial_model(
    *,
    cli_provider: str | None = None,
    cli_model: str | None = None,
    scoped_models: list[ScopedModel],
    is_continuing: bool,
    default_provider: str | None = None,
    default_model_id: str | None = None,
    default_thinking_level: str | None = None,
    model_runtime,
) -> InitialModelResult:
    """Find the initial model: CLI args, scoped models, saved default, then
    the first available model with valid auth."""
    # 1. CLI args take priority
    if cli_provider and cli_model:
        resolved = resolve_cli_model(cli_provider=cli_provider, cli_model=cli_model, model_runtime=model_runtime)
        if resolved.error:
            print(f"\x1b[31m{resolved.error}\x1b[0m", file=sys.stderr)
            sys.exit(1)
        if resolved.model is not None:
            return InitialModelResult(model=resolved.model, thinking_level=DEFAULT_THINKING_LEVEL)

    # 2. Use first model from scoped models (skip if continuing/resuming)
    if scoped_models and not is_continuing:
        thinking = scoped_models[0].thinking_level or default_thinking_level or DEFAULT_THINKING_LEVEL
        return InitialModelResult(model=scoped_models[0].model, thinking_level=thinking)

    # 3. Try saved default from settings if auth is configured.
    if default_provider and default_model_id:
        found = model_runtime.get_model(default_provider, default_model_id)
        if found is not None and model_runtime.has_configured_auth(found.provider):
            thinking = default_thinking_level if default_thinking_level else DEFAULT_THINKING_LEVEL
            return InitialModelResult(model=found, thinking_level=thinking)

    # 4. Try first available model with valid API key
    available_models = list(model_runtime.get_available_snapshot())

    if available_models:
        for provider, default_id in DEFAULT_MODEL_PER_PROVIDER.items():
            match = next(
                (model for model in available_models if model.provider == provider and model.id == default_id), None
            )
            if match is not None:
                return InitialModelResult(model=match, thinking_level=DEFAULT_THINKING_LEVEL)

        return InitialModelResult(model=available_models[0], thinking_level=DEFAULT_THINKING_LEVEL)

    # 5. No model found
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
            print(f"Restored model: {saved_provider}/{saved_model_id}")
        return restored_model, None

    reason = "model no longer exists" if restored_model is None else "no auth configured"

    if should_print_messages:
        _warn(f"Could not restore model {saved_provider}/{saved_model_id} ({reason}).")

    if current_model is not None:
        if should_print_messages:
            print(f"Falling back to: {current_model.provider}/{current_model.id}")
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
            print(f"Falling back to: {fallback_model.provider}/{fallback_model.id}")

        return fallback_model, (
            f"Could not restore model {saved_provider}/{saved_model_id} ({reason}). "
            f"Using {fallback_model.provider}/{fallback_model.id}."
        )

    return None, None
