"""Mirror of pi coding-agent src/core/project-trust.ts."""

from collections.abc import Callable
from dataclasses import dataclass

import tonio.colored as tonio

from ..config import APP_NAME, CONFIG_DIR_NAME
from .extensions.runner import emit_project_trust_event
from .extensions.types import LoadExtensionsResult, ProjectTrustContext
from .trust_manager import (
    ProjectTrustOption,
    ProjectTrustStore,
    get_project_trust_options,
    has_trust_requiring_project_resources,
)


# AppMode = "interactive" | "print" | "json" | "rpc"


@dataclass(slots=True, kw_only=True)
class ResolveProjectTrustedOptions:
    cwd: str
    trust_store: ProjectTrustStore
    trust_override: bool | None = None
    default_project_trust: str | None = None
    extensions_result: LoadExtensionsResult | None = None
    project_trust_context: ProjectTrustContext | None = None
    on_extension_error: Callable[[str], None] | None = None


def format_project_trust_prompt(cwd: str) -> str:
    return (
        f"Trust project folder?\n{cwd}\n\nThis allows {APP_NAME} to load {CONFIG_DIR_NAME} settings and "
        "resources, install missing project packages, and execute project extensions."
    )


async def _select_project_trust_option(cwd: str, ctx: ProjectTrustContext) -> ProjectTrustOption | None:
    options = get_project_trust_options(cwd, include_session_only=True)
    selected = await ctx.ui.select(format_project_trust_prompt(cwd), [option.label for option in options])
    return next((option for option in options if option.label == selected), None)


async def _save_project_trust_prompt_result(trust_store: ProjectTrustStore, result: ProjectTrustOption) -> None:
    if result.updates:
        await trust_store.set_many(result.updates)


async def resolve_project_trusted(options: ResolveProjectTrustedOptions) -> bool:
    if options.trust_override is not None:
        return options.trust_override
    # Walks cwd's ancestors probing for project resources — one blocking unit.
    if not await tonio.spawn_blocking(has_trust_requiring_project_resources, options.cwd):
        return True

    if options.extensions_result is not None:
        result, errors = await emit_project_trust_event(
            options.extensions_result,
            {"type": "project_trust", "cwd": options.cwd},
            options.project_trust_context,
        )
        for error in errors:
            if options.on_extension_error is not None:
                options.on_extension_error(f'Extension "{error.extension_path}" project_trust error: {error.error}')
        if result is not None:
            trusted = result.get("trusted") == "yes"
            if result.get("remember") is True:
                await options.trust_store.set(options.cwd, trusted)
            return trusted

    decision = await options.trust_store.get(options.cwd)
    if decision is not None:
        return decision

    default_trust = options.default_project_trust if options.default_project_trust is not None else "ask"
    if default_trust == "always":
        return True
    if default_trust == "never":
        return False

    if options.project_trust_context is None or not options.project_trust_context.has_ui:
        return False

    selected = await _select_project_trust_option(options.cwd, options.project_trust_context)
    if selected is not None:
        await _save_project_trust_prompt_result(options.trust_store, selected)
        return selected.trusted
    return False
