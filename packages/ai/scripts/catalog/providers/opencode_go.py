"""OpenCode Go (pi: catalog/providers/opencode-go.ts)."""

from catalog.define import ProviderDefinition
from catalog.providers.opencode import OPENCODE_COMPLETIONS_COMPAT, finalize_opencode_model, opencode_models


PROVIDER = ProviderDefinition(
    id="opencode-go",
    models=opencode_models,
    compat={"openai-completions": OPENCODE_COMPLETIONS_COMPAT},
    finalize=finalize_opencode_model,
)
