"""Mirror of pi coding-agent src/core/experimental.ts."""

import os

from pidrei_ai.types import JsonSchemaConstrainedSampling


def are_experimental_features_enabled() -> bool:
    return os.environ.get("PIDREI_EXPERIMENTAL") == "1"


def get_experimental_tool_sampling() -> JsonSchemaConstrainedSampling | None:
    return JsonSchemaConstrainedSampling(strict="prefer") if are_experimental_features_enabled() else None
