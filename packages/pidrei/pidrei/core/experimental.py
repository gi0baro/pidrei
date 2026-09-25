"""Mirror of pi coding-agent src/core/experimental.ts."""

import os


def are_experimental_features_enabled() -> bool:
    return os.environ.get("PIDREI_EXPERIMENTAL") == "1"
