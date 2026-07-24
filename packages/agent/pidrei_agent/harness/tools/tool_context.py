"""Filesystem and shell context required by the built-in execution tools (port of pi `tools/tool-context.ts`)."""

from dataclasses import dataclass

from ..types import ExecutionEnv


@dataclass
class ExecutionToolContext:
    env: ExecutionEnv
