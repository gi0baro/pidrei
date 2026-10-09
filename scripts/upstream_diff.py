"""Upstream sync report: what changed in pi since the last ported commit.

`make upstream-diff` walks `$(cat .last_upstream_ref)..HEAD` in the pi checkout
(`PI_ROOT`) and prints a commit-by-commit porting checklist, oldest first —
upstream commits are the porting unit, and new pi tests are the spec by
construction.

Every changed file is classified:

- **src / test / doc / example** — maps to a pidrei file through PREFIX_MAP
  (kebab-case → snake_case, `.ts` → `.py`; pi's nested test dirs flatten into
  our flat `tests/` layout; example dirs snake-case but their non-code assets
  — subagent prompts etc. — keep their upstream names). Ports whose pidrei
  name diverges from the mechanical mapping live in RENAMES, one
  hand-verified entry per divergence.
- **dropped** — documented divergences (the radius provider, the llama.cpp
  extension, pi's evals and storage packages, the npm/wasm example
  extensions). Surfaced as a one-liner, no port needed.
- **noise** — pi-internal machinery: lockfiles, changelogs, CI, vitest
  configs. Top-level `package.json` changes in ported packages are
  additionally summarized at the end: new runtime deps need a manual look.
- **UNMAPPED** — anything else. Loud, and the exit code is 2: either a new
  upstream file class (extend the tables) or a new pi package (decide port vs
  drop, and document the decision here).

A portable file listed in DIVERGED additionally carries a diverged-region
warning: part of its pidrei mirror deliberately diverges from pi, and a hunk
landing in that region is translated per the named recipe in
spec/upstream-sync.md ("Diverged regions") instead of ported side-by-side.
Hunks outside the region port normally.

A mapped target that does not exist is `[NEW]` for files pi added; for files
pi *modified* it usually means pidrei renamed the module — verify and extend
RENAMES instead of porting to the mechanical name.

`--bump <sha>` records progress once everything up to `<sha>` is ported:
verifies the sha sits on the ported-ref → HEAD line, then writes both
`.last_upstream_ref` and `upstream.py`'s UPSTREAM_REF. When the sha crosses a
pi release tag it warns that UPSTREAM_VERSION and the package versions must
move together (release_check gates that).
"""

import argparse
import os
import re
import shutil
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_FILE = os.path.join(ROOT, ".last_upstream_ref")
UPSTREAM_PY = os.path.join(ROOT, "packages", "pidrei", "pidrei", "upstream.py")

#: pi path prefix -> (pidrei path prefix, kind). Kind drives the name
#: transform: src/doc keep pi's directory shape, test flattens.
PREFIX_MAP = (
    ("packages/ai/src/", "packages/ai/pidrei_ai/", "src"),
    ("packages/ai/scripts/", "packages/ai/scripts/", "src"),
    ("packages/ai/test/", "packages/ai/tests/", "test"),
    ("packages/agent/src/", "packages/agent/pidrei_agent/", "src"),
    ("packages/agent/test/", "packages/agent/tests/", "test"),
    ("packages/codemode/src/", "packages/codemode/pidrei_codemode/", "src"),
    ("packages/codemode/test/", "packages/codemode/tests/", "test"),
    ("packages/mcp/src/", "packages/mcp/pidrei_mcp/", "src"),
    ("packages/mcp/test/", "packages/mcp/tests/", "test"),
    ("packages/coding-agent/src/", "packages/pidrei/pidrei/", "src"),
    ("packages/coding-agent/test/", "packages/pidrei/tests/", "test"),
    ("packages/coding-agent/docs/", "packages/pidrei/pidrei/docs/", "doc"),
    # Example extensions are mirrored product surface, not noise: 1d08508ef
    # (agent_settled in the examples) was silently lost while this prefix sat
    # in NOISE_PREFIXES. Unmirrored examples are DROPPED entries below.
    ("packages/coding-agent/examples/", "packages/pidrei/pidrei/examples/", "example"),
    ("packages/tui/src/", "packages/tui/pidrei_tui/", "src"),
    ("packages/tui/test/", "packages/tui/tests/", "test"),
    ("packages/server/src/", "packages/server/pidrei_server/", "src"),
    ("packages/server/test/", "packages/server/tests/", "test"),
    ("packages/protocol/src/", "packages/protocol/pidrei_protocol/", "src"),
    ("packages/protocol/test/", "packages/protocol/tests/", "test"),
    ("packages/client/src/", "packages/client/pidrei_client/", "src"),
    ("packages/client/test/", "packages/client/tests/", "test"),
)

#: pi paths pidrei deliberately does not port, with the recorded reason.
DROPPED_PREFIXES = (
    ("packages/evals/", "pi-internal eval harness, not ported"),
    ("packages/storage/", "storage backend, not ported"),
    (
        "packages/coding-agent/src/extensions/llama/",
        "llama.cpp extension not ported (see pidrei/extensions/__init__.py)",
    ),
    (
        "packages/coding-agent/test/llama-extension.test.ts",
        "llama.cpp extension not ported (see pidrei/extensions/__init__.py)",
    ),
    (
        "packages/ai/test/xhigh.test.ts",
        "live-API test (skipIf !OPENAI_API_KEY), not ported",
    ),
    (
        "packages/ai/test/openai-responses-reasoning-replay-e2e.test.ts",
        "live-API test, not ported (offline mirror: test_azure_openai_responses_reasoning_replay.py)",
    ),
    (
        "packages/ai/test/anthropic-thinking-binding-e2e.test.ts",
        "live-API test (skipIf !ANTHROPIC_API_KEY), not ported (offline mirror: test_anthropic_mid_conversation_effort.py)",
    ),
    (
        "packages/ai/src/api/pi-messages.ts",
        "pi-messages adapter not ported (radius wire protocol; types.py keeps the api literal only)",
    ),
    (
        "packages/ai/test/pi-messages.test.ts",
        "pi-messages adapter not ported (radius wire protocol; types.py keeps the api literal only)",
    ),
    (
        "packages/agent/src/proxy.ts",
        "server-proxied stream fn: public pi-agent API with no pidrei consumer, not ported",
    ),
    (
        "packages/coding-agent/test/sdk-codex-cache-probe-tool-loop.ts",
        "manual SDK probe script, not ported",
    ),
    # 0.84.x additions (0.84.1 delta port).
    (
        "packages/session-backends/",
        "storage backends not ported (packages/storage/ renamed upstream in 79cc1ef0)",
    ),
    (
        "packages/telemetry/",
        "telemetry not ported (no phone-home; 0.84.1 delta port)",
    ),
    (
        "packages/agent/src/harness/telemetry.ts",
        "telemetry not ported (no phone-home; 0.84.1 delta port)",
    ),
    (
        "packages/agent/test/harness/telemetry.test.ts",
        "telemetry not ported (no phone-home; 0.84.1 delta port)",
    ),
    (
        "packages/agent/scripts/generate-telemetry-docs.ts",
        "telemetry not ported (no phone-home; 0.84.1 delta port)",
    ),
    (
        "packages/ai/test/telemetry-options.test.ts",
        "telemetry not ported (no phone-home; 0.84.1 delta port)",
    ),
    (
        "packages/ai/docs/telemetry-schema.md",
        "telemetry not ported (no phone-home; 0.84.1 delta port)",
    ),
    (
        "packages/agent/telemetry-schema.md",
        "telemetry not ported (no phone-home; 0.84.1 delta port)",
    ),
    (
        "packages/tui/native/",
        "native modifier addon not ported (terminal.py stubs _is_native_modifier_pressed)",
    ),
    (
        "packages/tui/src/native-modifiers.ts",
        "native modifier addon not ported (terminal.py stubs _is_native_modifier_pressed)",
    ),
    (
        "packages/agent/test/harness/sqlite-migrations.test.ts",
        "SQLite session backend not ported (moved to session-backends upstream)",
    ),
    (
        "packages/agent/test/harness/sqlite-node.test.ts",
        "SQLite session backend not ported (moved to session-backends upstream)",
    ),
    (
        "packages/agent/test/harness/sqlite-branch-cache.test.ts",
        "SQLite session backend not ported (moved to session-backends upstream)",
    ),
    (
        "packages/coding-agent/src/modes/interactive/components/mermaid.ts",
        "Mermaid rendering not ported (grok-mermaid JS dep; 0.84.1 delta port)",
    ),
    (
        "packages/coding-agent/test/mermaid.test.ts",
        "Mermaid rendering not ported (grok-mermaid JS dep; 0.84.1 delta port)",
    ),
    (
        "packages/ai/src/cli.ts",
        "manual OAuth helper for live-API tests, not ported",
    ),
    (
        "packages/ai/test/oauth.ts",
        "live-API test auth helper, not ported",
    ),
    (
        "packages/coding-agent/src/rpc-entry.ts",
        "separate rpc bin entry not ported (pidrei exposes --mode rpc on its single console script)",
    ),
    ("packages/tui/test/chat-simple.ts", "manual demo script, not ported"),
    ("packages/tui/test/image-test.ts", "manual demo script, not ported"),
    ("packages/tui/test/key-tester.ts", "manual demo script, not ported"),
    ("packages/tui/test/viewport-overwrite-repro.ts", "manual repro script, not ported"),
    ("packages/tui/test/render-churn-bench.ts", "manual V8-heap-profiler benchmark, not ported"),
    ("packages/agent/test/scratch/", "manual scratch scripts, not ported"),
    ("packages/coding-agent/test/streaming-render-debug.ts", "manual debug script, not ported"),
    # 0.84.2 drops (user-approved 2026-08-16).
    (
        "packages/ai/src/api/cloudflare-gateway-binding.ts",
        "Cloudflare Workers env.AI binding shim — JS-runtime object, no pidrei consumer",
    ),
    (
        "packages/ai/test/cloudflare-gateway-binding.test.ts",
        "Cloudflare Workers env.AI binding shim — JS-runtime object, no pidrei consumer",
    ),
    (
        "packages/agent/test/proxy.test.ts",
        "server-proxied stream fn: public pi-agent API with no pidrei consumer, not ported",
    ),
    # 0.84.3 drops (0.84.3 delta port).
    (
        "packages/coding-agent/src/migrations.ts",
        (
            "legacy pi-version config migrations not ported (pidrei starts from a "
            "fresh ~/.pidrei; config-slice decision)"
        ),
    ),
)
#: 0.84.3 additions (0.84.3 delta port).
DROPPED_PREFIXES += (
    (
        "packages/tui/src/native-module-path.ts",
        "native addon resolution not ported (POSIX-only; terminal.py stubs the native modifier check)",
    ),
    (
        "packages/tui/test/native-module-path.test.ts",
        "native addon resolution not ported (POSIX-only; terminal.py stubs the native modifier check)",
    ),
    (
        "packages/coding-agent/src/utils/highlight-js.d.ts",
        "TypeScript ambient declaration for highlight.js; pidrei highlights with pygments",
    ),
    (
        "packages/coding-agent/src/core/tools/powershell.ts",
        (
            "powershell tool not ported (Windows-only by construction; 0.84.3 delta port) — "
            "the shell-tool refactor it shares with bash did land, see core/tools/bash.py"
        ),
    ),
    (
        "packages/coding-agent/test/powershell-tool.test.ts",
        "powershell tool not ported (Windows-only by construction; 0.84.3 delta port)",
    ),
    (
        "packages/coding-agent/src/modes/interactive/session-share.ts",
        (
            "Radius share flow not ported (0.84.3 delta port); the surviving gist half "
            "stays inline in interactive_mode.py and the export half is core/session_export.py"
        ),
    ),
)
#: Example extensions with no pidrei mirror (all other examples are mirrored
#: 1:1 and map through PREFIX_MAP — see the `example` kind).
DROPPED_PREFIXES += (
    (
        "packages/coding-agent/examples/extensions/doom-overlay/",
        "C/wasm Doom overlay demo (emscripten build), not ported",
    ),
    (
        "packages/coding-agent/examples/extensions/gondolin/",
        "Gondolin micro-VM tool routing (npm runtime dep), not ported",
    ),
    (
        "packages/coding-agent/examples/extensions/sandbox/",
        "@anthropic-ai/sandbox-runtime bash sandbox (npm runtime dep), not ported",
    ),
    (
        "packages/coding-agent/examples/extensions/with-deps/",
        "npm-dependency packaging demo; pidrei extensions have no npm",
    ),
    (
        "packages/coding-agent/examples/rpc-extension-ui.ts",
        "Node terminal UI driving RPC mode, not ported",
    ),
    (
        "packages/coding-agent/examples/rpc-client.ts",
        "RpcClient example; the RPC surface stays unpublished (0.87.1 delta port)",
    ),
    (
        "packages/coding-agent/docs/rpc-commands.md",
        "RPC docs stay out of the curated set (0.87.1 delta port; transport remnants are unpublished)",
    ),
    (
        "packages/coding-agent/docs/rpc-extension-ui.md",
        "RPC docs stay out of the curated set (0.87.1 delta port; transport remnants are unpublished)",
    ),
    (
        "packages/coding-agent/examples/sdk/",
        "pi TS-SDK example scripts, not ported",
    ),
)

#: pi's experimental stack — the durable harness runtime, Chord, the
#: server/worker/client split and the Chord-era protocol/server/client — is
#: not ported (2026-09-04 ruling; DROPPED prefixes per its §9). The transport
#: survivors (protocol cbor + framing, server unix listener + byte contracts,
#: client unix transport) keep their mechanical mapping. The stable harness
#: helper layer went with pi 1.0.0 (`7fd478a2`); the modules the product used
#: now live in pidrei (spec/upstream-sync.md).
_EXPERIMENTAL_REASON = "experimental stack not ported (spec/upstream-sync.md, what is not ported)"
DROPPED_PREFIXES += tuple(
    (path, _EXPERIMENTAL_REASON)
    for path in (
        "packages/chord/",
        # Pico5 durable package (0.86.0 `08016016` moved pico out of agent)
        "packages/durable/",
        # pi-env: remote execution environments for Pi Durable (an SSH-deployed
        # Rust daemon and its ExecutionEnv client); no product consumer (1.0.4)
        "packages/env/",
        "packages/agent/benchmark/",
        # durable harness runtime (agent)
        "packages/agent/src/harness/agent-harness.ts",
        "packages/agent/src/harness/compaction/",
        "packages/agent/src/harness/config.ts",
        "packages/agent/src/harness/context.ts",
        "packages/agent/src/harness/events.ts",
        "packages/agent/src/harness/hooks.ts",
        "packages/agent/src/harness/result.ts",
        "packages/agent/src/harness/execution/",
        "packages/agent/src/harness/runtime/",
        "packages/agent/src/harness/session/",
        # usage arithmetic whose only consumers are the session storage and
        # harness compaction above (0.85.0 `fd9a45aa`/`29c41fc6`)
        "packages/agent/src/harness/utils/usage.ts",
        "packages/agent/src/search/",
        # transient in-range shapes (added and removed inside 0.85.0)
        "packages/agent/src/harness/runtime2/",
        "packages/agent/src/harness/restore.ts",
        "packages/agent/src/harness/agent-harness-runtime.ts",
        "packages/agent/src/plugins/",
        "packages/agent/test/plugins/",
        "packages/agent/test/harness/scratch.ts",
        "packages/agent/test/harness/runtime2-restore.test.ts",
        "packages/agent/test/harness/agent-harness-r",
        "packages/agent/test/harness/agent-harness-runtime.test.ts",
        "packages/agent/test/harness/restore.test.ts",
        "packages/agent/test/harness/lane.test.ts",
        "packages/agent/test/harness/harness.test.ts",
        "packages/agent/test/harness/progress.test.ts",
        "packages/agent/test/harness/session-tree.test.ts",
        "packages/agent/test/harness/session-create-lane.test.ts",
        "packages/agent/test/harness/utils.test.ts",
        "packages/agent/test/harness/id-generator.test.ts",
        "packages/agent/test/harness/lane-mutations.test.ts",
        "packages/agent/test/harness/scratch/",
        "packages/agent/test/harness/runtime2/",
        "packages/agent/test/harness/runtime/",
        "packages/agent/test/harness/branch-summarization.test.ts",
        "packages/agent/test/harness/branch.test.ts",
        "packages/agent/test/harness/compaction.test.ts",
        "packages/agent/test/harness/context.test.ts",
        "packages/agent/test/harness/execution-",
        "packages/agent/test/harness/gating-storage.test.ts",
        "packages/agent/test/harness/instrumented-storage.test.ts",
        "packages/agent/test/harness/jsonl-",
        "packages/agent/test/harness/memory-",
        "packages/agent/test/harness/mutation-line.test.ts",
        "packages/agent/test/harness/session-context.test.ts",
        "packages/agent/test/harness/session-create-branch.test.ts",
        "packages/agent/test/harness/session-codec.test.ts",
        "packages/agent/test/harness/storage-backed-session.test.ts",
        "packages/agent/test/harness/types.test.ts",
        "packages/agent/test/harness/values.test.ts",
        # pico / pico3 prototypes (0.86.0, before the move to packages/durable)
        "packages/agent/src/harness/pico",
        "packages/agent/test/harness/pico",
        "packages/agent/src/pico/",
        "packages/agent/test/pico/",
        # experimental coding-agent (server/worker/client split, facets, mini)
        "packages/coding-agent/src/experimental/",
        "packages/coding-agent/src/cli/experimental/",
        "packages/coding-agent/src/client/",
        "packages/coding-agent/src/server/",
        "packages/coding-agent/src/bun/",
        "packages/coding-agent/test/client/",
        "packages/coding-agent/test/server/",
        "packages/coding-agent/test/experimental-agent-controller.test.ts",
        "packages/coding-agent/test/experimental-cli-",
        "packages/coding-agent/test/experimental-client-tui.test.ts",
        "packages/coding-agent/test/experimental-harness-wire-adapter.test.ts",
        "packages/coding-agent/test/experimental-internal-process.test.ts",
        "packages/coding-agent/test/experimental-lane-replica.test.ts",
        "packages/coding-agent/test/experimental-plugin-reload.test.ts",
        "packages/coding-agent/test/experimental-presentation-facets.test.ts",
        "packages/coding-agent/test/experimental-remote-runtime.test.ts",
        "packages/coding-agent/test/experimental-server-",
        "packages/coding-agent/test/experimental-service-",
        "packages/coding-agent/test/experimental-session-",
        "packages/coding-agent/test/experimental-slash-commands.test.ts",
        "packages/coding-agent/test/experimental-transcript-provider.test.ts",
        "packages/coding-agent/test/experimental-chat-service.test.ts",
        "packages/coding-agent/test/experimental-facet",
        "packages/coding-agent/test/plugin-app-",
        "packages/coding-agent/test/fixtures/faux-session-worker.ts",
        "packages/coding-agent/test/fixtures/keyed-service.ts",
        "packages/coding-agent/test/fixtures/plugin-app/",
        "packages/coding-agent/test/fixtures/session-worker-fixture.ts",
        "packages/coding-agent/examples/plugins/",
        "packages/coding-agent/examples/facets/",
        # Chord-era protocol/server/client (transport survivors excluded)
        "packages/protocol/src/protocol.ts",
        "packages/protocol/src/codec.ts",
        "packages/protocol/src/harness.ts",
        "packages/protocol/src/json-value.ts",
        "packages/protocol/src/rpc.ts",
        "packages/protocol/src/service-state.ts",
        "packages/protocol/src/schemas.ts",
        "packages/protocol/src/index.ts",
        "packages/protocol/test/protocol.test.ts",
        "packages/server/src/server.ts",
        "packages/server/src/session-router.ts",
        "packages/server/src/sessions.ts",
        "packages/server/src/snapshots.ts",
        "packages/server/src/protocol.ts",
        "packages/server/src/hosted-harness-manager.ts",
        "packages/server/src/remote-session-manager.ts",
        "packages/server/src/service-id.ts",
        "packages/server/test/remote-session.test.ts",
        "packages/server/src/types.ts",
        "packages/server/src/errors.ts",
        "packages/server/src/index.ts",
        "packages/server/src/testing/",
        "packages/server/src/transports/unix/preset.ts",
        "packages/server/test/conformance.test.ts",
        "packages/server/test/listener.test.ts",
        "packages/server/test/protocol.test.ts",
        "packages/server/test/server.test.ts",
        "packages/server/test/sessions.test.ts",
        "packages/client/src/client.ts",
        "packages/client/src/control.ts",
        "packages/client/test/control.test.ts",
        "packages/client/src/connection.ts",
        "packages/client/src/errors.ts",
        "packages/client/src/index.ts",
        "packages/client/src/session-handle.ts",
        "packages/client/src/state.ts",
        "packages/client/src/types.ts",
        "packages/client/test/client.test.ts",
        "packages/client/test/connection.test.ts",
        "packages/client/test/disposal.test.ts",
        "packages/client/test/requests.test.ts",
        "packages/client/test/sessions.test.ts",
        "packages/client/test/state.test.ts",
        "packages/client/test/support.ts",
    )
)
DROPPED_PREFIXES += (
    # 0.85.0 drops (0.85.0 delta port).
    (
        "packages/coding-agent/test/session-share.test.ts",
        "Radius share flow not ported (0.84.3 delta port)",
    ),
    (
        "packages/tui/test/alt-screen-large-transcript-bench.ts",
        "manual render benchmark, not ported",
    ),
    (
        "packages/ai/src/api/cloudflare-ai-binding.ts",
        "Cloudflare Workers env.AI binding fetch — JS-runtime object, no pidrei consumer",
    ),
    (
        "packages/ai/test/cloudflare-ai-binding.test.ts",
        "Cloudflare Workers env.AI binding fetch — JS-runtime object, no pidrei consumer",
    ),
)
DROPPED_PREFIXES += (
    # 0.87.1 drops (0.87.1 delta port).
    (
        "packages/ai/src/bun-oauth.ts",
        "Bun bundle OAuth loader, JS-runtime only",
    ),
    (
        "packages/ai/src/compat.ts",
        (
            "deprecated global pi-ai API shim, never ported (initial port); its dispatch deltas land "
            "through models.ts → registry.py Models.stream"
        ),
    ),
    (
        "packages/ai/test/codex-websocket-cached-probe.ts",
        "manual live Codex probe script, not ported",
    ),
    (
        "packages/coding-agent/test/documentation.test.ts",
        "docs-navigation eval, not ported",
    ),
    (
        "packages/tui/src/native-platform.ts",
        "native platform helper (clipboard, VT input) not ported: pidrei's clipboard runs on the platform tools",
    ),
    (
        "packages/tui/test/native-platform.test.ts",
        "native platform helper (clipboard, VT input) not ported: pidrei's clipboard runs on the platform tools",
    ),
    (
        "packages/tui/test/native-clipboard-linux.test.ts",
        "native platform helper (clipboard, VT input) not ported: pidrei's clipboard runs on the platform tools",
    ),
    (
        "packages/tui/test/fixtures/clipboard-",
        "native platform helper (clipboard, VT input) not ported: pidrei's clipboard runs on the platform tools",
    ),
    (
        "packages/agent/test/harness/text-line-reader.test.ts",
        (
            "harness/env TextLineReader not ported: its only consumer is the experimental JSONL fork "
            "(0.87.1 delta port; the env/nodejs.ts + types.ts hunks are per-hunk skips)"
        ),
    ),
    (
        "packages/coding-agent/test/suite/regressions/8261-subagent-project-trust.test.ts",
        (
            "not mirrored: it relies on a child process failing at spawn, and pidrei's subagent "
            "example runs in process (recipe subagent-inprocess); example extensions carry no pidrei tests"
        ),
    ),
)
#: pi's tests of its example extensions: examples carry no pidrei tests
#: (2026-09-26 ruling); the examples themselves are still ported.
_EXAMPLE_TESTS_REASON = "tests an example extension; examples carry no pidrei tests (2026-09-26 ruling)"
DROPPED_PREFIXES += tuple(
    (path, _EXAMPLE_TESTS_REASON)
    for path in (
        "packages/coding-agent/test/git-merge-and-resolve-extension.test.ts",
        "packages/coding-agent/test/input-transform-streaming-example.test.ts",
        "packages/coding-agent/test/jev-router-example.test.ts",
        "packages/coding-agent/test/plan-mode-extension.test.ts",
        "packages/coding-agent/test/plan-mode-utils.test.ts",
        "packages/coding-agent/test/tool-renderer-examples.test.ts",
        "packages/coding-agent/test/trigger-compact-extension.test.ts",
    )
)
#: `/bug` bug reporting (Radius upload, zip export, crash log) is dropped
#: completely (0.87.1 delta port).
_BUG_REPORT_REASON = "/bug reporting not ported (0.87.1 delta port)"
DROPPED_PREFIXES += tuple(
    (path, _BUG_REPORT_REASON)
    for path in (
        "packages/coding-agent/src/core/bug-report.ts",
        "packages/coding-agent/src/core/bug-report-upload.ts",
        "packages/coding-agent/src/core/crash-log.ts",
        "packages/coding-agent/src/core/radius.ts",
        "packages/coding-agent/src/modes/interactive/bug-report.ts",
        # zip archive writer; bug-report.ts is its only consumer
        "packages/coding-agent/src/utils/zip.ts",
        "packages/coding-agent/test/bug-report.test.ts",
        "packages/coding-agent/test/interactive-mode-bug-report-hint.test.ts",
        "packages/coding-agent/test/crash-log.test.ts",
    )
)
#: MCP (1.0.0.0): what stays out of the port.
DROPPED_PREFIXES += (
    (
        "packages/coding-agent/test/mcp-conformance/",
        "MCP conformance suite not ported: pidrei's client mirrors pi's, which passes it (1.0.0.0)",
    ),
    (
        "packages/agent/examples/mcp-codemode/",
        "pi's MCP codemode example builds its own JavaScript codemode tool on sandbox internals (1.0.0.0)",
    ),
    # the lazy imports of the MCP runtime and CLI: pidrei imports them with
    # the extension (extensions/mcp/runtime.py docstring)
    ("packages/coding-agent/src/extensions/mcp/runtime.lazy.ts", "lazy-import shim; pidrei imports eagerly (1.0.0.0)"),
    ("packages/coding-agent/src/extensions/mcp/cli.lazy.ts", "lazy-import shim; pidrei imports eagerly (1.0.0.0)"),
)
#: Codemode's JavaScript sandbox machinery (1.0.0.0, recipe `codemode-python`):
#: pidrei's scripts run on Monty, driven by pidrei_codemode/runtime/host.py.
_CODEMODE_JS_REASON = "QuickJS/wasm codemode machinery; pidrei runs scripts on Monty (recipe codemode-python)"
DROPPED_PREFIXES += tuple(
    (path, _CODEMODE_JS_REASON)
    for path in (
        "packages/codemode/src/wasm.ts",
        "packages/codemode/src/runtime/worker.ts",
        "packages/codemode/src/runtime/protocol.ts",
        # a stand-in worker that posts raw bridge payloads to the host (1.0.4 `b223082b`)
        "packages/codemode/test/fixtures/raw-worker.ts",
        # the worker entry of pi's Bun/Node release builds (config.ts hunks are per-hunk skips)
        "packages/coding-agent/src/extensions/codemode/worker.ts",
        "packages/coding-agent/test/codemode-worker-config.test.ts",
        # the executor's lazy import: pidrei imports it with the extension
        # (extensions/codemode/__init__.py docstring)
        "packages/coding-agent/src/extensions/codemode/execute.lazy.ts",
    )
)
#: 0.99.1 drops (0.99.1 delta port).
_LLAMA_CLASSIFY_REASON = (
    "llama.cpp classifier API not ported: its only consumer is the unported llama extension "
    "(0.99.1 delta port; types.py keeps the api literal and ClassifierOptions.temperature)"
)
DROPPED_PREFIXES += (
    # llama-cpp-classify.ts and its .lazy.ts shim
    ("packages/ai/src/api/llama-cpp-classify", _LLAMA_CLASSIFY_REASON),
    ("packages/ai/test/llama-cpp-classify.test.ts", _LLAMA_CLASSIFY_REASON),
    (
        "packages/coding-agent/test/model-catalog-protocol.test.ts",
        (
            "catalog publish protocol not ported (0.99.1 delta port); the client half "
            "(?types=, merge by type+id, unknown types ignored) is covered by test_remote_catalog_provider.py"
        ),
    ),
    (
        "packages/coding-agent/test/rpc-example.ts",
        "interactive RpcClient example script; the RPC surface stays unpublished (0.87.1 delta port)",
    ),
    (
        "packages/ai/src/image-models.ts",
        (
            "deprecated static image-catalog reads, kept upstream for npm consumers: image_models.py deleted "
            "(0.99.1 delta port); get_builtin_image_model(s) in providers/all.py replace them"
        ),
    ),
)
#: 1.0.0 drops and deferrals (1.0.0 delta port).
DROPPED_PREFIXES += (
    (
        "packages/ai/test/models-entry.test.ts",
        "import footprint of pi's `pi-ai/models` npm entry point; no Python counterpart (1.0.0 delta port)",
    ),
    ("packages/coding-agent/test/experimental-durable-support.ts", _EXPERIMENTAL_REASON),
    # pi-logo-animation.ts and its .lazy.ts shim; TuiAltScreen.getScreenLines() is ported
    (
        "packages/coding-agent/src/modes/interactive/components/pi-logo-animation",
        "header logo easter egg deferred (1.0.0 delta port)",
    ),
    # 1.0.2 renamed it (armin-3d, then easter-egg-3d) and added the 3D Armin of /arminsayshi
    (
        "packages/coding-agent/src/modes/interactive/components/armin-3d",
        "3D easter eggs deferred (1.0.2 delta port)",
    ),
    (
        "packages/coding-agent/src/modes/interactive/components/easter-egg-3d",
        "3D easter eggs deferred (1.0.2 delta port)",
    ),
    (
        "packages/ai/scripts/hydrate-model-catalog.ts",
        "offline catalog hydration for pi's Nix build; PiDrei's catalog is committed (1.0.2 delta port)",
    ),
)

#: Live-API ai tests (`skipIf(!API_KEY)` upstream): they exercise real
#: providers, so pidrei drops them; offline mirrors exist where noted in
#: TEST_HOMES / test docstrings.
LIVE_API_AI_TESTS = (
    "abort",
    "context-overflow",
    "cross-provider-handoff",
    "empty",
    "image-tool-result",
    "images",
    "openai-completions-thinking-as-text",
    "openai-responses-tool-result-images",
    "responseid",
    "stream",
    "tokens",
    "tool-call-without-result",
    "total-tokens",
    "unicode-surrogate",
)
DROPPED_PREFIXES += tuple(
    (f"packages/ai/test/{name}.test.ts", "live-API test, not ported") for name in LIVE_API_AI_TESTS
)
DROPPED_PREFIXES += (
    # 1.1.0 drops (1.1.0 delta port).
    (
        "packages/coding-agent/src/utils/image-resize-worker.ts",
        "Node worker thread for image resizing; pidrei resizes with Pillow on the blocking pool (image_process.py)",
    ),
)
#: The radius provider (pi's own gateway) is the documented provider drop;
#: its files carry "radius" in the basename wherever they sit.
DROPPED_BASENAME_RE = re.compile(r"radius")
DROPPED_BASENAME_REASON = "radius provider dropped (pi-specific gateway; initial port)"

#: pi path -> pidrei path where the port's name diverges from the mechanical
#: mapping. Hand-verified; extend when the report flags a modified file whose
#: mechanical target is missing.
RENAMES = {
    "packages/coding-agent/src/core/remote-catalog-provider.ts": "packages/pidrei/pidrei/core/remote_catalog.py",
    "packages/coding-agent/test/package-command-paths.test.ts": "packages/pidrei/tests/test_package_commands.py",
    # 0.84.x additions (0.84.1 delta port).
    "packages/coding-agent/src/package-manager-cli.ts": "packages/pidrei/pidrei/cli/package_commands.py",
    # 0.85.1 (0.85.1 delta port): pi split its process-title/env/undici prologue into
    # cli/setup.ts; pidrei's equivalent has always been the entry module's run().
    "packages/coding-agent/src/cli/setup.ts": "packages/pidrei/pidrei/__main__.py",
    # 1.0.0: the calculate tool's only consumer is the e2e suite, which holds it.
    "packages/agent/test/utils/calculate.ts": "packages/agent/tests/test_e2e.py",
    "packages/ai/src/models.ts": "packages/ai/pidrei_ai/registry.py",
    # pi's name refers to the Node `http` module it configures; nothing in
    # pidrei is Node (docstring of http_proxy.py). The HTTP seam's modules live
    # in pidrei-http (1.0.0.0), shared by pidrei-ai and pidrei-mcp; the test
    # mirrors stay with pi-ai's. callback-server.ts maps mechanically to its
    # provider half; the lower layer is pidrei_http/callback_server.py.
    "packages/ai/src/utils/node-http-proxy.ts": "packages/http/pidrei_http/http_proxy.py",
    "packages/ai/test/node-http-proxy.test.ts": "packages/ai/tests/test_http_proxy.py",
    "packages/ai/src/auth/oauth/pkce.ts": "packages/http/pidrei_http/pkce.py",
    # EXIF orientation is Pillow's `ImageOps.exif_transpose` inside the image
    # pipeline; pi's hand-rolled APP1 scanner has no separate mirror.
    "packages/coding-agent/src/utils/exif-orientation.ts": "packages/pidrei/pidrei/utils/image_process.py",
    # PNG conversion for Kitty lives in the same module (Pillow replaces Photon).
    "packages/coding-agent/src/utils/image-convert.ts": "packages/pidrei/pidrei/utils/image_process.py",
    # 0.84.4: consolidated into generate_models.py like its models-dev sibling.
    "packages/ai/scripts/openrouter-reasoning-options.ts": "packages/ai/scripts/generate_models.py",
    "packages/ai/src/models.generated.ts": "packages/ai/pidrei_ai/models_generated.py",
    "packages/ai/src/image-models.generated.ts": "packages/ai/pidrei_ai/image_models_generated.py",
    "packages/coding-agent/src/cli.ts": "packages/pidrei/pidrei/__main__.py",
    "packages/coding-agent/src/core/http-dispatcher.ts": "packages/pidrei/pidrei/core/http_config.py",
    # pi's manifest is package.json's `pi` key; pidrei's is pyproject.toml's
    # `[tool.pidrei]`, so the module follows the pi→pidrei rename (U10).
    "packages/coding-agent/src/core/pi-manifest.ts": "packages/pidrei/pidrei/core/pidrei_manifest.py",
    "packages/coding-agent/src/modes/interactive/theme/theme-schema.json": "packages/pidrei/pidrei/modes/interactive/theme/theme-schema.json",
    "packages/tui/src/TuiAltScreen.ts": "packages/tui/pidrei_tui/tui_alt_screen.py",
    # Server transport (U4): unix-lifecycle.test.ts was consolidated into
    # unix.test.ts (546e00235); the stale-socket child fixture is inlined as a
    # python -c script in test_unix.py (both packages' copies).
    "packages/server/test/unix-lifecycle.test.ts": "packages/server/tests/test_unix.py",
    "packages/server/test/fixtures/stale-socket-server.mjs": "packages/server/tests/test_unix.py",
    "packages/client/test/fixtures/stale-socket-server.mjs": "packages/client/tests/test_unix.py",
    # 0.85.0: upstream renamed unix.test.ts to unix-transport.test.ts when the
    # client grew server discovery; pidrei's transport-only test keeps its name.
    "packages/client/test/unix-transport.test.ts": "packages/client/tests/test_unix.py",
    # 0.84.2 additions. test-theme-colors.ts is a manual CLI tool, not a
    # collected test; its mirror keeps the non-test_ name for the same reason.
    "packages/coding-agent/test/test-theme-colors.ts": "packages/pidrei/tests/theme_colors_tool.py",
    # 0.84.3 additions. pi split its user-agent helper out under a pi-prefixed
    # name; pidrei's has always been utils/user_agent.py.
    "packages/ai/src/utils/pi-user-agent.ts": "packages/ai/pidrei_ai/utils/user_agent.py",
    # 0.99.1 additions. model-catalog.ts's flatten*ModelCatalog helpers are the
    # catalog loader pidrei always had; the per-provider *.models.ts stubs map
    # through GENERATED_CATALOG_STUB_RE below.
    "packages/ai/src/model-catalog.ts": "packages/ai/pidrei_ai/models_generated.py",
    # Consolidated into generate_models.py like openrouter-reasoning-options.ts.
    "packages/ai/scripts/openrouter-catalog.ts": "packages/ai/scripts/generate_models.py",
    # 1.0.0.0 codemode: pi's prelude is JavaScript source run in the VM;
    # pidrei's is host-side Python plus a small in-session prelude, one module.
    "packages/codemode/src/runtime/prelude-source.ts": "packages/codemode/pidrei_codemode/runtime/prelude.py",
    # combineAbortSignals lives with the cancel token every package shares
    # (1.0.0.0, pidrei-utils).
    "packages/ai/src/utils/abort-signals.ts": "packages/utils/pidrei_utils/cancel.py",
    # 1.0.0.0 MCP: the testing facade is one module; the node fixtures are
    # Python scripts under tests/fixtures; the test helpers module is named so
    # the other packages' tests can import it; the SDK license travels as is.
    "packages/mcp/src/testing/index.ts": "packages/mcp/pidrei_mcp/testing.py",
    "packages/mcp/test/fixtures/stdio-server.mjs": "packages/mcp/tests/fixtures/stdio_server.py",
    "packages/mcp/test/fixtures/stubborn-server.mjs": "packages/mcp/tests/fixtures/stubborn_server.py",
    "packages/mcp/test/helpers.ts": "packages/mcp/tests/mcp_helpers.py",
    "packages/mcp/LICENSES/modelcontextprotocol-typescript-sdk.txt": (
        "packages/mcp/LICENSES/modelcontextprotocol-typescript-sdk.txt"
    ),
    # 1.1.0: resizing is Pillow on the blocking pool, in the one image module
    # (the worker-thread half is dropped below).
    "packages/coding-agent/src/utils/image-resize-core.ts": "packages/pidrei/pidrei/utils/image_process.py",
    "packages/coding-agent/src/utils/image-resize.ts": "packages/pidrei/pidrei/utils/image_process.py",
    # 1.1.0: consolidated into generate_models.py like the OpenRouter catalog helpers.
    "packages/ai/scripts/ai-gateway-pricing.ts": "packages/ai/scripts/generate_models.py",
}

#: pi's generated per-provider catalog stubs (`providers/<id>.models.ts`, one
#: `flatten*ModelCatalog` call per model type over `data/<id>.json`). pidrei
#: loads every vendored JSON in models_generated.py; there is no per-provider
#: module.
GENERATED_CATALOG_STUB_RE = re.compile(r"packages/ai/src/providers/[^/]+\.models\.ts")

#: pi file → diverged regions inside its pidrei mirror, as (recipe id, note)
#: pairs. The file still ports through the normal mapping, but a hunk landing
#: in the named region is translated per the recipe in spec/upstream-sync.md
#: ("Diverged regions"), not side-by-side. Same maintenance model as
#: RENAMES/TEST_HOMES: one hand-verified entry per divergence, added with its
#: recipe. Pattern-shaped divergences with no single upstream file (e.g.
#: recipe `cancel-token`) live only in the spec.
_FREEZE_ADAPTER_NOTE = (
    "streamed partial is built via producer-private builders "
    "(pidrei_ai/builders.py): construction sites use *Builder type names, "
    "mutation lines port verbatim; new message/block fields land in both "
    "frozen type and builder"
)

DIVERGED: dict[str, tuple[tuple[str, str], ...]] = {
    "packages/agent/src/agent.ts": (
        (
            "dispatch-observe",
            (
                "event observation runs on a per-run dispatcher task "
                "(_dispatch_events, with the PIDREI_DISPATCH_STALL_LOG meter); "
                "emit/listener diffs land around it"
            ),
        ),
        (
            "freeze-at-seam",
            (
                "messages are frozen values: streaming_message holds per-delta "
                "snapshots; upstream mutation of a message translates to "
                "constructing (dataclasses.replace)"
            ),
        ),
        (
            "agent-mailbox",
            (
                "queue/lifecycle state (PendingMessageQueue, activeRun, "
                "abort/signal/waitForIdle, 'already processing' admission) "
                "is owned by the standing _AgentMailbox actor task; "
                "hasQueuedMessages and peekQueuedMessages are awaited in pidrei"
            ),
        ),
        (
            "state-epochs",
            (
                "AgentState.messages is a rebind-only tuple: upstream push/splice "
                "on state.messages lands as a rebind; _reduce publishes (*old, msg)"
            ),
        ),
        (
            "nested-calls-channel",
            (
                "pidrei-only observe(event): nested tool events enter the run's "
                "dispatcher as observe-only tickets (no _reduce); dropped once "
                "the dispatcher closed (events_guard)"
            ),
        ),
    ),
    # Nested tool calls (0.99.1 port): a per-top-level-call channel folded by
    # the tool wrapper replaces pi's shared NestedCallRecorder/scopes map.
    "packages/agent/src/agent-loop.ts": (
        (
            "nested-calls-channel",
            (
                "the wrapper's fold rides on AgentToolResult (nested_calls/"
                "nested_usage), moved off the result before the hooks and "
                "tool_execution_end and combined onto the tool-result message "
                "in _create_tool_result_message (pi: message_start listener)"
            ),
        ),
    ),
    "packages/coding-agent/src/core/nested-tool-calls.ts": (
        (
            "nested-calls-channel",
            (
                "NestedCallRecorder is fold_nested_calls over NestedCallFeed "
                "messages; scopes/takeRecord/clear do not exist (NestedCallScope "
                "rides in the tool context); queueTail is a TonIO sync.Lock"
            ),
        ),
    ),
    "packages/coding-agent/src/core/tools/tool-definition-wrapper.ts": (
        (
            "nested-calls-channel",
            (
                "owner (no inherited scope) opens the feed and drains it when "
                "execute returns or raises with calls recorded; nested calls run "
                "through with_nested_scope bound copies as producers"
            ),
        ),
    ),
    # State epochs: writes to agent.state.messages
    # are rebinds; identity WeakMap/WeakSet become a run-scoped strong id() table.
    "packages/coding-agent/src/core/agent-session.ts": (
        (
            "state-epochs",
            (
                "agent.state.messages writes land as rebinds; the _entryIdsByMessage/"
                "_boundaryDispatchedMessages identity maps land in the run-scoped "
                "id() table (cleared per prompt-loop iteration) with the positional "
                "projection walk as fallback"
            ),
        ),
        (
            "state-epochs",
            (
                "_applyToolLoadout publishes (tools, hidden_declarations) as one "
                "_ToolLoadoutEpoch under _tool_loadout_guard (sync RLock, also "
                "around the registry rebind); readers pin one epoch"
            ),
        ),
        (
            "nested-calls-channel",
            (
                "_executeNestedToolCall/_handleAgentEvent: no takeRecord in the "
                "message_start listener; nested events go through agent.observe; "
                "the NestedToolCallRunner is created with the session"
            ),
        ),
    ),
    # Freeze at the seam: message/content types
    # are frozen dataclasses; producers build via builders and
    # AssistantMessageEventStream.push() freezes at publication.
    "packages/ai/src/types.ts": (
        (
            "freeze-at-seam",
            (
                "message/content/usage types are frozen dataclasses; a new or "
                "changed field lands in the frozen type AND its builder mirror "
                "(pidrei_ai/builders.py), including freeze()"
            ),
        ),
    ),
    "packages/ai/src/utils/event-stream.ts": (
        (
            "freeze-at-seam",
            (
                "AssistantMessageEventStream.push() is the publication seam "
                "(freezes event message payloads); `partial` is the "
                "producer-private builder and _abort handles both shapes; "
                "_time() sets duration_ms on the final message at the seam "
                "(a builder before freeze(), a frozen message replaced)"
            ),
        ),
    ),
    "packages/ai/src/api/anthropic-messages.ts": (
        ("freeze-at-seam", _FREEZE_ADAPTER_NOTE),
        (
            "anthropic-federation",
            (
                "the SDK's workload identity federation (token exchange and cache) "
                "is auth/anthropic_federation.py, applied per request by "
                "_PunkreqAnthropicClient"
            ),
        ),
    ),
    "packages/ai/src/api/openai-completions.ts": (("freeze-at-seam", _FREEZE_ADAPTER_NOTE),),
    "packages/ai/src/api/openai-responses.ts": (("freeze-at-seam", _FREEZE_ADAPTER_NOTE),),
    "packages/ai/src/api/openai-responses-shared.ts": (("freeze-at-seam", _FREEZE_ADAPTER_NOTE),),
    "packages/ai/src/api/azure-openai-responses.ts": (("freeze-at-seam", _FREEZE_ADAPTER_NOTE),),
    "packages/ai/src/api/openai-codex-responses.ts": (("freeze-at-seam", _FREEZE_ADAPTER_NOTE),),
    "packages/ai/src/api/google-generative-ai.ts": (("freeze-at-seam", _FREEZE_ADAPTER_NOTE),),
    "packages/ai/src/api/google-vertex.ts": (("freeze-at-seam", _FREEZE_ADAPTER_NOTE),),
    "packages/ai/src/api/mistral-conversations.ts": (("freeze-at-seam", _FREEZE_ADAPTER_NOTE),),
    "packages/ai/src/api/bedrock-converse-stream.ts": (("freeze-at-seam", _FREEZE_ADAPTER_NOTE),),
    "packages/ai/src/providers/faux.ts": (("freeze-at-seam", _FREEZE_ADAPTER_NOTE),),
    # The TUI island: passive UI state under one
    # reentrant lock, with input, render and terminal-event loops.
    "packages/tui/src/tui.ts": (
        (
            "tui-island",
            (
                "requestRender is a one-slot channel send to a render loop "
                "(no throttle; force rides in the request object); "
                "handleInput's stages run under the state lock in "
                "_route_input, terminal replies are split off by the reader "
                "(_consume_terminal_reply), key I/O uses "
                "finish_before_next_input"
            ),
        ),
        (
            "terminal-colors-loop",
            (
                "queryTerminalColors registers the query in a sync prefix and "
                "returns its applied Event; every colour report (settled, "
                "timeout partial, late reply) is sent under _query_lock to "
                "_terminal_events, whose consumer is the only caller of the "
                "on_terminal_colors listener"
            ),
        ),
    ),
    "packages/tui/src/terminal.ts": (
        (
            "tui-island",
            (
                "stdin is a read-ahead reader (_read_input) plus one consumer "
                "(_consume_input); parser deadlines are the reader's; the "
                "Kitty activation is queued in input order; start takes "
                "on_reply/on_error; stop drops queued items; program status "
                "(held status, support, pending query) is under _protocol_lock, "
                "the reader consumes the query's echo there"
            ),
        ),
        (
            "terminal-colors-loop",
            "only the DA1 owed to the Kitty query is swallowed; later DA1 replies go to on_reply",
        ),
    ),
    "packages/tui/src/stdin-buffer.ts": (
        (
            "tui-island",
            "a synchronous parser: the flush setTimeout is a deadline (deadline/expire) the reader expires",
        ),
    ),
    "packages/tui/src/tui-main-screen.ts": (
        (
            "tui-island",
            (
                "doRender splits into _compose_frame (tree walk, under the "
                "state lock) and _write_frame (diff and output, render lock "
                "only)"
            ),
        ),
    ),
    "packages/tui/src/tui-alt-screen.ts": (
        (
            "tui-island",
            (
                "doRender splits into _compose_frame (layout, published "
                "layout and visible screen, under the state lock) and "
                "_write_frame (diff and output, render lock only)"
            ),
        ),
    ),
    "packages/coding-agent/src/core/extensions/runner.ts": (
        (
            "tui-island",
            (
                "withUIPrompt opens the prompt at call time and returns the "
                "spawn handle of the wait; the prompt "
                "depth is guarded; noOpUIContext mirrors the TUI context's shape"
            ),
        ),
    ),
    "packages/coding-agent/src/modes/interactive/interactive-mode.ts": (
        (
            "tui-island",
            (
                "helpers mutate in place under `with self.ui.state_lock:` "
                "(diffs port 1:1 inside the hold); the session listener "
                "applies _handle_event under the lock; multi-await flows "
                "apply each stretch in one hold; sync-reached async handlers "
                "spawn their rest with _spawn_flow; extension UI (§10): ctx.ui "
                "setters sync under the lock, dialogs mount at call time and "
                "return a spawn handle, component factories are sync and get "
                "the guarded ExtensionTui"
            ),
        ),
        (
            "freeze-at-seam",
            (
                "messages are frozen values: pi mutations of a message (e.g. "
                "the message_end abort decoration) translate to display-only "
                "dataclasses.replace copies"
            ),
        ),
    ),
    "packages/coding-agent/src/modes/interactive/program-status-reporter.ts": (
        (
            "tui-island",
            (
                "called under the UI state lock and outside it (logins), so "
                "its state has its own thread lock, held while it writes to "
                "the terminal; lock order: UI state lock, reporter lock, "
                "terminal protocol lock"
            ),
        ),
    ),
    # System theme (0.99.1 port): theme applications are serialized; terminal
    # colour globals are guarded.
    "packages/coding-agent/src/modes/interactive/theme/theme-controller.ts": (
        (
            "terminal-colors-loop",
            (
                "every theme application (settings, selections, previews, "
                "instance sets, both terminal listeners) runs under _apply_lock; "
                "no TUI lock is held across its awaits"
            ),
        ),
    ),
    "packages/coding-agent/src/modes/interactive/theme/theme.ts": (
        (
            "terminal-colors-loop",
            (
                "terminal colour globals live under _theme_state_lock with a "
                "conftest guard; Theme.colors publishes one (terminal, colors) "
                "tuple"
            ),
        ),
    ),
    # Config epochs: config services publish
    # immutable snapshots swapped atomically; readers pin one attribute read.
    "packages/coding-agent/src/core/settings-manager.ts": (
        (
            "config-epochs",
            (
                "published scope dicts are immutable epochs: setter mutation "
                "lines run inside _update_global_settings/_set_global/"
                "_set_global_nested closures on a private copy; a new pi "
                "setter becomes a _set_global call (or closure for multi-key "
                "logic); compound getters pin one _settings read"
            ),
        ),
        ("no-telemetry", "no enableAnalytics/trackingId accessors"),
    ),
    "packages/coding-agent/src/core/auth-storage.ts": (
        (
            "config-epochs",
            (
                "the shared read state is an atomically-rebound "
                "_AuthFileSnapshot (data, revision) pair; readers pin "
                "state.snapshot — diffs touching the read cache land around "
                "_update_read_state/_read_latest_data"
            ),
        ),
    ),
    "packages/coding-agent/src/core/models-store.ts": (
        (
            "config-epochs",
            (
                "file read state is an atomically-rebound _ModelsFileSnapshot "
                "pair; the in-memory store swaps _entries on write instead of "
                "mutating in place"
            ),
        ),
    ),
    "packages/coding-agent/src/core/runtime-credentials.ts": (
        (
            "config-epochs",
            ("_overrides is an immutable dict swapped under the writer guard; readers take no lock"),
        ),
    ),
    "packages/coding-agent/src/core/model-runtime.ts": (
        (
            "config-epochs",
            (
                "composition inputs publish as _CompositionEpoch via "
                "_publish_composition (under _composition_guard); readers "
                "(get_error, get_registered_*, get_provider_auth_status, "
                "get_compatibility_request_config, _get_model_auth) pin one "
                "epoch; _prepare_request resolves the provider once and "
                "auths against it (_get_model_auth); virtual models live in "
                "the epoch (virtual_models, inner dicts replaced, never "
                "mutated) and resolve_model pins it"
            ),
        ),
    ),
    "packages/ai/src/models.ts": (
        (
            "config-epochs",
            (
                "the provider map swaps on write (readers lock-free); "
                "_apply_auth takes the caller-pinned provider and "
                "get_auth_for_provider is the pidrei-only pinned variant of "
                "get_auth"
            ),
        ),
    ),
    "packages/ai/src/models-store.ts": (
        (
            "config-epochs",
            ("the in-memory store swaps _entries on write instead of mutating in place"),
        ),
    ),
    # In-process subagent example (2026-08-28): pi spawns a `pi` child process
    # per task; pidrei runs each task as an in-process AgentSession.
    "packages/coding-agent/examples/extensions/subagent/index.ts": (
        (
            "subagent-inprocess",
            (
                "runSingleAgent and the subprocess plumbing are an in-process "
                "session runner (CLI flags become session options); result "
                "dicts carry status/errorMessage instead of exitCode/stderr; "
                "mode logic, params, and rendering port 1:1 on the renamed "
                "fields"
            ),
        ),
    ),
    # No telemetry (2026-09-25): pidrei never asks for an analytics opt-in.
    "packages/coding-agent/src/modes/interactive/components/first-time-setup.ts": (
        ("no-telemetry", "theme-only dialog; the analytics opt-in step is not ported"),
    ),
    "packages/coding-agent/test/first-time-setup.test.ts": (
        ("no-telemetry", "analytics-settings cases are not mirrored"),
    ),
    # Codemode scripts are Python on Monty (1.0.0.0): behaviour ports, the
    # JavaScript-shaped text and machinery are translated or dropped.
    "packages/codemode/src/runtime/prelude-source.ts": (
        (
            "codemode-python",
            (
                "the JS prelude is host-side Python (text/image/store/load, close-match errors) plus "
                "all_settled/has_tool fed into the session; error strings keep pi's wording minus JS syntax"
            ),
        ),
    ),
    "packages/codemode/src/runtime/host.ts": (
        (
            "codemode-python",
            (
                "the worker host is a Monty driver (feed/resume snapshots on the blocking pool); "
                "worker/wasm/interrupt changes are dropped, call/output/store/limit behaviour ports"
            ),
        ),
    ),
    "packages/codemode/src/declarations.ts": (
        (
            "codemode-python",
            "TypeScript declarations are Python stubs (TypedDicts, keyword-only async defs) that Monty must type-check",
        ),
    ),
    "packages/codemode/src/source.ts": (
        ("codemode-python", "`# @options` instead of `// @options`; messages say Python"),
    ),
    "packages/coding-agent/src/extensions/codemode/tool.ts": (
        (
            "codemode-python",
            (
                "description, snippet, guideline, describeScriptCall/describeOutput and the globals list "
                "are pidrei's Python text; catalog selection, loadout and modes port 1:1"
            ),
        ),
    ),
    "packages/coding-agent/src/extensions/codemode/execute.ts": (
        (
            "codemode-python",
            (
                "script API globals are snake_case with keyword options; model-call hints show Python "
                "dicts; nested rows, limiter, truncation, spill and store entries port 1:1; console items "
                "are print() runs, split back into lines in the <console_output> block"
            ),
        ),
    ),
    "packages/coding-agent/docs/codemode.md": (
        ("codemode-python", "the whole page is rewritten for Python scripts; behaviour text keeps pi's wording"),
    ),
    "packages/codemode/test/sandbox.test.ts": (
        ("codemode-python", "scripts translated to Python; JS-only cases dropped or re-specified (module docstring)"),
    ),
    "packages/codemode/test/declarations.test.ts": (
        ("codemode-python", "same structure, every expected declaration is a Python stub"),
    ),
    "packages/codemode/test/source.test.ts": (
        ("codemode-python", "`#` options line"),
    ),
    # Read-then-set of the active tools is one guarded update (1.0.0.0; the
    # API is pi.update_active_tools, documented in docs/extensions.md).
    "packages/coding-agent/src/extensions/tool-search/tool.ts": (
        (
            "update-active-tools",
            (
                "getActiveTools()+setActiveTools() is one update_active_tools(update): candidates, "
                "ranking and the new list are computed inside the update"
            ),
        ),
    ),
    # MCP (1.0.0.0): the client package's runtime layer (recipe `mcp-client`)
    # and the extension's (recipe `mcp-extension`).
    "packages/mcp/src/transports/transport.ts": (
        (
            "mcp-client",
            "delivery is one consumer per transport over a channel; send() returns a SendResult, taking its "
            "place in the outgoing stream when called (stdio, in-memory; HTTP exchanges have no order)",
        ),
    ),
    "packages/mcp/src/transports/in-memory.ts": (
        ("mcp-client", "messages go onto the peer's delivery channel; send() returns a SendResult"),
    ),
    "packages/mcp/src/transports/stdio.ts": (
        ("mcp-client", "send() returns a SendResult; the close ladder runs on tonio's process API"),
    ),
    "packages/mcp/src/transports/streamable-http.ts": (
        (
            "mcp-client",
            "no signal: requests are bounded by MCP_TIMEOUT and run under run_cancellable with a token the "
            "close fires (a head-pending request is cancelled), close spawns the session DELETE (1 s) and the "
            "closing of held responses and joins both; the DELETE reuses the last request's token",
        ),
    ),
    "packages/mcp/src/client.ts": (
        (
            "mcp-client",
            "state under one lock with a published view; pending entries settled once; an abandoned request "
            "is cleaned up synchronously; incoming requests run detached",
        ),
    ),
    "packages/mcp/src/oauth/provider.ts": (
        ("mcp-client", "state() reads and, when none is stored, writes in one locked step"),
    ),
    "packages/mcp/src/oauth/callback.ts": (
        ("mcp-client", "close() is synchronous: waiters settled, server and connections closed"),
    ),
    "packages/mcp/src/oauth/flow.ts": (
        (
            "mcp-client",
            (
                "the flow's signal is OAuthFlowOptions.cancel, a CancelToken; a wrapped fetch runs each request "
                "under run_cancellable, so new per-step signal parameters need no threading"
            ),
        ),
    ),
    "packages/coding-agent/src/extensions/mcp/index.ts": (
        (
            "mcp-extension",
            "state behind one lock (server list a tuple); `ready` is an Event; background connects detached; "
            "the manager changes its view through tui.apply; attempts are object() compared by identity, "
            "_run_in_background learns its operation's attempt through on_attempt; the session's "
            "AbortController is a CancelToken and tracked work registers Events shutdown joins",
        ),
        ("update-active-tools", "the read-then-set of the active tools is one update_active_tools(update)"),
    ),
    "packages/coding-agent/src/extensions/mcp/runtime.ts": (
        (
            "mcp-extension",
            "one published connection snapshot; the shared open runs detached and is joined through an Event; "
            "close() also ends a connect in flight; on_change is async; factories are async",
        ),
    ),
    "packages/coding-agent/src/extensions/mcp/oauth.ts": (
        (
            "mcp-extension",
            "the shared refresh runs detached and is joined through an Event; refresh requests are capped by "
            "the fetch's timeout_ms; the refresh lock is a renewing FileLock; settings() is async",
        ),
    ),
    "packages/coding-agent/src/extensions/mcp/config.ts": (
        ("mcp-extension", "reads through tonio.colored.fs; every mcp.json edit runs under a FileLock on the file"),
    ),
    "packages/coding-agent/src/extensions/mcp/log.ts": (
        ("mcp-extension", "one pool job per message, under a lock for the size bookkeeping and rotation"),
    ),
    "packages/coding-agent/src/extensions/mcp/ui.ts": (
        ("mcp-extension", "the view changes through tui.apply; key handlers settle the answer manage() awaits"),
    ),
    "packages/coding-agent/src/extensions/mcp/cli.ts": (
        (
            "mcp-extension",
            (
                "the pasted redirect URL is read with FdReader under run_cancellable; --timeout is a cancel token "
                "over the whole sign-in"
            ),
        ),
    ),
    "packages/tui/src/components/image.ts": (
        (
            "image-conversion",
            "Image takes its TUI first and converts through tui.image_conversions off the render path (async "
            "converter, detached conversion, cache on the TUI); no setImageTranscoder or module-level cache",
        ),
    ),
    "packages/coding-agent/src/utils/image-convert.ts": (
        (
            "image-conversion",
            "the transcoder is the async convert_image_to_png_base64 (Pillow on the blocking pool), set on the TUI "
            "by interactive mode; ensurePngTranscoder/loadPngTranscoder have no counterpart",
        ),
    ),
}

#: pi test files whose pidrei coverage is not a 1:1 mirror. Phase-1 `ai` tests
#: are organized by pidrei module, so several pi files map many-to-many; the
#: note names where the changed cases go. Entries marked PARITY GAP have a
#: ported production module but no mirrored tests — backfill when a commit
#: touches them.
TEST_HOMES = {
    # Pre-existing gaps surfaced by the 0.85.0 triage (0.85.0 delta port).
    "packages/ai/test/openai-completions-cache-control-format.test.ts": (
        "PARITY GAP: anthropic-style cache-marker placement for openai-completions has no standalone "
        "mirror (cache_control cases live in test_openai_completions.py); port deltas there"
    ),
    "packages/ai/test/tool-call-id-normalization.test.ts": (
        "PARITY GAP: cross-provider tool-call-id normalization has no standalone mirror (adapter "
        "suites cover it piecemeal); port deltas into the adapter test that owns the case"
    ),
    "packages/coding-agent/test/status-indicator.test.ts": (
        "home: packages/pidrei/tests/test_interactive_mode_status.py (working-indicator cases; verify per delta)"
    ),
    "packages/coding-agent/test/image-processing.test.ts": (
        "PARITY GAP: image conversion/EXIF cases unmirrored (Pillow's exif_transpose replaces pi's "
        "APP1 scanner, so orientation deltas are usually moot); port behaviour deltas into a new mirror"
    ),
    "packages/ai/test/env-api-keys.test.ts": "covered by packages/ai/tests/test_providers.py (+ test_registry.py; see its docstring)",
    "packages/ai/test/supports-xhigh.test.ts": "covered by packages/ai/tests/test_registry.py + test_models_generated.py (get_supported_thinking_levels)",
    "packages/ai/test/models-runtime.test.ts": "covered by packages/ai/tests/test_registry.py (models.ts ported as registry.py)",
    "packages/ai/test/provider-error-body-regression.test.ts": "PARITY GAP: per-adapter 403-body passthrough (4 cases) unmirrored — needs punkreq fault injection per adapter",
    "packages/ai/test/provider-error-body-passthrough.test.ts": (
        "PARITY GAP: openrouter-images 403-body passthrough (1 case) unmirrored, like provider-error-body-regression"
    ),
    "packages/ai/test/openai-responses-partial-json-cleanup.test.ts": "covered by packages/ai/tests/test_openai_responses.py",
    "packages/ai/test/openai-responses-terminal-event.test.ts": "covered by packages/ai/tests/test_openai_responses.py",
    "packages/ai/test/constrained-sampling.test.ts": (
        "partial mirror: test_constrained_sampling.py holds the 0.84.2 strict-schema cases "
        "and the 1.0.0 foreign-item-id replay case; "
        "the grammar/replay cases stay covered by adapter tests and the rest is a PARITY GAP"
    ),
    "packages/ai/test/openai-completions-tool-choice.test.ts": "PARITY GAP: tool_choice forwarding in openai_completions.py unmirrored",
    "packages/coding-agent/test/git-update.test.ts": "PARITY GAP: package_manager.py git update (force-push handling) unmirrored",
    "packages/coding-agent/test/suite/regressions/8237-node-sea-extension-loading.test.ts": (
        "not ported: guards jiti virtualModules vs filesystem aliases inside a Node SEA binary; "
        "pidrei extensions import the live modules from sys.path (loader.py docstring)"
    ),
    "packages/coding-agent/test/suite/agent-session-bash-persistence.test.ts": "partial mirror: test_agent_session_bash_persistence.py holds the 0.83.0 concurrency cases; the rest of the characterization suite is a PARITY GAP",
    "packages/coding-agent/test/suite/regressions/6647-compaction-retries-transient-stream-drop.test.ts": "PARITY GAP: compaction transient-retry regression unmirrored",
    "packages/coding-agent/test/suite/regressions/5943-session-start-notify.test.ts": "PARITY GAP: session_start transient-UI regression unmirrored",
    "packages/coding-agent/test/suite/agent-session-codemode.test.ts": (
        "home: packages/pidrei/tests/test_agent_session_codemode.py (the script cases, translated to Python "
        "per recipe codemode-python) + test_agent_session_tool_orchestration.py (the core cases: nested calls "
        "in parallel, hooks on nested calls, nested usage, structured content through the hooks, bash's "
        "structured result, hidden declarations left out of the system prompt, codemode and tool_search "
        "registered inactive)"
    ),
    "packages/coding-agent/test/sdk-skills.test.ts": "PARITY GAP: SDK-level skills flows unmirrored (skills.test.ts is mirrored as test_skills.py)",
    "packages/coding-agent/test/test-harness.ts": "pi test infra; pidrei equivalents are tests/harness.py + conftest.py — absorb deltas where ported tests need them",
    "packages/coding-agent/test/utilities.ts": "pi test infra; pidrei equivalents are tests/harness.py + conftest.py — absorb deltas where ported tests need them",
    "packages/coding-agent/test/test-network-env.ts": "pi test infra (PI_OFFLINE stub); pidrei conftest.py is hermetic via the PIDREI_OFFLINE equivalent",
    "packages/coding-agent/test/test-harness.test.ts": "pi test-infra self-tests, not mirrored",
    # 0.84.x additions (0.84.1 delta port).
    "packages/tui/test/overlay-non-capturing.test.ts": "covered by packages/tui/tests/test_tui_overlays.py (+ test_tui_focus.py for focus cases)",
    "packages/tui/test/overlay-options.test.ts": "covered by packages/tui/tests/test_tui_overlays.py (+ test_tui_focus.py for focus cases)",
    "packages/tui/test/overlay-short-content.test.ts": "covered by packages/tui/tests/test_tui_overlays.py",
    "packages/tui/test/regression-overlay-cjk-boundary.test.ts": "covered by packages/tui/tests/test_tui_overlays.py",
    "packages/tui/test/tui-overlay-style-leak.test.ts": "covered by packages/tui/tests/test_tui_overlays.py",
    "packages/tui/test/tui-cell-size-input.test.ts": "covered by packages/tui/tests/test_tui_queries.py",
    "packages/tui/test/tui-shrink.test.ts": "covered by packages/tui/tests/test_tui_render.py",
    "packages/tui/test/settings-list.test.ts": "PARITY GAP: components/settings_list.py ported, pre-existing cases unmirrored — new in-range cases port into a new test_settings_list.py",
    "packages/coding-agent/test/model-runtime-cloudflare-compat.test.ts": "covered by packages/pidrei/tests/test_model_registry.py + test_model_runtime.py",
    "packages/coding-agent/test/sdk-openrouter-attribution.test.ts": "covered by packages/pidrei/tests/test_provider_attribution.py",
    "packages/coding-agent/test/model-runtime-test-utils.ts": "pi test infra; pidrei equivalent is packages/pidrei/tests/model_runtime_helpers.py — absorb deltas where ported tests need them",
    "packages/ai/test/deferred-tools.test.ts": (
        "deleted upstream in 9e05370b (deferred tools replaced by transcript system messages); "
        "the test_deferred_tools.py partial mirror was removed with it"
    ),
    "packages/ai/test/openai-completions-prompt-cache.test.ts": (
        "partial mirror: test_openai_completions_prompt_cache.py holds the 0.87.1 session-affinity cases "
        "(OpenRouter default/opt-out, Baseten catalog); the rest of the prompt-cache suite is a PARITY GAP"
    ),
    "packages/ai/test/openai-completions-tool-result-images.test.ts": (
        "partial mirror: test_openai_completions_tool_result_images.py holds the 0.87.1 empty-text-part case; "
        "the tool-result image batching cases are a PARITY GAP"
    ),
    "packages/coding-agent/test/agent-session-dynamic-tools.test.ts": "PARITY GAP: dynamic tool registration flows unmirrored",
    "packages/coding-agent/test/edit-tool-no-full-redraw.test.ts": "PARITY GAP: edit-tool render regression unmirrored",
    "packages/coding-agent/test/rpc-prompt-response-semantics.test.ts": (
        "PARITY GAP: rpc prompt/response semantics suite unmirrored (incl. the 0.99.1 data.disposition cases; "
        "steer/follow_up 'queued' is covered in test_agent_session.py)"
    ),
    "packages/coding-agent/test/sdk-session-manager.test.ts": "PARITY GAP: SDK session-manager flows unmirrored",
    "packages/coding-agent/test/model-runtime-auth-options.test.ts": (
        "covered by packages/pidrei/tests/test_model_runtime.py (the ModelRuntime auth options cases)"
    ),
    "packages/coding-agent/test/model-runtime-modify-models-compat.test.ts": (
        "partial mirror: test_model_runtime_modify_models_compat.py holds the 1.0.0 native-provider "
        "registration case (#9962); the rest of the modifyModels compat suite is a PARITY GAP"
    ),
    "packages/coding-agent/test/suite/agent-session-prompt.test.ts": (
        "partial mirror: test_agent_session_prompt.py holds the 0.87.0 image-normalization case; the rest of "
        "the prompt characterization suite is a PARITY GAP"
    ),
    "packages/coding-agent/test/suite/regressions/5303-bash-output-truncation.test.ts": "PARITY GAP: bash output truncation regression unmirrored",
    "packages/coding-agent/test/suite/regressions/6999-models-json-hot-reload.test.ts": "PARITY GAP: models.json hot-reload regression unmirrored",
    "packages/coding-agent/test/http-dispatcher.test.ts": "PARITY GAP: core/http_config.py ported, dispatcher tests unmirrored",
    # 0.84.2 additions.
    "packages/coding-agent/test/tools-manager.test.ts": (
        "N/A: specs ensureTool's download-status callback; pidrei's ensure_tool is lookup-only "
        "(downloads deliberately unported — see utils/tools_manager.py docstring)"
    ),
    "packages/ai/test/lazy-module-load.test.ts": (
        "N/A: Node module-loader probe asserting provider SDKs stay unloaded at import — "
        "pidrei has no provider SDKs (native punkreq transports)"
    ),
    # 0.84.3 additions (0.84.3 delta port).
    "packages/coding-agent/test/session-manager/tree-traversal.test.ts": (
        "covered by packages/pidrei/tests/test_session_manager.py (tree-traversal cases "
        "live with the rest of the session-manager mirror)"
    ),
    "packages/coding-agent/test/config.test.ts": (
        "N/A: specs install-method detection and the self-update machinery, neither ported "
        "(no uv equivalent — see pidrei/config.py; config-slice decision)"
    ),
    "packages/coding-agent/test/package-distribution.test.ts": (
        "N/A: asserts package.json points its executables at the JS bundle and its library "
        "exports at modular dist output; pidrei ships wheels with no bundled interpreter"
    ),
    "packages/coding-agent/test/syntax-highlight.test.ts": (
        "partial mirror: the theme-level cases in test_syntax_highlight.py; the renderer "
        "block is N/A (engine-specific — it specs highlight.js grammar loading and HTML-span "
        "re-rendering; pidrei highlights with pygments, which has no grammar registry and "
        "emits no HTML)"
    ),
    # 0.84.4 additions.
    "packages/coding-agent/test/session-manager/file-operations.test.ts": (
        "covered by packages/pidrei/tests/test_session_manager.py (file-operations cases "
        "live with the rest of the session-manager mirror)"
    ),
    # 0.85.0 additions (0.85.0 delta port).
    "packages/coding-agent/test/suite/regressions/4167-thinking-toggle-pending-tool-render.test.ts": (
        "PARITY GAP: pending tool render surviving a thinking toggle (pi 755da309, pre-dates this "
        "sync) has no mirror; the sibling 8611 case lives in test_8611_thinking_toggle_pending_bash_output.py. "
        "In-range deltas only touch the fake-`this` fixture (maybeShowAssistantDiagnostics stub)"
    ),
    # 0.87.1 additions (0.87.1 delta port).
    "packages/coding-agent/test/clipboard-image-native-errors.test.ts": (
        "N/A: specs paste aborting on native clipboard helper errors; pidrei has no native helper and "
        "read_clipboard_image never raises (clipboard_image.py docstring)"
    ),
    "packages/ai/test/context-estimate.test.ts": (
        "covered by packages/ai/tests/test_estimate.py (+ test_simple_options.py for the buildBaseOptions clamp)"
    ),
    "packages/ai/test/openai-completions-empty-tools.test.ts": (
        "partial mirror: test_openai_completions.py holds tools:[]-with-history and max_tokens params, "
        "test_simple_options.py the context clamp, test_cloudflare_stream.py gateway auth/base URL; "
        "empty/undefined-tools omission, default maxTokens, Cloudflare /compat conservative fields, BYOK "
        "Authorization and Workers AI session-affinity headers are a PARITY GAP"
    ),
    "packages/ai/test/openai-completions-retry.test.ts": (
        "covered by packages/ai/tests/test_openai_completions.py (retry cases; the SDK-retry case is structural)"
    ),
    "packages/ai/test/openai-responses-empty-tool-result.test.ts": "covered by packages/ai/tests/test_openai_responses.py",
    "packages/ai/test/openai-responses-foreign-toolcall-id.test.ts": "covered by packages/ai/tests/test_openai_responses.py",
    "packages/ai/test/openai-responses-message-id.test.ts": "covered by packages/ai/tests/test_openai_responses.py",
    "packages/coding-agent/test/agent-session-concurrent.test.ts": (
        "covered by packages/pidrei/tests/test_agent_session.py (TestConcurrentPromptGuard); "
        "the extension-origin steering case is a PARITY GAP"
    ),
    "packages/coding-agent/test/agent-session-retry.test.ts": (
        "covered by packages/pidrei/tests/test_agent_session.py (TestRetry)"
    ),
    "packages/coding-agent/test/agent-session-stats.test.ts": (
        "covered by packages/pidrei/tests/test_agent_session.py (TestGetSessionStats)"
    ),
    "packages/coding-agent/test/image-process.test.ts": (
        "partial mirror: test_image_process.py holds the 0.87.1 GIF-signature cases; BMP->PNG conversion "
        "is covered by test_tool_result_images.py + test_tools.py; the rest is a PARITY GAP"
    ),
    "packages/coding-agent/test/image-resize-callers.test.ts": (
        "partial mirror: test_image_resize_callers.py holds the 0.87.1 model-resize-profile cases; the "
        "'Image omitted' resize fallbacks are a PARITY GAP"
    ),
    "packages/coding-agent/test/sdk-stream-options.test.ts": (
        "partial mirror: test_sdk_stream_options.py holds the 0.87.1 cache-warming scheduling cases "
        "and the 0.99.1 provider_stream_event case; "
        "sdk.py stream_fn option forwarding (timeout/websocket/provider-retry/transform_headers) is a PARITY GAP"
    ),
    "packages/coding-agent/test/suite/agent-session-queue.test.ts": (
        "partial mirror: test_agent_session_queue.py holds the 0.87.1 input-handler case; the rest of the "
        "session-level queue characterization is a PARITY GAP (steer/followUp-while-streaming in "
        "test_agent_session.py, agent-level queue semantics in agent/tests/test_agent.py)"
    ),
    "packages/coding-agent/test/suite/agent-session-retry-events.test.ts": (
        "partial mirror: test_agent_session.py (TestRetry) holds the transient/exhaust/delayed-message_end/"
        "tool-loop retry cases and the 0.87.1 abort-after-failed-retry case; retry-disabled, non-retryable, "
        "abortRetry and the event-order/delta/agent_end cases are a PARITY GAP"
    ),
    "packages/coding-agent/test/suite/regressions/1717-2113-agent-session-event-settlement.test.ts": (
        "covered by packages/pidrei/tests/test_agent_session.py (message_end order with slow handlers, "
        "tool_call after assistant settlement)"
    ),
    "packages/coding-agent/test/suite/regressions/2791-fswatch-error-crash.test.ts": (
        "PARITY GAP: theme watcher error survival unmirrored (polling utils/fs_watch; its error handler "
        "is exercised only via test_footer_data_provider.py)"
    ),
    "packages/coding-agent/test/suite/regressions/2860-replaced-session-context.test.ts": (
        "partial mirror: test_agent_session_runtime.py holds stale-ctx invalidation on rebind; "
        "withSession callbacks for new/fork/switch are a PARITY GAP"
    ),
    "packages/coding-agent/test/suite/regressions/3302-find-path-glob.test.ts": (
        "partial mirror: test_tools.py (TestFindTool) covers basename globs; path-based patterns are a PARITY GAP"
    ),
    "packages/coding-agent/test/suite/regressions/3303-find-nested-gitignore.test.ts": (
        "PARITY GAP: nested .gitignore scoping in find unmirrored (test_tools.py covers the root .gitignore only)"
    ),
    "packages/coding-agent/test/suite/regressions/6363-agent-settled-event.test.ts": (
        "covered by packages/pidrei/tests/test_agent_session_boundaries.py "
        "(test_marks_the_exhausted_retry_run_as_final and the agent_settled assertions around it)"
    ),
}

NOISE_BASENAMES = {
    "package.json",
    "package-lock.json",
    "npm-shrinkwrap.json",
    "CHANGELOG.md",
    "README.md",
    "tsconfig.json",
    "tsconfig.base.json",
    "tsconfig.build.json",
    "tsconfig.test.json",
    "vitest.config.ts",
    "vitest.base.ts",
    "vitest.harness.config.ts",
    "vitest.benchmark.config.ts",
    "biome.json",
    "test.sh",
    "mini-test.sh",
    "pi-test.sh",
    "pi-test.ps1",
    ".npmignore",
    ".gitignore",
    ".gitattributes",
    # pi's Nix flake (1.0.2): packaging of pi itself
    "flake.nix",
    "flake.lock",
}
NOISE_PREFIXES = (
    ".github/",
    # pi repo-local dogfood extensions/config, not product code
    ".pi/",
    "scripts/",
    "packages/coding-agent/install-lock/",
    # pi's Nix flake (1.0.2): the package expression and its model catalog pin
    "nix/",
    # pi-internal design docs, not user documentation
    "packages/agent/docs/",
)

#: Top-level manifests of ported packages: still noise for porting purposes,
#: but a new runtime dependency there needs a human decision, so they get
#: their own summary section.
DEPS_REVIEW_PATHS = {
    f"packages/{name}/package.json"
    for name in ("ai", "agent", "codemode", "coding-agent", "tui", "server", "protocol", "client")
}


GIT = shutil.which("git") or "git"


def git(pi_root: str, *args: str) -> str:
    result = subprocess.run(  # noqa: S603
        [GIT, "-C", pi_root, *args], check=True, capture_output=True, text=True
    )
    return result.stdout


def is_ancestor(pi_root: str, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(  # noqa: S603
        [GIT, "-C", pi_root, "merge-base", "--is-ancestor", ancestor, descendant],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def read(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def snake(segment: str) -> str:
    return segment.replace("-", "_")


def map_path(pi_path: str) -> tuple[str, str] | None:
    """Return (kind, pidrei_path) for a portable pi file, else None."""
    if pi_path in RENAMES:
        return "src", RENAMES[pi_path]
    if GENERATED_CATALOG_STUB_RE.fullmatch(pi_path):
        return "src", "packages/ai/pidrei_ai/models_generated.py"
    for prefix, target, kind in PREFIX_MAP:
        if not pi_path.startswith(prefix):
            continue
        rest = pi_path[len(prefix) :]
        if kind == "doc":
            return kind, target + rest
        if kind == "test":
            # pidrei test trees are flat; pi nests (test/suite/, regressions/).
            base = rest.rsplit("/", 1)[-1]
            if base.endswith(".test.ts"):
                return kind, target + "test_" + snake(base[: -len(".test.ts")]) + ".py"
            if base.endswith(".ts"):
                return kind, target + snake(base[: -len(".ts")]) + ".py"
            return kind, target + base
        if kind == "example":
            # Mirrored examples: dirs and .ts files snake-case, but non-code
            # assets (subagent prompts/*.md etc.) keep their upstream names.
            parts = rest.split("/")
            parts[:-1] = [snake(part) for part in parts[:-1]]
            if parts[-1] == "index.ts":
                parts[-1] = "__init__.py"
            elif parts[-1].endswith(".ts"):
                parts[-1] = snake(parts[-1][: -len(".ts")]) + ".py"
            return kind, target + "/".join(parts)
        parts = [snake(part) for part in rest.split("/")]
        if parts[-1] == "index.ts":
            # package/subpackage facade convention (may be a deliberately-empty
            # facade on the pidrei side: pidrei_agent, pidrei — judge per delta)
            parts[-1] = "__init__.py"
        elif parts[-1].endswith(".lazy.ts"):
            # deferred-import shims: pidrei names them <module>_lazy.py
            parts[-1] = parts[-1][: -len(".lazy.ts")] + "_lazy.py"
        elif parts[-1].endswith(".ts"):
            parts[-1] = parts[-1][: -len(".ts")] + ".py"
        return kind, target + "/".join(parts)
    return None


def classify(pi_path: str) -> tuple[str, object]:
    """Return (category, detail): portable -> (kind, target), else a reason."""
    for prefix, reason in DROPPED_PREFIXES:
        if pi_path.startswith(prefix):
            return "dropped", reason
    basename = pi_path.rsplit("/", 1)[-1]
    if DROPPED_BASENAME_RE.search(basename):
        return "dropped", DROPPED_BASENAME_REASON
    if basename in NOISE_BASENAMES or pi_path.startswith(NOISE_PREFIXES):
        return "noise", None
    if "/" not in pi_path and pi_path.endswith(".md"):
        return "noise", None
    mapped = map_path(pi_path)
    if mapped is not None:
        return "portable", mapped
    return "unmapped", None


def is_merge(pi_root: str, sha: str) -> bool:
    return len(git(pi_root, "rev-list", "--parents", "-n", "1", sha).split()) > 2


def commit_files(pi_root: str, sha: str) -> list[tuple[str, str]]:
    """[(status, path)] for a non-merge commit; renames/copies report the new path.

    Merges are never diffed: `diff-tree -m` emits one block per parent (even
    with --first-parent), and the block against the branch parent lists every
    mainline change since the fork point — hundreds of already-ported files.
    Every commit a merge brings to the mainline is either reachable from the
    ported ref (already ported) or inside ported_ref..HEAD (listed on its own
    line), so the merge itself carries nothing to port.
    """
    out = git(pi_root, "diff-tree", "-r", "--no-commit-id", "--name-status", sha)
    files = []
    for line in out.splitlines():
        fields = line.split("\t")
        status = fields[0][0]
        files.append((status, fields[-1]))
    return files


def marker(status: str, kind: str, target: str) -> str:
    exists = os.path.exists(os.path.join(ROOT, target))
    if status == "D":
        return "  [DELETE]" if exists else "  [already absent]"
    if exists:
        return ""
    if status == "A":
        return "  [NEW]"
    if kind == "doc":
        # pidrei ports a curated subset of pi's docs; a modified doc outside
        # the subset stays unported unless it became relevant.
        return "  [not in curated docs subset]"
    return "  [MISSING — pidrei rename? verify and extend RENAMES]"


def report(pi_root: str) -> int:
    ported_ref = read(REF_FILE).strip()
    head = git(pi_root, "rev-parse", "HEAD").strip()
    describe = git(pi_root, "describe", "--tags", head).strip()
    print(f"pi checkout : {pi_root} @ {head[:8]} ({describe})")
    print(f"ported ref  : {ported_ref[:8]} ({git(pi_root, 'describe', '--tags', ported_ref).strip()})")

    log = git(pi_root, "log", "--reverse", "--format=%H\t%s", f"{ported_ref}..HEAD")
    commits = [line.split("\t", 1) for line in log.splitlines()]
    print(f"{len(commits)} upstream commits\n")

    to_port = 0
    files_to_port = 0
    new_files = 0
    missing_on_modify: list[str] = []
    unmapped: list[str] = []
    deps_review: dict[str, list[str]] = {}

    for index, (sha, subject) in enumerate(commits, start=1):
        if is_merge(pi_root, sha):
            print(f"[{index:>2}/{len(commits)}] {sha[:8]}  {subject} — merge (constituent commits listed individually)")
            continue
        portable: list[tuple[str, str, str, str]] = []
        dropped_reasons: list[str] = []
        noise = 0
        for status, path in commit_files(pi_root, sha):
            category, detail = classify(path)
            if category == "portable":
                kind, target = detail
                portable.append((status, kind, path, target))
            elif category == "dropped":
                if detail not in dropped_reasons:
                    dropped_reasons.append(detail)
            elif category == "noise":
                noise += 1
                if path in DEPS_REVIEW_PATHS:
                    deps_review.setdefault(path, []).append(sha[:8])
            else:
                unmapped.append(f"{path}  ({sha[:8]} {subject})")

        head_line = f"[{index:>2}/{len(commits)}] {sha[:8]}  {subject}"
        if not portable:
            why = "; ".join(dropped_reasons) if dropped_reasons else "noise only"
            print(f"{head_line} — nothing to port ({why})")
            continue

        to_port += 1
        print(head_line)
        for status, kind, path, target in portable:
            files_to_port += 1
            print(f"    {status} {kind:<4} {path}")
            note = TEST_HOMES.get(path)
            if note is not None:
                print(f"             → {note}")
                continue
            mark = marker(status, kind, target)
            print(f"             → {target}{mark}")
            for recipe_id, region_note in DIVERGED.get(path, ()):
                print(f"             ⚠ diverged [{recipe_id}]: {region_note} — recipe in spec/upstream-sync.md")
            if mark == "  [NEW]":
                new_files += 1
            elif mark.startswith("  [MISSING"):
                missing_on_modify.append(f"{path} → {target}")
        for reason in dropped_reasons:
            print(f"      (also touches dropped surface: {reason})")

    print("\n== summary ==")
    print(
        f"{to_port} commits to port ({files_to_port} files, {new_files} new), "
        f"{len(commits) - to_port} with nothing to port"
    )
    if deps_review:
        print("package.json changed in ported packages — review for new runtime deps:")
        for path, shas in deps_review.items():
            print(f"  {path}  ({', '.join(shas)})")
    if missing_on_modify:
        print("modified upstream, but the mechanical pidrei target is missing — verify rename vs new file:")
        for entry in missing_on_modify:
            print(f"  {entry}")
    if unmapped:
        print("UNMAPPED — extend PREFIX_MAP / DROPPED / NOISE tables:")
        for entry in unmapped:
            print(f"  {entry}")
        return 2
    print("port in upstream order; after each commit lands:")
    print("  make upstream-bump REF=<sha>")
    return 0


def bump(pi_root: str, ref: str) -> int:
    sha = git(pi_root, "rev-parse", "--verify", f"{ref}^{{commit}}").strip()
    ported_ref = read(REF_FILE).strip()
    if not is_ancestor(pi_root, ported_ref, sha):
        print(f"refusing: {sha[:8]} is not a descendant of the ported ref {ported_ref[:8]}")
        return 1
    if not is_ancestor(pi_root, sha, "HEAD"):
        print(f"refusing: {sha[:8]} is not an ancestor of the pi checkout's HEAD")
        return 1

    with open(REF_FILE, "w", encoding="utf-8") as handle:
        handle.write(sha + "\n")
    source = read(UPSTREAM_PY)
    updated, count = re.subn(r'UPSTREAM_REF = "[0-9a-f]{40}"', f'UPSTREAM_REF = "{sha}"', source)
    if count != 1:
        print(f"could not find UPSTREAM_REF assignment in {UPSTREAM_PY}")
        return 1
    with open(UPSTREAM_PY, "w", encoding="utf-8") as handle:
        handle.write(updated)
    print(f"bumped .last_upstream_ref and upstream.py to {sha[:8]}")

    release = git(pi_root, "describe", "--tags", "--abbrev=0", sha).strip().lstrip("v")
    version = re.search(r'UPSTREAM_VERSION = "([^"]+)"', updated).group(1)
    if release != version:
        print(
            f"NOTE: pi released v{release} at or before this commit, but "
            f"UPSTREAM_VERSION is still {version} — bump it and the "
            f"package versions together (release_check gates this)."
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pi-root", default=os.environ.get("PI_ROOT"), help="path to the pi checkout (or set PI_ROOT)")
    parser.add_argument("--bump", metavar="SHA", help="record everything up to SHA as ported")
    args = parser.parse_args()
    if not args.pi_root:
        parser.error("pass --pi-root or set PI_ROOT")
    if not os.path.isdir(os.path.join(args.pi_root, ".git")):
        parser.error(f"{args.pi_root} is not a git checkout")
    if args.bump:
        return bump(args.pi_root, args.bump)
    return report(args.pi_root)


if __name__ == "__main__":
    sys.exit(main())
